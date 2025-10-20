import os
import torch
from pathlib import Path

from EnviromentSetup.Trainer.train_base import BaseTrainer
from DataHandler.dataset_builder import DatasetBuilder
from Helpers.configLoader import Config
from typing import Any
import time
import numpy as np
import csv
import json
from Framework.SelfCheck import SelfCheckManager
from Framework.SelfCheck.s5_deep_check.calibration import SafeThresholdCalibrator
from Framework.SelfCheck.s5_deep_check.deep_check_eval import DeepCheckManager
from Framework.SelfCheck.s5_deep_check.kg_eval import KGConsistencyEvaluator
from Framework.SelfCheck.s5_deep_check.signature_eval import SignatureEvaluator
from Framework.SelfCheck.s5_deep_check.activation_eval import ActivationOutlierDetector

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CKPT_BASELINE = BASE_DIR / "checkpoints" / "base_model"
ACT_BASELINE = BASE_DIR / "checkpoints" / "activation"
CKPT_BASELINE.mkdir(parents=True, exist_ok=True)
ACT_BASELINE.mkdir(parents=True, exist_ok=True)

class Server:
    def __init__(self, model_cls, 
                    config=None, self_check: Any = None, anchor_loader: Any = None,
                    checkpoint_dir="checkpoints/base_model", device="cpu", dataset_path=None,
                    text_col=None, label_col=None):
        
        self.model_cls = model_cls
        self.config = config if config else Config.load(BASE_DIR / "config" / "config.yaml")
        self.device = torch.device(device if isinstance(device, str) else device)
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else CKPT_BASELINE
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Dataset set up for base model
        self.dataset_path = BASE_DIR / "data" / "animal" / "base" / "base_model.csv" if dataset_path is None else dataset_path
        self.text_col = "text" if text_col is None else text_col
        self.label_col = "label" if label_col is None else label_col

        # load / train base model
        self.global_model, self.base_vocab, self.base_label2id = self._load_or_train_base()

        # canonical param key order used for flattening and aggregation
        self.param_key_order = list(self.global_model.state_dict().keys())

        self.reputation_store = {}   # client_id -> reputation score
        self.ledger_store = []       # list of ledger dict entries

        if self_check is not None:
            self.selfcheck =  self_check
        else:
            # Load anchors and KG if available
            if anchor_loader is None:
                print("[WARNING] [Server] No anchor loader provided.")

            kg_edges, label_map = [], {}
            entity_embeddings, entity2id = None, None

            try:
                kg_dir = BASE_DIR / "knowledge_graph" 

                # Load trained KG embeddings (if available)
                emb_path = kg_dir / "node_embeddings_best.npy"
                id_path = kg_dir / "node2id.json"
                if emb_path.exists() and id_path.exists():
                    print(f"[Server] Loading trained KG embeddings from {kg_dir}")
                    entity_embeddings = torch.tensor(
                        np.load(emb_path),
                        dtype=torch.float32,
                        device=self.device
                    )
                    with open(id_path, "r") as f:
                        entity2id = json.load(f)
                else:
                    print("[Server] No trained KG embeddings found — will fallback to symbolic KG only.")

                # Load triples (for structure-based consistency)
                triples_csv = kg_dir / "triples.csv"
                if triples_csv.exists():
                    with open(triples_csv, "r") as f:
                        reader = csv.reader(f)
                        next(reader)  # skip header
                        kg_edges = {(int(h), int(t)) for h, _, t in reader}
                    print(f"[Server] Loaded {len(kg_edges)} triples from KG.")

                # Build label map based on base model's label2id
                label_map = {v: k.lower() for k, v in self.base_label2id.items()}


            except Exception as e:
                print(f"[Server] Failed to load KG: {e}")

            # Hybrid KG evaluator: supports both symbolic triples + learned embeddings
            self.kg_eval = KGConsistencyEvaluator(
                kg_edges=kg_edges,
                label_map=label_map,
                entity_embeddings=entity_embeddings,
                entity2id=entity2id,
                alpha=0.5  # weighting factor between graph edges and embedding similarity
            )

            self.act = ActivationOutlierDetector(layer_name="encoder", max_samples=256, n_components=4)
            self.sig_eval = SignatureEvaluator(loss_type="cosine", use_salt=True, dp_sigma=0.0)

            self.deep_check = DeepCheckManager(
                anchor_loader=anchor_loader,
                kg_eval=self.kg_eval,
                activation_eval=self.act,
                sig_eval=self.sig_eval
            )

            # Load reference signatures
            try:
                self.ref_path = Path(__file__).resolve().parents[2] / "deepcheck_ledger" / "ref_sigs.pt"
                if self.ref_path.exists():
                    self.deep_check.load_ref_sigs(str(self.ref_path))
            except Exception as e:
                print(f"[Server] Warning: failed to load deep_check ref_sigs: {e}")

            self.selfcheck = SelfCheckManager(
                global_model=self.global_model,
                anchor_loader=anchor_loader,
                deep_check=self.deep_check
            )

        try:
            self.prepare_activation_baseline(
                baseline_path=ACT_BASELINE / "activation_baseline.npz",
                rebuild_if_missing=True,
                max_samples=512,
                percentile=99,
                method="percentile"
            )
        except Exception as e:
            print(f"[Server] Activation baseline setup failed: {e}")

        self.global_ref_sig = None       # torch.Tensor or None
        self.global_ref_alpha = 0.1      # EMA update rate for global ref sig
        self.min_trusted_for_global = 2  # update only if >=2 trusted contributors

        # --- Locate KG embeddings directory automatically ---
        self.kg_dir = BASE_DIR / "knowledge_graph"
        self.kg_emb_path = self.kg_dir / "node_embeddings_best.npy"
        self.kg_node2id_path = self.kg_dir / "node2id.json"

        if self.kg_emb_path.exists() and self.kg_node2id_path.exists():
            print(f"[Server] Using KG embeddings from {self.kg_dir}")
        else:
            print("[Server] No trained KG embeddings found — clients will skip KG alignment.")
            self.kg_dir = None
        self.calibrator = SafeThresholdCalibrator(self.selfcheck.deep_check)

        # Allowance for dynamically adjust
        if config is None:
            cfg = Config.load(BASE_DIR / "config" / "config.yaml")
        self.allow_dynamic_label_expansion = self.config.allow_dynamic_label_expansion
        if not isinstance(self.allow_dynamic_label_expansion, bool):
            self.allow_dynamic_label_expansion = cfg.allow_dynamic_label_expansion

        # Share label space globally
        self.share_label_space = self.config.share_label_space
        if not isinstance(self.share_label_space, bool):
            self.share_label_space = cfg.share_label_space

        print(f"[Server] Allow dynamic label expansion: {self.allow_dynamic_label_expansion}")
        print(f"[Server] Share label space globally: {self.share_label_space}")

        if not self.share_label_space:
            print("[Server][Privacy Mode] Label sharing is DISABLED. "
                "Clients’ new labels will be kept private and anonymized.")

    def _load_or_train_base(self):
        latest_ckpt = self._get_latest_checkpoint()

        train_ds, val_ds, test_ds, vocab, label2id = DatasetBuilder.build_dataset(
            path=self.dataset_path,
            max_len=self.config.max_seq_len,
            text_col=self.text_col,
            label_col=self.label_col
        )

        vocab_size = len(vocab)
        num_classes = len(label2id)
        model = self.model_cls(
            vocab_size=vocab_size,
            num_classes=num_classes,
            d_model=self.config.model_dim,
            nhead=self.config.num_heads,
            num_layers=self.config.num_layers,
            dim_ff=self.config.ffn_dim,
            max_len=self.config.max_seq_len,
            dropout=self.config.dropout
        ).to(self.device)

        if latest_ckpt is None:
            print("[Server] No base model found, training new base model...")
            trainer = BaseTrainer(
                model=model,
                train_dataset=train_ds,
                val_dataset=val_ds,
                test_dataset=test_ds,
                batch_size=self.config.batch_size,
                lr=self.config.lr,
                cfg=self.config,
                use_wandb=False,
                device=str(self.device)
            )
            trainer.train(epochs=self.config.epochs)
            model = trainer.model
            ckpt_path = self.checkpoint_dir / "epoch_final.pt"
            torch.save(model.state_dict(), ckpt_path)
            latest_ckpt = ckpt_path

        print(f"[Server] Loading base model from {latest_ckpt}")
        model.load_state_dict(torch.load(latest_ckpt, map_location=self.device))
        model.to(self.device)

        print("\n[DEBUG] Base model constructed")
        print(f"  Expected num_classes from dataset: {num_classes}")
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                w = module.weight.shape
                b = getattr(module, 'bias', None)
                print(f"  Linear layer: {name:25s} weight={tuple(w)} "
                    f"bias={None if b is None else tuple(b.shape)}")

        return model, vocab, label2id

    def prepare_activation_baseline(
        self,
        baseline_path: str = None,
        rebuild_if_missing: bool = True,
        max_samples: int = 512,
        percentile: int = 95,
        method: str = "percentile",
        layer_name: str = None
    ):
        """
        Ensure ActivationOutlierDetector has a baseline:
        - load baseline_path if present
        - otherwise compute zmax distribution on anchor_loader and set threshold
        """
        if self.selfcheck is None or self.selfcheck.anchor_loader is None:
            print("[Server] No anchor_loader available — cannot build activation baseline.")
            return False

        detector = self.act  # ActivationOutlierDetector instance
        baseline_path = baseline_path or (ACT_BASELINE / "activation_baseline.npz")

        # Try load existing baseline file first
        if baseline_path.exists():
            try:
                detector.load_baseline(str(baseline_path))
                print(f"[Server] Loaded activation baseline from {baseline_path}")
                return True
            except Exception as e:
                print(f"[Server] Failed to load baseline: {e}")
                if not rebuild_if_missing:
                    return False

        # Compute empirical zmax distribution (may take a few seconds)
        print("[Server] Computing activation zmax distribution for baseline...")
        dist = detector.compute_zmax_distribution(
            global_model=self.global_model,
            anchor_loader=self.selfcheck.anchor_loader,
            layer_name=layer_name or detector.layer_name,
            max_samples=max_samples
        )

        zarr = dist.get("zmax", None)
        if zarr is None or zarr.size == 0:
            print("[Server] Baseline zmax computation returned no data; baseline not created.")
            return False

        # Save baseline array (so future restarts can load quickly)
        np.savez(str(baseline_path), zmax=zarr)
        print(f"[Server] Saved activation baseline -> {baseline_path}")

        # Set threshold from that file (percentile or mad) and initialize ema_zmax
        detector.set_threshold_from_baseline(str(baseline_path), method=method, p=percentile)
        return True

    def get_kg_info(self):
        """Return KG embedding directory and availability flag."""
        if self.kg_dir and self.kg_emb_path.exists():
            return str(self.kg_dir), True
        return None, False

    def _get_latest_checkpoint(self):
        ckpts = list(self.checkpoint_dir.glob("*.pt"))
        if not ckpts:
            return None
        return max(ckpts, key=os.path.getctime)

    def weighted_fedavg(self, client_updates):
        """
        Trust-weighted FedAvg with shape alignment.
        Each item in client_updates must be:
            (state_dict, num_samples, trust_score)
        where trust_score is a float in [0,1].
        """
        server_sd = self.global_model.state_dict()
        new_state = {
            k: torch.zeros_like(v, device=self.device, dtype=v.dtype)
            for k, v in server_sd.items()
        }
        canonical_keys = list(server_sd.keys())

        if len(client_updates) == 0:
            raise ValueError("No client updates to aggregate")

        total_weight = float(sum(num * float(trust) for _, num, trust in client_updates))
        if total_weight <= 0:
            raise ValueError("All trust weights are zero — cannot aggregate.")

        first_state, _, _ = client_updates[0]
        new_state = {
            k: torch.zeros_like(v, device=self.device, dtype=v.dtype)
            for k, v in first_state.items()
        }

        for state, num_samples, trust_score in client_updates:
            w = (float(num_samples) * float(trust_score)) / (total_weight + 1e-12)
            for k in canonical_keys:
                v = state.get(k, None)
                target = new_state[k]
                if v is None:
                    # client didn't provide this param -> treat as zeros (no contribution)
                    continue
                v = v.to(self.device)
                # adjust flattened sizes deterministically
                if v.shape != target.shape:
                    flat_v = v.flatten()
                    if flat_v.numel() < target.numel():
                        pad = torch.zeros(target.numel() - flat_v.numel(), device=self.device, dtype=flat_v.dtype)
                        flat_v = torch.cat([flat_v, pad])
                    else:
                        flat_v = flat_v[:target.numel()]
                    v = flat_v.view_as(target)
                    print(f"[Aggregator] Adjusted param '{k}' -> {target.shape}")
                new_state[k] += v * w

        return new_state

    def aggregate_with_trust(self, client_updates):
        """
        Generic wrapper:
        - If each tuple has length 3 -> use weighted_fedavg
        - Else fallback to plain fedavg (expect list of (state_dict, num_samples))
        """
        if len(client_updates) == 0:
            raise ValueError("No client updates")
        if len(client_updates[0]) == 3:
            return self.weighted_fedavg(client_updates)
        # adapt to old format
        adapted = [(s, n, 1.0) for s, n in client_updates]
        return self.weighted_fedavg(adapted)

    def save_checkpoint(self, global_weights, round_num=None):
        """Save global model checkpoint."""
        if round_num:
            ckpt_path = self.checkpoint_dir / f"round{round_num}.pt"
        else:
            ckpt_path = self.checkpoint_dir / "final.pt"
        torch.save(global_weights, ckpt_path)
        print(f"[Server] Saved global model -> {ckpt_path}")

    def evaluate_global(self, dataset, batch_size=16):
        """Evaluate current global model on given dataset (returns accuracy float)."""
        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=batch_size)
        self.global_model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for ids, mask, y in loader:
                ids, mask, y = ids.to(self.device), mask.to(self.device), y.to(self.device)
                logits = self.global_model(ids, attention_mask=mask)
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        acc = float(correct / total) if total > 0 else 0.0
        print(f"[Server] Global Model Accuracy: {acc:.4f}")
        return acc
    
    def flatten_state_dict_to_tensor(self, state_dict, key_order=None):
        """Concatenate tensors in canonical order. Pad missing keys with zeros of expected shape."""
        key_order = key_order or self.param_key_order
        parts = []
        model_sd = self.global_model.state_dict()
        for k in key_order:
            expected = model_sd[k]
            v = state_dict.get(k, None)
            if v is None:
                parts.append(torch.zeros_like(expected).flatten())
            else:
                parts.append(v.detach().to(expected.device).flatten())
        return torch.cat(parts)

    def update_ledger(self, entry: dict):
        """
        Append a single trust/reputation entry to the server's in-memory ledger
        and persist it to disk (JSON file for transparency).
        """
        self.ledger_store.append(entry)

        # Define the ledger file path
        ledger_path = Path("checkpoints/ledger_log.json")
        ledger_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Load existing entries if file exists
            existing = []
            if ledger_path.exists():
                with open(ledger_path, "r") as f:
                    existing = json.load(f)

            existing.append(entry)
            with open(ledger_path, "w") as f:
                json.dump(existing, f, indent=2)

            print(f"[Ledger] Logged round {entry['round']} entry for client {entry['client_id']}")
        except Exception as e:
            print(f"[Ledger] Warning: failed to update ledger for {entry.get('client_id', '?')}: {e}")

    def safe_load_state_dict(self, model, state_dict):
        """
        Load model state dict safely, ignoring classifier mismatches
        when class count expands.
        """
        model_dict = model.state_dict()
        compatible = {}
        for k, v in state_dict.items():
            if k in model_dict and model_dict[k].shape == v.shape:
                compatible[k] = v.to(model_dict[k].device)
            else:
                print(f"[SafeLoad] Skipping {k}: shape {getattr(v, 'shape', None)}")
                print (f" expected {model_dict.get(k).shape if k in model_dict else 'MISSING'}")
        
        model.load_state_dict(compatible, strict=False)
        print(f"[SafeLoad] Loaded {len(compatible)} params (strict=False).")


        model_dict.update(compatible)
        model.load_state_dict(model_dict)
        print(f"[SafeLoad] Loaded {len(compatible)}/{len(model_dict)} params successfully.")

    def sync_labels_and_expand_model(self, client_label_sets):
        """
        Ensure the server's classifier head covers all labels seen by clients.
        Handles both orientations: [num_labels, hidden_dim] or [hidden_dim, num_labels].
        """
        current_labels = set(self.base_label2id.keys())
        new_labels = set().union(*client_label_sets) - current_labels

        if not new_labels:
            return

        print(f"[Server] New labels discovered from clients: {new_labels}")
        next_id = max(self.base_label2id.values()) + 1
        for label in sorted(new_labels):
            self.base_label2id[label] = next_id
            next_id += 1

        num_labels = len(self.base_label2id)

        # --- Find the classifier (last Linear layer) ---
        classifier = None
        for name, module in self.global_model.named_modules():
            if isinstance(module, torch.nn.Linear):
                classifier = module
        if classifier is None:
            print("[Server] Warning: no classifier layer found to expand.")
            return

        old_w = classifier.weight.data
        old_shape = old_w.shape
        print(f"[Server] Expanding classifier: old weight shape = {old_shape}")

        # --- Determine orientation more robustly ---
        # if first dim == old num_labels → row_labels orientation
        # if second dim == old num_labels → col_labels orientation
        old_num_labels = len(current_labels)
        if old_shape[0] == old_num_labels:
            orientation = "row_labels"
        elif old_shape[1] == old_num_labels:
            orientation = "col_labels"
        else:
            orientation = "row_labels"  # safe fallback

        if orientation == "row_labels":
            # expand along rows (num_labels, hidden_dim)
            old_num_labels, hidden_dim = old_shape
            new_w = torch.zeros((num_labels, hidden_dim), device=old_w.device, dtype=old_w.dtype)
            new_w[:old_num_labels, :] = old_w
            classifier.weight = torch.nn.Parameter(new_w)

            if classifier.bias is not None:
                old_b = classifier.bias.data
                new_b = torch.zeros((num_labels,), device=old_b.device, dtype=old_b.dtype)
                new_b[:old_num_labels] = old_b
                classifier.bias = torch.nn.Parameter(new_b)
            print(f"[Server] Expanded (row_labels) -> new shape {classifier.weight.shape}")

        else:
            # expand along columns (hidden_dim, num_labels)
            hidden_dim, old_num_labels = old_shape
            new_w = torch.zeros((hidden_dim, num_labels), device=old_w.device, dtype=old_w.dtype)
            new_w[:, :old_num_labels] = old_w
            classifier.weight = torch.nn.Parameter(new_w)

            if classifier.bias is not None:
                old_b = classifier.bias.data
                new_b = torch.zeros((num_labels,), device=old_b.device, dtype=old_b.dtype)
                new_b[:old_num_labels] = old_b
                classifier.bias = torch.nn.Parameter(new_b)
            print(f"[Server] Expanded (col_labels) -> new shape {classifier.weight.shape}")

    def run_round(self, round_id, client_updates):
        """
        Full FL round:
        SelfCheckManager -> DeepCheck -> Ledger -> Reputation -> Weighted Aggregation.
        - Uses staged candidate ref_sigs from DeepCheckManager and commits them
          atomically only for fully trusted clients after acceptance decisions.
        """
        print(f"\n[Server] --- Round {round_id} starting ---")

        # Reset activation EMA at start of round to avoid unwanted cross-round statefulness.
        try:
            self.deep_check.reset_all_ema()
        except Exception:
            # best-effort; continue even if not available
            pass

        client_label_sets = [set(cu.get("labels", [])) for cu in client_updates if "labels" in cu]

        if self.allow_dynamic_label_expansion and client_label_sets:
            # Expand classifier if allowed
            if self.share_label_space:
                self.sync_labels_and_expand_model(client_label_sets)
                print("[Server] Label space expanded globally and shared with all clients.")
            else:
                # Privacy-preserving mode — expand internally only
                private_label_union = set().union(*client_label_sets)
                current_labels = set(self.base_label2id.keys())
                unseen_private_labels = private_label_union - current_labels

                if unseen_private_labels:
                    print(f"[Server] [Privacy Mode] Detected {len(unseen_private_labels)} new private labels.")
                    # Expand classifier internally but DO NOT expose label names
                    self.sync_labels_and_expand_model([current_labels | unseen_private_labels])

                    # Optionally: record only anonymized info to ledger
                    self.update_ledger({
                        "round": round_id,
                        "privacy_mode": True,
                        "num_new_private_labels": len(unseen_private_labels),
                        "timestamp": time.time(),
                    })
        else:
            if client_label_sets:
                current_labels = set(self.base_label2id.keys())
                unseen = set().union(*client_label_sets) - current_labels
                if unseen:
                    print(f"[Server][Warning] Unseen labels detected but expansion disabled "
                        f"({len(unseen)} new labels withheld).")

        # Build input dict for SelfCheckManager / DeepCheckManager
        client_struct_updates = {}
        for cu in client_updates:
            cid = cu["client_id"]
            delta = cu.get("delta", {}) or {}

            # Ensure each param is a float tensor
            param_dict = {}
            for k, v in delta.items():
                param_dict[k] = (
                    v.detach().cpu().float()
                    if torch.is_tensor(v)
                    else torch.tensor(v, dtype=torch.float32)
                )

            client_struct_updates[cid] = param_dict

        # Precompute client signatures (client-side provided or server-side encode)
        client_sigs = {}
        for cu in client_updates:
            cid = cu["client_id"]
            delta = cu.get("delta", {}) or {}
            norm_delta = self.deep_check.normalize_client_delta(delta)
            try:
                client_sigs[cid] = self.deep_check.sig_eval.encode(norm_delta, dim=256, device="cpu")
            except Exception:
                try:
                    client_sigs[cid] = self.deep_check.sig_eval.make_signature_from_delta(norm_delta, dim=256, device="cpu")
                except Exception:
                    client_sigs[cid] = None

        # Build ref_sigs to send (prefer stored per-client, fallback to global ref, then client sig)
        ref_sigs_to_send = {}
        for cid in client_sigs:
            if cid in self.deep_check.ref_sigs:
                ref_sigs_to_send[cid] = self.deep_check.ref_sigs[cid]
            elif self.global_ref_sig is not None:
                ref_sigs_to_send[cid] = self.global_ref_sig
            else:
                ref_sigs_to_send[cid] = client_sigs[cid]

        # Run the full self-check pipeline (privacy-safe)
        public_out = self.selfcheck.run_round(
            client_updates=client_struct_updates,
            round_id=round_id,
            global_model=self.global_model,
            anchor_loader=self.selfcheck.anchor_loader,
            client_sigs=client_sigs,
            ref_sigs=ref_sigs_to_send,
        )

        # canonicalize accepted / downweighted / rejected lists
        accepted_client_ids = public_out.get("accepted", []) or []
        downweighted = public_out.get("downweighted", []) or []
        trust_scores = public_out.get("trust_scores_quantized", {}) if "trust_scores_quantized" in public_out else {}
        # derive lists if quantized scores provided
        if trust_scores:
            accepted = [cid for cid, t in trust_scores.items() if t and float(t) > 0.0]
            rejected = [cid for cid, t in trust_scores.items() if not t or float(t) == 0.0]
        else:
            accepted = list(accepted_client_ids)
            all_cids = [cu["client_id"] for cu in client_updates]
            rejected = [cid for cid in all_cids if cid not in accepted and cid not in downweighted]

        # ----- IMPORTANT: collect deep-check results separately to obtain candidate_ref_sig -----
        # We call deep_check.run_batch to extract per-client candidate_ref_sig and debug metrics
        try:
            deep_results = self.deep_check.run_batch(
                global_model=self.global_model,
                client_deltas=client_struct_updates,
                client_sigs=client_sigs,
                ref_sigs=ref_sigs_to_send,
                anchor_loader=self.selfcheck.anchor_loader,
            )
        except Exception as e:
            print(f"[Server] Warning: deep_check.run_batch failed: {e}")
            deep_results = {}

        # Debug print L_sig / L_max / S_sig / S_activation per-client (C)
        candidate_ref_pool = {}
        for cid, res in deep_results.items():
            if cid == "_summary":
                continue
            try:
                L_sig = res.get("L_sig", None)
                L_max = res.get("L_max", None)
                S_sig = res.get("S_sig", None)
                S_activation = res.get("S_activation", None)
                print(f"[DeepCheck DBG] {cid}: L_sig={L_sig} L_max={L_max} S_sig={S_sig} S_act={S_activation}")
                # collect candidate ref sigs (staged by deep_check as 'candidate_ref_sig')
                if res.get("candidate_ref_ok"):
                    candidate_ref_pool[cid] = res.get("candidate_ref_sig")
            except Exception:
                pass

        # Determine which candidate refs to commit:
        # Only commit refs for clients that were accepted (fully trusted).
        accepted_candidates = {cid: candidate_ref_pool[cid] for cid in accepted if cid in candidate_ref_pool}

        if accepted_candidates:
            try:
                committed_ok = self.deep_check.commit_ref_sigs(accepted_candidates, path=str(self.ref_path))
                if committed_ok:
                    print(f"[Server] Committed {len(accepted_candidates)} ref_sigs for accepted clients.")
                else:
                    print("[Server] Warning: commit_ref_sigs returned False.")
            except Exception as e:
                print(f"[Server] Warning: failed to commit ref_sigs: {e}")
        else:
            print("[Server] No candidate ref_sigs to commit for accepted clients this round.")

        # Update global_ref_sig from trusted_sigs (same logic as before)
        trusted_sigs = [client_sigs[cid] for cid in accepted if cid in client_sigs and client_sigs[cid] is not None]
        if len(trusted_sigs) >= self.min_trusted_for_global:
            mean_sig = torch.stack(trusted_sigs, dim=0).mean(dim=0)
            if self.global_ref_sig is None:
                self.global_ref_sig = mean_sig.detach().cpu().clone()
            else:
                alpha = float(self.global_ref_alpha)
                self.global_ref_sig = (alpha * mean_sig.detach().cpu()) + ((1.0 - alpha) * self.global_ref_sig)
            print(f"[Server] Updated global_ref_sig from {len(trusted_sigs)} trusted clients")

        # Atomically save global_ref_sig if present (do not overwrite ref_sigs here)
        try:
            if self.global_ref_sig is not None:
                tmpg = str(self.ref_path.with_name("global_ref_sig.pt")) + ".tmp"
                torch.save(self.global_ref_sig.detach().cpu(), tmpg)
                os.replace(tmpg, str(self.ref_path.with_name("global_ref_sig.pt")))
        except Exception as e:
            print(f"[Server] Warning: failed to save global_ref_sig: {e}")

        # Build trust_scores mapping (fallback)
        if "trust_scores_quantized" in public_out:
            trust_scores = public_out["trust_scores_quantized"]
        else:
            # default to 1.0 accepted, 0.5 downweighted, 0.0 rejected
            trust_scores = {}
            for cu in client_updates:
                cid = cu["client_id"]
                if cid in accepted:
                    trust_scores[cid] = 1.0
                elif cid in downweighted:
                    trust_scores[cid] = 0.5
                else:
                    trust_scores[cid] = 0.0

        # Update ledger & reputation for every client
        for cu in client_updates:
            cid = cu["client_id"]
            trust = float(trust_scores.get(cid, 1.0))
            old_rep = self.reputation_store.get(cid, 0.5)
            new_rep = 0.9 * old_rep + 0.1 * trust
            self.reputation_store[cid] = new_rep

            entry = {
                "round": round_id,
                "client_id": cid,
                "trust": trust,
                "reputation": new_rep,
                "timestamp": time.time(),
            }
            self.update_ledger(entry)

        # Aggregate weighted by trust
        agg_inputs = []
        for cu in client_updates:
            cid = cu["client_id"]
            agg_inputs.append((
                cu["state_dict"],
                cu.get("num_samples", 1),
                trust_scores.get(cid, 1.0)
            ))

        new_global = self.aggregate_with_trust(agg_inputs)
        self.safe_load_state_dict(self.global_model, new_global)
        self.save_checkpoint(new_global, round_num=round_id)

        print(f"[Server] Round {round_id} completed | Aggregated {len(agg_inputs)} clients.")
        return public_out
