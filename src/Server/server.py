import os
import torch
from pathlib import Path

from EnviromentSetup.Trainer.train_base import BaseTrainer
from DataHandler.dataset_builder import DatasetBuilder
from Helpers.configLoader import Config
from typing import Any
import time
import json
from Framework.SelfCheck import SelfCheckManager
from Framework.SelfCheck.s5_deep_check.calibration import SafeThresholdCalibrator

BASE_DIR = Path(__file__).resolve().parent.parent.parent

class Server:
    def __init__(self, model_cls, 
                    config=None, self_check: Any = None, anchor_loader: Any = None,
                    checkpoint_dir="checkpoints/base_model", device="cpu"):
        self.model_cls = model_cls
        self.config = config if config else Config.load(BASE_DIR / "config" / "config.yaml")
        self.device = torch.device(device if isinstance(device, str) else device)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # load / train base model
        self.global_model = self._load_or_train_base()

        self.reputation_store = {}   # client_id -> reputation score
        self.ledger_store = []       # list of ledger dict entries

        if self_check is not None:
            self.selfcheck =  self_check
        else:
            if anchor_loader is None:
                print("[WARNING] [Server] No anchor loader provided.")
            self.selfcheck = SelfCheckManager(
                global_model=self.global_model,
                anchor_loader=anchor_loader
            )
        self.calibrator = SafeThresholdCalibrator(self.selfcheck.deep_check)

    def _load_or_train_base(self):
        latest_ckpt = self._get_latest_checkpoint()

        # build base dataset so we know vocab_size / num_classes
        dataset_path = Path("data/medical_data/base/base_model.csv")
        train_ds, val_ds, test_ds, vocab, label2id = DatasetBuilder.build_dataset(
            path=dataset_path,
            max_len=self.config.max_seq_len
        )
        vocab_size = len(vocab)
        num_classes = len(label2id)

        # construct model with explicit args from config
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
            model = trainer.model  # trained model instance
            ckpt_path = self.checkpoint_dir / "epoch_final.pt"
            torch.save(model.state_dict(), ckpt_path)
            latest_ckpt = ckpt_path

        print(f"[Server] Loading base model from {latest_ckpt}")
        model.load_state_dict(torch.load(latest_ckpt, map_location=self.device))
        model.to(self.device)
        return model

    def _get_latest_checkpoint(self):
        ckpts = list(self.checkpoint_dir.glob("*.pt"))
        if not ckpts:
            return None
        return max(ckpts, key=os.path.getctime)

    def weighted_fedavg(self, client_updates):
        """
        Trust-weighted FedAvg.
        Each item in client_updates must be:
            (state_dict, num_samples, trust_score)
        where trust_score is a float in [0,1].
        """
        if len(client_updates) == 0:
            raise ValueError("No client updates to aggregate")

        total_weight = float(sum(num * float(trust) for _, num, trust in client_updates))
        if total_weight <= 0:
            raise ValueError("All trust weights are zero — cannot aggregate.")

        first_state, _, _ = client_updates[0]
        new_state = {k: torch.zeros_like(v, device=self.device, dtype=v.dtype)
                     for k, v in first_state.items()}

        for state, num_samples, trust_score in client_updates:
            w = (float(num_samples) * float(trust_score)) / (total_weight + 1e-12)
            for k, v in state.items():
                new_state[k] += v.to(self.device) * w

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

    def run_round(self, round_id, client_updates):
        """
        Full FL round:
        SelfCheckManager -> DeepCheck -> Ledger -> Reputation -> Weighted Aggregation.
        """
        print(f"\n[Server] --- Round {round_id} starting ---")

        # Step 1: Run the SelfCheck pipeline (calls DeepCheck internally)
        result = self.selfcheck.run_round(
            fake_client_updates={cu["client_id"]: cu["delta"] for cu in client_updates},
            round_id=round_id,
            global_model=self.global_model,
            anchor_loader=self.selfcheck.anchor_loader,
            client_sigs={cu["client_id"]: cu.get("client_sig") for cu in client_updates},
            ref_sigs={cu["client_id"]: cu.get("ref_sig") for cu in client_updates},
        )

        deep_results = result.get("stage4_deepcheck", {})
        final_trust_scores = {}

        # Step 2: Derive trust scores and update reputation
        for cu in client_updates:
            cid = cu["client_id"]
            S_final = float(deep_results.get(cid, {}).get("S_final", 1.0))
            old_rep = self.reputation_store.get(cid, 0.5)
            new_rep = 0.9 * old_rep + 0.1 * S_final
            self.reputation_store[cid] = new_rep
            final_trust = 0.7 * S_final + 0.3 * new_rep
            final_trust_scores[cid] = final_trust

            # Log ledger entry
            entry = {
                "round": round_id,
                "client_id": cid,
                "S_final": S_final,
                "reputation": new_rep,
                "final_trust": final_trust,
                "timestamp": time.time(),
            }
            self.update_ledger(entry)

            if round_id % self.config.calib_interval == 0:
                high_rep_clients = {cid: cu["delta"] for cu in client_updates
                                    if self.reputation_store.get(cid, 0.5) > 0.7}
                if len(high_rep_clients) >= 5:
                    print(f"[Server] Running SafeThresholdCalibrator on {len(high_rep_clients)} trusted clients...")
                    self.calibrator.calibrate(self.global_model, high_rep_clients, anchor_loader=self.selfcheck.anchor_loader)

        # Step 3: Weighted aggregation using trust
        agg_inputs = [
            (cu["state_dict"], cu.get("num_samples", 1), final_trust_scores.get(cu["client_id"], 1.0))
            for cu in client_updates
        ]
        new_global = self.aggregate_with_trust(agg_inputs)
        self.global_model.load_state_dict(new_global)
        self.save_checkpoint(new_global, round_num=round_id)

        print(f"[Server] Round {round_id} completed | Aggregated {len(client_updates)} clients.")
        return final_trust_scores

