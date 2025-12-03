import json
import csv
import torch
import numpy as np
import copy
import random
from pathlib import Path
from torch.utils.data import DataLoader
import datetime
import pandas as pd
from typing import Dict, Tuple, Any, List

from Server.server import Server
from Client.client import Client
from DataHandler.dataset_builder import DatasetBuilder
from Helpers.configLoader import Config
from Helpers.configRunLoader import ConfigRun
from EnviromentSetup.model.model import ToyBERTClassifier
from EnviromentSetup.corrupt.corruptSetup import ExperimentConfig, AttackEngines, safe_param_subtract
from Helpers.Helpers import _device_from_state_dict, numpy_delta_to_torch, torch_delta_to_numpy, toy_dataset_to_df, df_to_toy_dataset

def main():
    BASE_DIR = Path(__file__).resolve().parent.parent
    cfg = Config.load(BASE_DIR / "config" / "config.yaml")
    run_cfg = ConfigRun.load(BASE_DIR / "config" / "config_run.yaml")
    attacker_ids = set(run_cfg.attacker_ids)

    # Set seeds
    random.seed(run_cfg.seed)
    np.random.seed(run_cfg.seed)
    torch.manual_seed(run_cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(run_cfg.seed)

    # Build experiment config & validate
    exp = ExperimentConfig(run_cfg)
    print(f"[ConfigRun] {exp.summary()} attackers={set(run_cfg.attacker_ids)}")

    # Build attack engines
    engines = AttackEngines(run_cfg)

    # Build dataset & server
    train_base, val_base, test_base, vocab_base, label2id_base = DatasetBuilder.build_dataset(
        path=BASE_DIR / "data" / "animal" / "base" / "base_model.csv",
        max_len=cfg.max_seq_len,
        text_col="Information",
        label_col="Group"
    )

    anchor_loader = DataLoader(train_base, batch_size=cfg.batch_size, shuffle=False)

    server = Server(
        model_cls=ToyBERTClassifier,
        config=cfg,
        device="cuda" if torch.cuda.is_available() else "cpu",
        text_col="Information",
        label_col="Group",
        anchor_loader=anchor_loader,
        checkpoint_dir="checkpoints/base_model"
    )

    kg_dir, has_kg = server.get_kg_info()
    print(f"[Main] KG embeddings loaded: {has_kg}")

    # Build clients
    client_paths = [
        BASE_DIR / "data" / "animal" / f"n{i}" / f"client_{i}_data.csv"
        for i in range(1, 4)
    ]

    clients = []
    for i, path in enumerate(client_paths):
        print(f"[ClientSetup] Loading client {i+1}")

        train_ds, val_ds, test_ds, vocab, label2id = DatasetBuilder.build_dataset(
            path=path,
            max_len=cfg.max_seq_len,
            vocab=vocab_base,
            label2id=label2id_base.copy(),
            text_col="Information",
            label_col="Group"
        )

        vocab_size = train_ds.vocab_size
        num_classes = len(label2id)

        def make_model_fn(vs=vocab_size, nc=num_classes, c=cfg):
            return lambda: ToyBERTClassifier(
                vocab_size=vs, num_classes=nc,
                d_model=c.model_dim, nhead=c.num_heads,
                num_layers=c.num_layers, dim_ff=c.ffn_dim,
                max_len=c.max_seq_len, dropout=c.dropout
            )
        
        # ---- create client object ----
        client_obj = Client(
            client_id=f"client_{i+1}",
            model_fn=make_model_fn(),
            dataset=train_ds,
            device="cuda" if torch.cuda.is_available() else "cpu",
            kg_dir=kg_dir,
            use_kg_align=has_kg
        )

        # ---- attach MC-GRAD engine if needed ----
        if run_cfg.mc_grad_train and f"client_{i+1}" in attacker_ids:
            client_obj.mc_grad_engine = engines.mc_grad_engine
        else:
            client_obj.mc_grad_engine = None

        # ---- store client ----
        clients.append({
            "id": f"client_{i+1}",
            "label2id": label2id,
            "client": client_obj,
            "val": val_ds,
            "test": test_ds
        })

    # Global test set
    _, _, global_test_ds, _, _ = DatasetBuilder.build_dataset(
        path=BASE_DIR / "data" / "animal" / "base" / "base_model.csv",
        max_len=cfg.max_seq_len,
        text_col="Information",
        label_col="Group"
    )

    # Logging setup
    log_dir = BASE_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    json_path = log_dir / "accuracy_log.json"
    csv_path = log_dir / "accuracy_log.csv"
    if not json_path.exists():
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump([], f, indent=2)

    def write_csv_header(num_clients):
        cols = ["round", "timestamp", "global_acc"]
        for i in range(1, num_clients + 1):
            cols += [f"client_{i}_acc", f"client_{i}_samples"]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(cols)

    write_csv_header(len(clients))

    # Federated Learning Loop preparation
    global_weights = server.global_model.state_dict()
    num_rounds = run_cfg.num_rounds
    local_epochs = run_cfg.local_epochs

    previous_updates_for_mc_grad: List[Dict[str, np.ndarray]] = []
    previous_sybil_updates: List[List[Dict[str, np.ndarray]]] = []

    # Helper for applying data attacks (returns modified dataset, fake_num_samples)
    def apply_data_attacks(client_dataset, client_id) -> Tuple[Any, Any]:
        """
        Input:
        - client_dataset: ToyTextDataset
        - client_id: str
        Returns:
        - ds: dataset in the same type as client_dataset (ToyTextDataset)
        - fake_num_samples: int or None (if attack changed num samples)
        """
        fake_num_samples = None
        ds = client_dataset  # keep original type; we'll return this type

        # ---------- MC-DATA ----------
        if exp.is_mc and exp.data_mode and client_id in attacker_ids:
            try:
                # convert ToyTextDataset -> DataFrame (if it's already a df, use it)
                src_df = getattr(ds, "df", None)
                if isinstance(src_df, pd.DataFrame):
                    df = src_df.copy()
                else:
                    df = toy_dataset_to_df(ds)

                # prepare mc instance and attach df
                mc_instance = copy.deepcopy(engines.mc_data_template)
                mc_instance.df = df.copy()

                # perform chosen MC attack
                if exp.mc_attack_type == "random_label_flip":
                    mc_instance.random_label_flip()
                elif exp.mc_attack_type == "random_text_noise":
                    mc_instance.random_text_noise()
                elif exp.mc_attack_type == "semantic_noise":
                    mc_instance.semantic_noise()
                elif exp.mc_attack_type == "backdoor":
                    mc_instance.add_backdoor_trigger()
                elif exp.mc_attack_type == "duplicate_flood":
                    mc_instance.duplicate_flood()
                elif exp.mc_attack_type == "ood":
                    mc_instance.ood_injection()
                elif exp.mc_attack_type == "targeted_flip":
                    mc_instance.targeted_label_flip(
                        src_label=run_cfg.src_label,
                        tgt_label=run_cfg.tgt_label
                    )
                else:
                    raise ValueError(f"Unknown mc_attack_type={exp.mc_attack_type}")

                # get corrupted df and convert back to ToyTextDataset
                corrupted_df = mc_instance.get_corrupted_dataset()
                ds = df_to_toy_dataset(corrupted_df, client_dataset)
                fake_num_samples = len(corrupted_df)
                print(f"[ATTACK][MC-DATA] {client_id} applied ({exp.mc_attack_type})")
            except Exception as e:
                print(f"[WARN] MC-DATA {client_id}: {e}")

        # ---------- FREE-RIDER DATA ----------
        if exp.is_fr and exp.data_mode and client_id in attacker_ids:
            try:
                # ensure we have a DataFrame to hand to the FR engine
                src_df = getattr(ds, "df", None)
                if isinstance(src_df, pd.DataFrame):
                    df_for_fr = src_df.copy()
                else:
                    df_for_fr = toy_dataset_to_df(ds)

                # The FR engine may return (modified_df, meta) or just modified_df.
                # Adapt depending on your engine's return shape.
                fr_out = engines.free_rider_data_engine.apply(
                    df_for_fr,
                    metadata={"num_samples": len(df_for_fr)}
                )

                # handle different return formats
                if isinstance(fr_out, tuple) and len(fr_out) == 2:
                    modified_df, meta = fr_out
                else:
                    modified_df = fr_out
                    meta = {}

                ds = df_to_toy_dataset(modified_df, client_dataset)
                fake_num_samples = meta.get("num_samples", fake_num_samples)
                print(f"[ATTACK][FR-DATA] {client_id} mode={engines.free_rider_data_engine.mode}")
            except Exception as e:
                print(f"[WARN] FR-DATA {client_id}: {e}")

        return ds, fake_num_samples

    # Helper for applying gradient attacks (input: delta_np dict of numpy arrays)
    def apply_grad_attacks(delta_np: Dict[str, np.ndarray], client_id, num_samples, prev_updates) -> Tuple[Dict[str, np.ndarray], int]:
        # Make a safe deep-copy of the delta (force same dtype as input arrays)
        d_np = {k: np.array(v, copy=True) for k, v in delta_np.items()}

        # small helper for validating attack output
        def _validate_and_convert(out_dict, src_keys):
            if out_dict is None:
                raise ValueError("attack returned None")
            if set(out_dict.keys()) != set(src_keys):
                raise ValueError(f"attack returned keys mismatch: expected {len(src_keys)} keys, got {len(out_dict.keys())}")
            # ensure outputs are numpy arrays and shapes preserved
            converted = {}
            for k in src_keys:
                v = out_dict[k]
                arr = np.array(v, copy=True)
                if arr.shape != np.array(delta_np[k]).shape:
                    raise ValueError(f"attack changed shape for key {k}: {arr.shape} vs {np.array(delta_np[k]).shape}")
                converted[k] = arr
            return converted

        # --- MC-Grad (whole delta) ---
        if (
            exp.is_mc
            and exp.grad_mode
            and client_id in attacker_ids
            and run_cfg.mc_grad_delta
        ):
            try:
                # optional: allow per-client engine instance (engines.get_mc_for_client)
                mc_engine = getattr(engines, "mc_grad_engine_for_client", None) or engines.mc_grad_engine
                # only pass prev_updates if meaningful (not None)
                if prev_updates is not None:
                    attacked = mc_engine.generate(d_np, prev_updates=prev_updates)
                else:
                    attacked = mc_engine.generate(d_np)
                d_np = _validate_and_convert(attacked, d_np.keys())
                print(f"[ATTACK][MC-GRAD-DELTA] client={client_id} keys={len(d_np)}")
            except Exception as e:
                # keep original d_np if attack fails (but log)
                print(f"[WARN] MC-GRAD-DELTA client={client_id} failed: {e}")

        # --- FreeRider grad attack (after MC-grad) ---
        if (
            exp.is_fr
            and exp.grad_mode
            and client_id in attacker_ids
        ):
            try:
                fr_engine = getattr(engines, "free_rider_grad_engine_for_client", None) or engines.free_rider_grad_engine
                gr, meta = fr_engine.apply(
                    d_np, client_metadata={"num_samples": num_samples}
                )
                d_np = _validate_and_convert(gr, d_np.keys())
                num_samples = meta.get("num_samples", num_samples)
                print(f"[ATTACK][FR-GRAD] client={client_id} num_samples={num_samples}")
            except Exception as e:
                print(f"[WARN] FR-GRAD client={client_id} failed: {e}")

        return d_np, num_samples

    # Helper for applying sybil (only runs if grad mode active or sybil-only)
    def apply_sybil(d_np: Dict[str, np.ndarray], client_id, num_samples) -> Tuple[Dict[str, np.ndarray], int]:
        if (exp.grad_mode or exp.is_sybil_only) and client_id in attacker_ids:
            try:
                sy_upd, meta = engines.sybil_engine.apply(d_np, client_metadata={"num_samples": num_samples})
                d_np = {k: np.array(sy_upd[k]) for k in sy_upd.keys()}
                num_samples = meta.get("num_samples", num_samples)
                print(f"[ATTACK][SYBIL] {client_id} collusion={engines.sybil_engine.collusion}")
            except Exception as e:
                print(f"[WARN] SYBIL {client_id}: {e}")
        return d_np, num_samples

    # Main rounds
    for rnd in range(1, num_rounds + 1):
        print(f"\n[Main] Round {rnd}/{num_rounds} - exp={exp.experiment_case}")
        client_updates = []
        per_client_metrics = []
        current_sybil_updates_np = []

        for cb in clients:
            client_obj = cb["client"]
            client_id = cb["id"]

            original_dataset = client_obj.dataset
            fake_num_samples = None

            # Data attacks
            if exp.data_mode:
                try:
                    attacked_ds, fake_num_samples = apply_data_attacks(client_obj.dataset, client_id)
                    client_obj.dataset = attacked_ds
                except Exception as e:
                    print(f"[WARN] DATA-ATTACKS {client_id}: {e}")

            # Local training
            new_weights, num_samples = client_obj.local_train(
                global_weights=global_weights,
                epochs=local_epochs,
                batch_size=cfg.batch_size,
                lr=cfg.lr
            )

            # restore dataset
            client_obj.dataset = original_dataset

            # if data attack changed the sample count
            if fake_num_samples is not None:
                num_samples = fake_num_samples

            # Local evaluation
            client_acc = client_obj.evaluate(weights=new_weights, batch_size=cfg.batch_size)
            print(f"[LocalEval] {client_id} acc={client_acc:.4f}")

            # Compute delta
            device = _device_from_state_dict(global_weights)
            new_weights = {k: v.to(device) for k, v in new_weights.items()}
            delta = {}
            for k in global_weights.keys():
                try:
                    delta[k] = safe_param_subtract(new_weights[k], global_weights[k])
                except:
                    delta[k] = torch.zeros_like(global_weights[k])

            # Convert to numpy for gradient-level attacks
            delta_np = torch_delta_to_numpy(delta)

            # Store clean copy for MC history
            clean_copy = {k: v.copy() for k, v in delta_np.items()}

            # SYBIL shared vector collection
            if client_id in attacker_ids and (exp.grad_mode or exp.is_sybil_only):
                current_sybil_updates_np.append(clean_copy)

            # MC-GRAD and FR-GRAD (order preserved)
            if exp.grad_mode:
                delta_np, num_samples = apply_grad_attacks(delta_np, client_id, num_samples, previous_updates_for_mc_grad)

            # Update clean history (MC uses clean history)
            previous_updates_for_mc_grad.append(clean_copy)
            if len(previous_updates_for_mc_grad) > engines.mc_grad_engine.history_window:
                previous_updates_for_mc_grad = previous_updates_for_mc_grad[-engines.mc_grad_engine.history_window:]

            # SYBIL (runs after other grad modifications)
            if exp.grad_mode or exp.is_sybil_only:
                delta_np, num_samples = apply_sybil(delta_np, client_id, num_samples)

            # convert back to torch
            try:
                delta = numpy_delta_to_torch(delta_np, device=device, ref_state_dict=global_weights)
            except Exception:
                delta = {k: torch.zeros_like(global_weights[k]) for k in global_weights.keys()}

            # reconstruct final_weights to send to server
            final_weights = {}
            for k in global_weights.keys():
                final_weights[k] = global_weights[k] + delta[k]

            client_updates.append({
                "client_id": client_id,
                "state_dict": final_weights,
                "delta": delta,
                "num_samples": num_samples,
                "labels": list(cb["label2id"].keys()),
            })

            per_client_metrics.append({
                "id": client_id,
                "acc": float(client_acc),
                "num_samples": int(num_samples)
            })

        # Update sybil shared vector for next round
        if run_cfg.sybil_mode in ("leader", "coordinated"):
            engines.sybil_engine.update_shared_vector(current_sybil_updates_np)

        previous_sybil_updates.append(current_sybil_updates_np)
        if len(previous_sybil_updates) > 5:  # window size can be tuned
            previous_sybil_updates = previous_sybil_updates[-5:]

        # server aggregation + evaluation
        public_out = server.run_round(rnd, client_updates)
        global_acc = server.evaluate_global(global_test_ds, batch_size=cfg.batch_size)
        print(f"[GLOBAL] Acc after round {rnd}: {global_acc:.4f}")

        # logging
        entry = {
            "round": rnd,
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "global_acc": float(global_acc),
            "clients": per_client_metrics,
            "trust_summary": public_out.get("trust_summary", {})
        }
        with open(json_path, "r+", encoding="utf-8") as f:
            logs = json.load(f)
            logs.append(entry)
            f.seek(0)
            json.dump(logs, f, indent=2)
            f.truncate()

        row = [rnd, entry["timestamp"], float(global_acc)]
        for cm in per_client_metrics:
            row += [cm["acc"], cm["num_samples"]]
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)

    print("Training completed.")

if __name__ == "__main__":
    main()
