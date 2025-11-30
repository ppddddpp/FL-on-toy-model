import json
import csv
import torch
from pathlib import Path
from Server.server import Server
from Client.client import Client
from DataHandler.dataset_builder import DatasetBuilder
from Helpers.configLoader import Config
from EnviromentSetup.model.model import ToyBERTClassifier
from torch.utils.data import DataLoader
import datetime

BASE_DIR = Path(__file__).resolve().parent.parent
cfg = Config.load(BASE_DIR / "config" / "config.yaml")

def safe_param_subtract(client_param, global_param):
    a, b = client_param, global_param
    if a.shape == b.shape:
        return a - b
    # allow trivial singleton-dimension mismatches with equal numel
    if a.numel() == b.numel():
        return a.reshape(b.shape) - b
    # allow classifier expansion mismatches (existing logic)
    if a.ndim == 2 and b.ndim == 2 and a.shape[1] == b.shape[1]:
        max_rows = max(a.shape[0], b.shape[0])
        def pad_rows(t, rows):
            if t.shape[0] == rows:
                return t
            pad = torch.zeros((rows - t.shape[0], t.shape[1]), device=t.device, dtype=t.dtype)
            return torch.cat([t, pad], dim=0)
        return pad_rows(a, max_rows) - pad_rows(b, max_rows)
    # fallback: zeros of target shape (so delta length matches server param)
    try:
        return torch.zeros_like(b)
    except Exception:
        return torch.zeros(b.shape, dtype=b.dtype, device=b.device)

# =========================================================
# Build base dataset (used for vocab, labels, anchors)
# =========================================================
base_train, base_val, base_test, base_vocab, base_label2id = DatasetBuilder.build_dataset(
    path=BASE_DIR / "data" / "animal" / "base" / "base_model.csv",
    max_len=cfg.max_seq_len,
    text_col="Information",
    label_col="Group"
)

# =========================================================
# Build Server 
# =========================================================
anchor_loader = DataLoader(base_train, batch_size=cfg.batch_size, shuffle=False)
server = Server(
    model_cls=ToyBERTClassifier,
    config=cfg,
    device="cuda" if torch.cuda.is_available() else "cpu",
    text_col="Information",
    label_col="Group",
    anchor_loader=anchor_loader,
    checkpoint_dir="checkpoints/base_model"
)

# --- Get KG info from the server (trained embeddings or none) ---
kg_dir, has_kg = server.get_kg_info()
if has_kg:
    print(f"[Main] Using shared KG embeddings from: {kg_dir}")
else:
    print("[Main] No trained KG embeddings found — clients will skip KG alignment.")


# =========================================================
# Build clients
# =========================================================
client_paths = [
    BASE_DIR / "data" / "animal" / f"n{i}" / f"client_{i}_data.csv"
    for i in range(1, 4)
]

clients = []
for i, path in enumerate(client_paths):
    print(f"[ClientSetup] Loading client {i+1} data...")
    train_ds, val_ds, test_ds, vocab, label2id = DatasetBuilder.build_dataset(
        path=path,
        max_len=cfg.max_seq_len,
        vocab=base_vocab,       # reuse base vocab
        label2id=base_label2id.copy(), # reuse base labels
        text_col="Information",
        label_col="Group"
    )

    # ----------------- IMMEDIATE DIAGNOSTIC: label consistency -----------------
    # collect label ids actually used in the dataset objects
    ds_label_ids = []
    try:
        # ToyTextDataset stores raw numeric labels in .labels
        ds_label_ids = list(set(train_ds.labels + val_ds.labels + test_ds.labels))
    except Exception:
        # fallback: try iterating a few items
        ds_label_ids = []
        for i in range(min(50, len(train_ds))):
            _, _, lbl = train_ds[i]
            ds_label_ids.append(int(lbl))
        ds_label_ids = list(set(ds_label_ids))

    min_id = min(ds_label_ids) if ds_label_ids else None
    max_id = max(ds_label_ids) if ds_label_ids else None
    n_label_map = len(label2id)

    print(f"[DIAG] client {i+1} label2id keys: {sorted(label2id.items(), key=lambda kv: kv[1])}")
    print(f"[DIAG] client {i+1} label stats: min={min_id}, max={max_id}, unique={len(ds_label_ids)}, label_map_size={n_label_map}")

    # contiguity check for label2id values
    id_set = set(label2id.values())
    if id_set != set(range(n_label_map)):
        print(f"[DIAG][WARNING] label2id ids are non-contiguous: {sorted(id_set)} vs expected 0..{n_label_map-1}")

    # Out-of-range check
    if ds_label_ids and (max_id is not None) and max_id >= n_label_map:
        raise RuntimeError(
            f"[DIAG][ERROR] dataset contains label id >= num_classes: max_label_id={max_id}, num_classes={n_label_map}. "
            "This will crash CrossEntropyLoss on GPU. Fix label mapping or disable dynamic expansion."
        )
    # -------------------------------------------------------------------------

    vocab_size = train_ds.vocab_size
    num_classes = len(label2id)
    print(f"[ClientSetup] Client {i+1} vocab size: {vocab_size}, num classes: {num_classes}")

    # factory to avoid late binding issues
    def make_model_fn(vs=vocab_size, nc=num_classes, c=cfg):
        return lambda: ToyBERTClassifier(
            vocab_size=vs,
            num_classes=nc,
            d_model=c.model_dim,
            nhead=c.num_heads,
            num_layers=c.num_layers,
            dim_ff=c.ffn_dim,
            max_len=c.max_seq_len,
            dropout=c.dropout
        )

    clients.append({
        "id": f"client_{i+1}",
        "label2id": label2id,
        "client": Client(
            client_id=f"client_{i+1}",
            model_fn=make_model_fn(),
            dataset=train_ds,
            device="cuda" if torch.cuda.is_available() else "cpu",
            kg_dir=kg_dir,
            use_kg_align=has_kg
        ),
        "val": val_ds,
        "test": test_ds
    })

    # determine required global num_classes (union / max across clients)
    required_num_classes = max(len(cb["label2id"]) for cb in clients)
    server_num_classes = None
    # attempt to infer server classifier keys (adjust keys if your classifier key differs)
    sd = server.global_model.state_dict()
    w_key = None
    b_key = None
    for k in sd.keys():
        if k.endswith("weight") and "classifier" in k and sd[k].ndim == 2:
            w_key = k
            # try to find corresponding bias
            bias_candidate = k[:-6] + "bias"
            if bias_candidate in sd:
                b_key = bias_candidate
            break

    if w_key is not None:
        server_num_classes = sd[w_key].shape[0]
        if required_num_classes > server_num_classes:
            print(f"[Main] Expanding server classifier from {server_num_classes} -> {required_num_classes}")
            # pad weight rows with zeros and pad bias
            new_w = torch.zeros((required_num_classes, sd[w_key].shape[1]), dtype=sd[w_key].dtype, device=sd[w_key].device)
            new_w[:server_num_classes, :] = sd[w_key]
            sd[w_key] = new_w
            if b_key is not None:
                new_b = torch.zeros((required_num_classes,), dtype=sd[b_key].dtype, device=sd[b_key].device)
                new_b[:server_num_classes] = sd[b_key]
                sd[b_key] = new_b
            # load back into server model
            server.global_model.load_state_dict(sd, strict=False)
            print("[Main] Server classifier expanded and server model state updated.")
        else:
            print(f"[Main] Server classifier ({server_num_classes}) already >= required ({required_num_classes}).")
    else:
        print("[Main] Could not detect server classifier weight key automatically — check key names.")

# =========================================================
# Global test dataset (for centralized evaluation)
# =========================================================
_, _, global_test_ds, _, _ = DatasetBuilder.build_dataset(
    path=BASE_DIR / "data" / "animal" / "base" / "base_model.csv",
    max_len=cfg.max_seq_len,
    text_col="Information",
    label_col="Group"
)

# =========================================================
# Setup logging
# =========================================================
log_dir = BASE_DIR / "logs"
log_dir.mkdir(parents=True, exist_ok=True)
json_path = log_dir / "accuracy_log.json"
csv_path = log_dir / "accuracy_log.csv"

if not json_path.exists():
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([], f, indent=2)

def write_csv_header(num_clients):
    base_cols = ["round", "timestamp", "global_acc"]
    client_cols = []
    for i in range(1, num_clients + 1):
        client_cols += [f"client_{i}_acc", f"client_{i}_samples"]
    header = base_cols + client_cols
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

num_clients = len(clients)
write_csv_header(num_clients)

# =========================================================
# Federated Learning Loop
# =========================================================

global_weights = server.global_model.state_dict()
num_rounds = 5
local_epochs = 3

for rnd in range(1, num_rounds + 1):
    print(f"\n[Main] Round {rnd}/{num_rounds}")
    client_updates = []
    per_client_metrics = []

    # --- Local training ---
    for cb in clients:
        client_obj = cb["client"]
        client_id = cb["id"]

        new_weights, num_samples = client_obj.local_train(
            global_weights=global_weights,
            epochs=local_epochs,
            batch_size=cfg.batch_size,
            lr=cfg.lr
        )

        client_acc = client_obj.evaluate(weights=new_weights, batch_size=cfg.batch_size)
        print(f"[Main] {client_id} -> acc={client_acc:.4f}, samples={num_samples}")

        # compute delta for SelfCheck
        device = next(iter(global_weights.values())).device

        # Move new weights to same device
        new_weights = {k: v.to(device) for k, v in new_weights.items()}

        # DEBUG: verify weight key sets and shapes
        server_keys = set(global_weights.keys())
        client_keys = set(new_weights.keys())
        extra_in_client = client_keys - server_keys
        missing_in_client = server_keys - client_keys
        if extra_in_client or missing_in_client:
            print(f"[DIAG][{client_id}] extra_keys={sorted(extra_in_client)} missing_keys={sorted(list(missing_in_client)[:10])} (truncated)")

        # show any per-param shape differences
        for k in sorted(server_keys & client_keys):
            if new_weights[k].shape != global_weights[k].shape:
                print(f"[DIAG][{client_id}] SHAPE MISMATCH {k}: client={new_weights[k].shape} server={global_weights[k].shape}")

        delta = {}
        for k in global_weights.keys():
            if k not in new_weights:
                continue
            try:
                delta[k] = safe_param_subtract(new_weights[k].to(device), global_weights[k].to(device))
            except Exception as e:
                print(f"[Warning] Could not compute delta for {k}: {e}")
                delta[k] = torch.zeros_like(global_weights[k])

        client_updates.append({
            "client_id": client_id,
            "state_dict": new_weights,
            "delta": delta,
            "num_samples": num_samples,
            "labels": list(label2id.keys())
        })

        per_client_metrics.append({
            "id": client_id,
            "acc": float(client_acc),
            "num_samples": int(num_samples)
        })

    # --- Server SelfCheck + Weighted Aggregation ---
    public_out = server.run_round(rnd, client_updates)

    # --- Evaluate global model ---
    global_acc = server.evaluate_global(global_test_ds, batch_size=cfg.batch_size)
    print(f"[Main] Global accuracy after round {rnd}: {global_acc:.4f}")

    # --- Logging ---
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
        writer = csv.writer(f)
        writer.writerow(row)

print("Training completed. Logs saved to:", json_path, csv_path)
