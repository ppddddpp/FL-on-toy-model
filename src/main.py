import json
import csv
import torch
from pathlib import Path
from Server.server import Server
from Client.client import Client
from DataHandler.dataset_builder import DatasetBuilder
from Helpers.configLoader import Config
from model.model import ToyBERTClassifier
import datetime

BASE_DIR = Path(__file__).resolve().parent.parent
cfg = Config.load(BASE_DIR / "config" / "config.yaml")

# --- Build clients (and ensure model_fn captures sizes) ---
client_paths = [
    BASE_DIR / "data" / "medical_data" / "n1" / "client_1_data.csv",
    BASE_DIR / "data" / "medical_data" / "n2" / "client_2_data.csv",
    BASE_DIR / "data" / "medical_data" / "n3" / "client_3_data.csv"
]

_, _, _, base_vocab, base_label2id = DatasetBuilder.build_dataset(
    path=BASE_DIR / "data" / "medical_data" / "base" / "base_model.csv",
    max_len=cfg.max_seq_len
)

clients = []
for i, path in enumerate(client_paths):
    train_ds, val_ds, test_ds, vocab, label2id = DatasetBuilder.build_dataset(
        path=path, 
        max_len=cfg.max_seq_len,
        vocab = base_vocab,
        label2id=base_label2id
    )
    vocab_size = train_ds.vocab_size
    num_classes = train_ds.num_classes

    # make model_fn capture sizes and cfg hyperparams (avoid late binding bug)
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
        "client": Client(
            client_id=f"client_{i+1}",
            model_fn=make_model_fn(),   # use proper vocab/labels
            dataset=train_ds,
            device="cuda" if torch.cuda.is_available() else "cpu"
        ),
        "val": val_ds,
        "test": test_ds
    })


# --- Build server (server will load/train base model) ---
server = Server(
    model_cls=ToyBERTClassifier,
    config=cfg,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# --- Global test dataset (centralized evaluation) ---
_, _, global_test_ds, _, _ = DatasetBuilder.build_dataset(
    path=BASE_DIR / "data" / "medical_data" / "base" / "base_model.csv",
    max_len=cfg.max_seq_len
)

# --- Setup logging files ---
log_dir = BASE_DIR / "logs"
log_dir.mkdir(parents=True, exist_ok=True)
json_path = log_dir / "accuracy_log.json"
csv_path = log_dir / "accuracy_log.csv"

# initialize JSON if missing
if not json_path.exists():
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([], f, indent=2)

# Prepare CSV header (will write the header on first write)
def write_csv_header(num_clients):
    base_cols = ["round", "timestamp", "global_acc"]
    # dynamic per-client columns
    client_cols = []
    for i in range(1, num_clients + 1):
        client_cols += [f"client_{i}_acc", f"client_{i}_samples"]
    header = base_cols + client_cols
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

# Create header once
num_clients = len(clients)
write_csv_header(num_clients)

# --- Federated loop (main orchestrates) ---
global_weights = server.global_model.state_dict()
num_rounds = 5
local_epochs = 3

for rnd in range(1, num_rounds + 1):
    print(f"\n[Main] Round {rnd}/{num_rounds}")
    client_updates = []
    per_client_metrics = []

    # --- Local training on each client ---
    for idx, cb in enumerate(clients):
        client_obj = cb["client"]
        client_id = cb["id"]

        new_weights, num_samples = client_obj.local_train(
            global_weights=global_weights,
            epochs=local_epochs,
            batch_size=cfg.batch_size,
            lr=cfg.lr
        )

        # local evaluation using updated weights
        client_acc = client_obj.evaluate(weights=new_weights, batch_size=cfg.batch_size)
        print(f"[Main] {client_id} after local train -> acc={client_acc:.4f}, samples={num_samples}")

        client_updates.append((new_weights, num_samples))
        per_client_metrics.append({
            "id": client_id,
            "acc": float(client_acc),
            "num_samples": int(num_samples)
        })

    # --- Aggregation (server) ---
    global_weights = server.fedavg(client_updates)
    server.global_model.load_state_dict(global_weights)

    # --- Global evaluation on central test set ---
    global_acc = server.evaluate_global(global_test_ds, batch_size=cfg.batch_size)
    print(f"[Main] Global accuracy after round {rnd}: {global_acc:.4f}")

    # --- Persist logs: JSON (structured) and CSV (flat) ---
    entry = {
        "round": rnd,
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "global_acc": float(global_acc),
        "clients": per_client_metrics
    }

    # append to JSON
    with open(json_path, "r+", encoding="utf-8") as f:
        logs = json.load(f)
        logs.append(entry)
        f.seek(0)
        json.dump(logs, f, indent=2)
        f.truncate()

    # append to CSV (flat format)
    # CSV row: round, timestamp, global_acc, client_1_acc, client_1_samples, client_2_acc, client_2_samples, ...
    row = [rnd, entry["timestamp"], float(global_acc)]
    for cm in per_client_metrics:
        row += [cm["acc"], cm["num_samples"]]

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)

    # --- Save server checkpoint ---
    server.save_checkpoint(global_weights, round_num=rnd)

print("Training completed. Logs saved to:", json_path, csv_path)
