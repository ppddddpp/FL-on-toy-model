import os
import torch
from pathlib import Path

from Trainer.train_base import BaseTrainer
from DataHandler.dataset_builder import DatasetBuilder
from Helpers.configLoader import Config

BASE_DIR = Path(__file__).resolve().parent.parent.parent

class Server:
    def __init__(self, model_cls, config=None, checkpoint_dir="checkpoints/base_model", device="cpu"):
        self.model_cls = model_cls
        self.config = config if config else Config.load(BASE_DIR / "config" / "config.yaml")
        self.device = torch.device(device if isinstance(device, str) else device)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # load / train base model
        self.global_model = self._load_or_train_base()

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

    def fedavg(self, client_updates):
        """
        client_updates: list of tuples (state_dict, num_samples)
        returns averaged state_dict on self.device
        """
        if len(client_updates) == 0:
            raise ValueError("No client updates to aggregate")

        total_samples = float(sum(num for _, num in client_updates))
        # start from zero tensors shaped like the first client's tensors
        first_state, _ = client_updates[0]
        new_state = {}
        for k, v in first_state.items():
            new_state[k] = torch.zeros_like(v, device=self.device, dtype=v.dtype)

        # accumulate weighted tensors
        for state, num_samples in client_updates:
            w = float(num_samples) / total_samples
            for k, v in state.items():
                new_state[k] += v.to(self.device) * w

        return new_state

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
