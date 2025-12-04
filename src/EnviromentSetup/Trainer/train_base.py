from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import json
import os
import csv
import wandb
from DataHandler import DatasetBuilder
from Helpers import Config, kg_alignment_loss
from EnviromentSetup.model.model import ToyBERTClassifier
from EnviromentSetup.KnowledgeGraphModel.KG_Builder import KGBuilder
from EnviromentSetup.KnowledgeGraphModel.KG_Trainer import KGTrainer
from EnviromentSetup.KnowledgeGraphModel.KGVocabAligner import KGVocabAligner

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent

class BaseTrainer:
    def __init__(
        self,
        model: nn.Module = None,
        train_dataset=None,
        val_dataset=None,
        test_dataset=None,
        batch_size: int = 32,
        lr: float = 1e-4,
        device: str = None,
        kg_dir: str = None,
        ttl_path: str = BASE_DIR / "data" / "animal" / "base" / "animal_kg.cleaned.ttl",
        save_dir: str = "checkpoints/base_model",
        cfg=None,
        use_wandb: bool = False
    ):
        # --- Load config ---
        cfg = Config.load(os.path.join(BASE_DIR, "config", "config.yaml")) if cfg is None else cfg
        self.cfg = cfg

        # --- If datasets not provided, build them ---
        if train_dataset is None:
            print("[BaseTrainer] Building datasets from config...")

            dataset_path = BASE_DIR / "data" / "animal" / "base" / "base_model.csv"
            train_dataset, val_dataset, test_dataset, vocab, label2id = DatasetBuilder.build_dataset(
                path=dataset_path,
                max_len=cfg.max_seq_len,
                text_col="Information",
                label_col="Group"
            )

            vocab_size = len(vocab)
            num_classes = len(label2id)
        else:
            vocab_size = getattr(train_dataset, "vocab_size", None)
            num_classes = getattr(train_dataset, "num_classes", None)

        # --- Create model if not provided ---
        if model is None:
            model = ToyBERTClassifier(
                vocab_size=vocab_size,
                num_classes=num_classes,
                d_model=cfg.model_dim,
                nhead=cfg.num_heads,
                num_layers=cfg.num_layers,
                dim_ff=cfg.ffn_dim,
                max_len=cfg.max_seq_len,
                dropout=cfg.dropout,
            )

        self.model = model
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)

        # --- DataLoaders ---
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size) if val_dataset else None
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size) if test_dataset else None

        # --- KG dir ---
        self.kg_dir = BASE_DIR / "knowledge_graph" if not kg_dir else Path(kg_dir)
        self.kg_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_path = ttl_path if ttl_path else BASE_DIR / "data" / "animal" / "base" / "animal_kg.cleaned.ttl"

        # --- Loss and optimizer ---
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # --- Save dir ---
        self.save_dir = Path(save_dir) if save_dir else BASE_DIR / "checkpoints" / "base_model"
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # --- W&B logging ---
        self.use_wandb = use_wandb and hasattr(self.cfg, "project_name")

        if self.use_wandb:
            wandb.init(
                project=self.cfg.project_name,
                config={
                    "epochs": self.cfg.epochs,
                    "batch_size": batch_size,
                    "lr": lr,
                    "model_dim": self.cfg.model_dim,
                    "ffn_dim": self.cfg.ffn_dim,
                    "num_heads": self.cfg.num_heads,
                    "max_seq_len": self.cfg.max_seq_len,
                    "kg_model": self.cfg.kg_model,
                    "kg_emb_dim": self.cfg.kg_emb_dim,
                    "cls_weight": self.cfg.cls_weight,
                    "kg_weight": self.cfg.kg_weight,
                }
            )
            wandb.watch(self.model, log="all")

    def _ensure_kg_embeddings(self, vocab=None):
        """
        Ensure KG embeddings exist, else train them.
        Then inject them into the model's embeddings using vocab with KG alignment.
        """
        if not self.cfg:
            print("[BaseTrainer] No KG config provided, skipping KG integration")
            return

        kg_trainer = KGTrainer(
            kg_dir=self.kg_dir,
            emb_dim=self.cfg.kg_emb_dim,
            joint_dim=self.cfg.model_dim,
            model_name=self.cfg.kg_model,
            model_kwargs=self.cfg.kg_model_kwargs,
            lr=self.cfg.kg_lr
        )

        best_node_path = self.kg_dir / "node_embeddings_best.npy"
        node2id_path = self.kg_dir / "node2id.json"

        if not best_node_path.exists():
            print("[BaseTrainer] Building & Training Knowledge Graph embeddings...")

            # build triples from ontology
            kg_builder = KGBuilder(
                ttl_path=self.ttl_path,
                namespace_filter="http://example.org/ontology#"
            )
            triples_id, entity2id, relation2id = kg_builder.build()
            kg_builder.summary()

            # dump maps + triples for KGTrainer
            with open(node2id_path, "w") as f:
                json.dump(entity2id, f, indent=2)
            with open(self.kg_dir / "relation2id.json", "w") as f:
                json.dump(relation2id, f, indent=2)
            with open(self.kg_dir / "triples.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["s_id", "r_id", "o_id"])
                for h, r, t in triples_id:
                    writer.writerow([h, r, t])

            # train embeddings
            kg_trainer.load_triples()
            kg_trainer.train(epochs=self.cfg.kg_epochs, log_to_wandb=self.use_wandb, negative_size=self.cfg.kg_neg_samples, batch_size=64)
            kg_trainer.save_embeddings()
        else:
            print("[BaseTrainer] Using cached KG embeddings")

        # Load embeddings
        node_embs = np.load(best_node_path)
        node_embs = torch.tensor(node_embs, dtype=torch.float32, device=self.device)

        # --- Projection if needed ---
        model_dim = self.model.token_embeddings.embedding_dim
        if node_embs.size(1) != model_dim:
            print(f"[BaseTrainer] Projecting KG embeddings {node_embs.size(1)} -> {model_dim}")
            proj = nn.Linear(node_embs.size(1), model_dim, bias=False).to(self.device)
            with torch.no_grad():
                node_embs = proj(node_embs)

        # Align vocab and KG
        if vocab is not None:
            with open(node2id_path, "r") as f:
                entity2id = json.load(f)
            kg_builder = KGBuilder(ttl_path="", namespace_filter=None)
            kg_builder.entity2id = entity2id  # restore mapping
            aligner = KGVocabAligner(kg_builder, vocab, device=self.device)
            aligner.inject_embeddings(self.model, node_embs)
        else:
            print("[BaseTrainer] No vocab provided, injected by index order")
            with torch.no_grad():
                num_to_copy = min(self.model.token_embeddings.weight.size(0), node_embs.size(0))
                self.model.token_embeddings.weight[:num_to_copy].copy_(node_embs[:num_to_copy])

    def train(self, epochs: int = 5, save_every: int = 1):
        """
        Train the model for a number of epochs, with optional validation.
        Combines classification loss with KG alignment loss (no structural loss here).
        """
        # --- Ensure KG embeddings before training ---
        self._ensure_kg_embeddings()

        # load node2id and KG embeddings for alignment
        node2id_path = self.kg_dir / "node2id.json"
        with open(node2id_path, "r") as f:
            self.kg_node2id = json.load(f)
        best_node_path = self.kg_dir / "node_embeddings_best.npy"
        self.kg_node_embs = torch.tensor(
            np.load(best_node_path),
            dtype=torch.float32,
            device=self.device
        )

        for epoch in range(1, epochs + 1):
            self.model.train()
            total_loss = 0.0

            for batch in tqdm(self.train_loader, desc=f"Epoch {epoch}/{epochs}"):
                # --- unpack batch ---
                if len(batch) == 2:
                    input_ids, labels = [x.to(self.device) for x in batch]
                    attention_mask = None
                elif len(batch) == 3:
                    input_ids, attention_mask, labels = [x.to(self.device) for x in batch]
                else:
                    raise ValueError("Unexpected batch format")

                # --- forward pass (logits + cls embedding) ---
                logits, cls_vec = self.model(
                    input_ids,
                    attention_mask=attention_mask,
                    return_hidden=True
                )

                # --- classification loss ---
                cls_loss = self.criterion(logits, labels)

                # --- alignment loss ---
                align_loss = torch.tensor(0.0, device=self.device)
                if getattr(self.cfg, "align_weight", 0.0) > 0:
                    # use dummy IDs if dataset doesn’t give report IDs
                    batch_ids = list(range(len(labels)))
                    align_loss = kg_alignment_loss(
                        joint_emb=cls_vec,
                        batch_ids=batch_ids,
                        kg_embs=self.kg_node_embs,
                        node2id=self.kg_node2id,
                        trainer=self,  # provides proj_to_kg
                        labels=None,
                        label_cols=None,
                        loss_type=self.cfg.kg_method  # "cosine" or "mse"
                    )

                # --- weighted sum ---
                loss = (
                    self.cfg.cls_weight * cls_loss
                    + self.cfg.kg_weight * align_loss
                )

                # --- optimize ---
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(self.train_loader)
            print(f"[BaseTrainer] Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f}")

            if self.use_wandb:
                wandb.log({"train_loss": avg_loss, "epoch": epoch})

            # Validation
            if self.val_loader:
                val_acc = self.evaluate(split="val")
                print(f"[BaseTrainer] Validation Accuracy: {val_acc:.4f}")
                if self.use_wandb:
                    wandb.log({"val_acc": val_acc, "epoch": epoch})

            # Save checkpoint
            if epoch % save_every == 0:
                self.save_checkpoint(epoch)

    def evaluate(self, split: str = "val"):
        if split == "val" and not self.val_loader:
            raise ValueError("No validation dataset provided")
        if split == "test" and not self.test_loader:
            raise ValueError("No test dataset provided")

        loader = self.val_loader if split == "val" else self.test_loader
        self.model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for batch in loader:
                if len(batch) == 2:
                    input_ids, labels = [x.to(self.device) for x in batch]
                    logits = self.model(input_ids)
                else:
                    input_ids, attention_mask, labels = [x.to(self.device) for x in batch]
                    logits = self.model(input_ids, attention_mask)

                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        acc = correct / total
        if self.use_wandb:
            wandb.log({f"{split}_acc": acc})

        return acc

    def save_checkpoint(self, epoch: int):
        ckpt_path = self.save_dir / f"epoch{epoch}.pt"
        torch.save(self.model.state_dict(), ckpt_path)
        print(f"[BaseTrainer] Saved checkpoint -> {ckpt_path}")

    def load_checkpoint(self, ckpt_path: str):
        ckpt_path = Path(ckpt_path)
        self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        self.model.to(self.device)
        print(f"[BaseTrainer] Loaded checkpoint <- {ckpt_path}")