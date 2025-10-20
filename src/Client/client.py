import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import numpy as np
import json

class Client:
    def __init__(
        self,
        client_id,
        model_fn,
        dataset,
        device="cpu",
        kg_dir=None,
        use_kg_align=True
    ):
        """
        Args:
            client_id (str): client identifier
            model_fn (callable): returns a new model instance
            dataset (Dataset): local dataset
            device (str): "cpu" or "cuda"
            kg_dir (str or Path, optional): path to KG embedding directory
            use_kg_align (bool): whether to use semantic KG alignment
        """
        self.client_id = client_id
        self.model_fn = model_fn
        self.dataset = dataset
        self.device = device
        self.kg_dir = kg_dir
        self.use_kg_align = use_kg_align and kg_dir is not None

        # preload KG if available
        if self.use_kg_align and os.path.exists(os.path.join(kg_dir, "node_embeddings_best.npy")):
            self.kg_embs = torch.tensor(
                np.load(os.path.join(kg_dir, "node_embeddings_best.npy")),
                dtype=torch.float32,
                device=self.device
            )
            with open(os.path.join(kg_dir, "node2id.json")) as f:
                self.node2id = json.load(f)
            print(f"[{client_id}] Loaded KG embeddings ({self.kg_embs.shape[0]} nodes)")
        else:
            self.kg_embs, self.node2id = None, None
            if kg_dir:
                print(f"[{client_id}] KG directory found but embeddings missing — skipping alignment")

    # ---------- helper for safe model loading ----------
    @staticmethod
    def _load_state_safely(model, global_weights):
        model_dict = model.state_dict()
        filtered = {}
        for k, v in global_weights.items():
            if k in model_dict and model_dict[k].shape == v.shape:
                filtered[k] = v
            else:
                print(f"[Warning] Skipped weight '{k}' due to shape mismatch: "
                      f"{v.shape} vs {model_dict.get(k, torch.empty(0)).shape}")
        model_dict.update(filtered)
        model.load_state_dict(model_dict)
        return model

    # ---------- optional semantic alignment ----------
    def _align_classifier_with_kg(self, model):
        if self.kg_embs is None:
            return model

        # use a local name so assignments below don't make kg_embs a local before this check
        kg_embs = self.kg_embs

        # find a classifier linear layer (try classifier attr first, then fallback)
        classifier = None
        # prefer an explicit classifier attribute if present
        if hasattr(model, "classifier"):
            # attempt to find the final linear layer inside model.classifier
            for m in reversed(list(model.classifier.modules())):
                if isinstance(m, nn.Linear):
                    classifier = m
                    break

        # fallback to scanning named_modules
        if classifier is None:
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    classifier = module
                    break

        if classifier is None:
            return model

        # determine target dim (prefer token_embeddings if available)
        target_dim = getattr(getattr(model, "token_embeddings", None), "embedding_dim", None)
        if target_dim is None:
            # fallback to classifier hidden dim (weight shape: [num_labels, hidden_dim])
            target_dim = classifier.weight.shape[1]

        # project KG embeddings if their dim != model embedding dim
        if kg_embs.size(1) != target_dim:
            print(f"[Client] Projecting KG embeddings from {kg_embs.size(1)} -> {target_dim}")
            proj = torch.nn.Linear(kg_embs.size(1), target_dim, bias=False).to(kg_embs.device)
            with torch.no_grad():
                kg_embs = proj(kg_embs)

        with torch.no_grad():
            W = classifier.weight  # [num_labels, hidden_dim]
            # cosine sim between label embeddings and KG embeddings
            sim = torch.nn.functional.cosine_similarity(
                W.unsqueeze(1), kg_embs.unsqueeze(0), dim=-1
            )  # -> [num_labels, num_kg_nodes]
            aligned = torch.matmul(sim, kg_embs)  # -> [num_labels, target_dim]
            # ensure shape matches the classifier weight exactly
            if aligned.shape == classifier.weight.shape:
                classifier.weight.copy_(aligned)
            else:
                print("[Client] Warning: aligned shape mismatch, skipping weight copy:",
                    aligned.shape, classifier.weight.shape)

        print(f"[{self.client_id}] Classifier semantically aligned with KG embeddings")
        return model


    # ---------- main training ----------
    def local_train(
        self,
        global_weights,
        epochs=1,
        batch_size=16,
        lr=1e-3,
        save_path=None,
        log_interval=10
    ):
        model = self.model_fn().to(self.device)
        model = self._load_state_safely(model, global_weights)  # Option B
        if self.use_kg_align:
            model = self._align_classifier_with_kg(model)        # Option C

        loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        model.train()
        for epoch in range(epochs):
            for batch_idx, (ids, mask, y) in enumerate(loader):
                ids, mask, y = ids.to(self.device), mask.to(self.device), y.to(self.device)

                optimizer.zero_grad()
                logits = model(ids, attention_mask=mask)
                loss = criterion(logits, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

                if (batch_idx + 1) % log_interval == 0:
                    print(f"[{self.client_id}] Epoch {epoch+1} "
                          f"Batch {batch_idx+1}/{len(loader)} Loss: {loss.item():.4f}")

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"[{self.client_id}] Saved model to {save_path}")

        return {k: v.cpu() for k, v in model.state_dict().items()}, len(self.dataset)

    # ---------- evaluation ----------
    def evaluate(self, weights=None, batch_size=16):
        model = self.model_fn().to(self.device)
        if weights is not None:
            model = self._load_state_safely(model, weights)

        loader = DataLoader(self.dataset, batch_size=batch_size)
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for ids, mask, y in loader:
                ids, mask, y = ids.to(self.device), mask.to(self.device), y.to(self.device)
                logits = model(ids, attention_mask=mask)
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        acc = correct / total if total > 0 else 0.0
        print(f"[{self.client_id}] Evaluation Accuracy: {acc:.4f}")
        return acc
