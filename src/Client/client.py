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
        self.client_id = client_id
        self.model_fn = model_fn
        self.dataset = dataset
        self.device = device
        self.kg_dir = kg_dir
        self.use_kg_align = use_kg_align and kg_dir is not None
        self.mc_grad_engine = None

        # preload KG if exists
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

    # ---------- safe load ----------
    @staticmethod
    def _load_state_safely(model, global_weights):
        model_dict = model.state_dict()
        filtered = {}
        for k, v in global_weights.items():
            if k in model_dict and model_dict[k].shape == v.shape:
                filtered[k] = v
            else:
                print(f"[Warning] Skipped weight '{k}' due to shape mismatch.")
        model_dict.update(filtered)
        model.load_state_dict(model_dict)
        return model

    # ---------- KG alignment ----------
    def _align_classifier_with_kg(self, model):
        if self.kg_embs is None:
            return model

        kg_embs = self.kg_embs

        classifier = None
        if hasattr(model, "classifier"):
            for m in reversed(list(model.classifier.modules())):
                if isinstance(m, nn.Linear):
                    classifier = m
                    break
        if classifier is None:
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    classifier = module
                    break
        if classifier is None:
            return model

        target_dim = getattr(getattr(model, "token_embeddings", None), "embedding_dim", None)
        if target_dim is None:
            target_dim = classifier.weight.shape[1]

        if kg_embs.size(1) != target_dim:
            print(f"[Client] Projecting KG embeddings {kg_embs.size(1)}→{target_dim}")
            proj = torch.nn.Linear(kg_embs.size(1), target_dim, bias=False).to(kg_embs.device)
            with torch.no_grad():
                kg_embs = proj(kg_embs)

        with torch.no_grad():
            W = classifier.weight
            sim = torch.nn.functional.cosine_similarity(
                W.unsqueeze(1), kg_embs.unsqueeze(0), dim=-1
            )
            aligned = torch.matmul(sim, kg_embs)
            if aligned.shape == classifier.weight.shape:
                classifier.weight.copy_(aligned)

        print(f"[{self.client_id}] Classifier aligned with KG embeddings")
        return model

    # ---------- LOCAL TRAIN ----------
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
        model = self._load_state_safely(model, global_weights)
        if self.use_kg_align:
            model = self._align_classifier_with_kg(model)

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

                if self.mc_grad_engine is not None:
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            grad_np = param.grad.detach().cpu().numpy()
                            attacked = self.mc_grad_engine.generate({name: grad_np})[name]
                            param.grad = torch.from_numpy(attacked).to(param.device)

                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

                if (batch_idx + 1) % log_interval == 0:
                    print(f"[{self.client_id}] Epoch {epoch+1} "
                          f"Batch {batch_idx+1}/{len(loader)} Loss: {loss.item():.4f}")

        # save
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
