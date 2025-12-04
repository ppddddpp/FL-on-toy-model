import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import numpy as np
import json
from typing import Any, Dict

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
        """
        Safely load global weights into client model.
        Allows classifier head replacement when shape differs (due to label expansion).
        """
        model_dict = model.state_dict()
        new_state = {}

        for k, v in global_weights.items():
            if k in model_dict:
                local_shape = model_dict[k].shape
                global_shape = v.shape

                # --- classifier shape changes are allowed ---
                if "classifier" in k and local_shape != global_shape:
                    # Classifier expansion allowed only when global is larger
                    if global_shape[0] > local_shape[0]:
                        print(f"[SafeLoad] Expanding classifier {k}: {local_shape} -> {global_shape}")
                        new_state[k] = v.to(model_dict[k].device)
                    else:
                        print(f"[SafeLoad] Ignoring smaller classifier {k}, keeping local shape {local_shape}")
                    continue

                # normal matching case
                if local_shape == global_shape:
                    new_state[k] = v.to(model_dict[k].device)
                else:
                    print(f"[SafeLoad] Skipped weight '{k}' due to shape mismatch "
                        f"{tuple(local_shape)} vs {tuple(global_shape)}")
            else:
                print(f"[SafeLoad] Key '{k}' does not exist in local model, skipping.")

        # update model
        model_dict.update(new_state)
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
        if hasattr(self, "_cached_model"):
            model = self._cached_model.to(self.device)
        else:
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

        # Update cached global model after local training (for next round)
        self._cached_model = model.to("cpu")
        self._cached_global_state = {
            k: v.detach().cpu().clone()
            for k, v in model.state_dict().items()
        }

        discovered = self.discover_local_labels()
        return {k: v.cpu() for k, v in model.state_dict().items()}, len(self.dataset), discovered

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

    def discover_local_labels(self) -> Dict[str, Any]:
        """
        Returns a dict with:
          - 'label_ids': set of integer label ids found in the dataset
          - 'label_names': optional set of label name strings if dataset provides mapping
        """
        label_ids = set()
        # try to iterate dataset quickly (works for TensorDataset of (x,mask,y))
        try:
            for _, _, y in DataLoader(self.dataset, batch_size=256):
                label_ids.update([int(v.item()) for v in y])
        except Exception:
            # fallback: try iterating element-wise
            try:
                for sample in self.dataset:
                    y = sample[-1]
                    label_ids.add(int(y.item()) if torch.is_tensor(y) else int(y))
            except Exception:
                pass

        label_names = None
        # If dataset or client has a mapping (id->name), expose it:
        if hasattr(self.dataset, "id2label"):
            label_names = {self.dataset.id2label[i] for i in label_ids if i in self.dataset.id2label}
        elif hasattr(self, "local_id2label"):
            label_names = {self.local_id2label[i] for i in label_ids if i in self.local_id2label}

        return {"label_ids": label_ids, "label_names": label_names}
    
    def load_global_model(self, global_state: Dict[str, torch.Tensor]):
        """
        Called by server when broadcasting an expanded global model 
        (with new labels / bigger classifier).
        """
        model = self.model_fn().to(self.device)

        try:
            model = self._load_state_safely(model, global_state)
        except Exception as e:
            print(f"[{self.client_id}] Safe load failed: {e}")
            model.load_state_dict(global_state, strict=False)

        # Cache for later local_train
        self._cached_global_state = {
            k: v.detach().cpu().clone()
            for k, v in model.state_dict().items()
        }
        self._cached_model = model

        print(f"[{self.client_id}] Loaded & cached global model.")
        return True
