from typing import Optional, Sequence, Dict, Any, Set, Tuple
from copy import deepcopy
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

class KGConsistencyEvaluator:
    """
    Hybrid KGConsistencyEvaluator:
    Combines symbolic KG edge relations + semantic embedding similarity
    to check whether model predictions respect domain knowledge.
    """

    def __init__(
        self,
        kg_edges: Set[Tuple[str, str]],
        label_map: Dict[int, str],
        entity_embeddings: Optional[torch.Tensor] = None,
        entity2id: Optional[Dict[str, int]] = None,
        device: Optional[torch.device] = None,
        alpha: float = 0.6,  # weight for embedding-based similarity
        symmetric: bool = True,
        reject_threshold: float = 0.4,
        ema_decay: float = 0.9,
        sim_threshold: float = 0.7  # cosine similarity threshold
    ):
        self.device = device
        self.label_map = dict(label_map)
        self.entity_embeddings = entity_embeddings
        self.entity2id = entity2id
        self.alpha = alpha
        self.sim_threshold = sim_threshold
        self.reject_threshold = reject_threshold
        self.ema_decay = ema_decay
        self.ema_skg = None
        self._rolling_scores = []

        # --- normalize edges ---
        if symmetric:
            edges_sym = set()
            for e in kg_edges:
                if len(e) == 3:
                    a, _, b = e
                else:
                    a, b = e
                edges_sym.add((a, b))
                edges_sym.add((b, a))
            self.kg_edges = edges_sym
        else:
            self.kg_edges = {
                (e[0], e[2]) if len(e) == 3 else (e[0], e[1])
                for e in kg_edges
            }

    # ----------------------------
    # Core prediction & consistency
    # ----------------------------
    @staticmethod
    def _clone_model(model: nn.Module) -> nn.Module:
        return deepcopy(model)

    @torch.no_grad()
    def _predict_labels(self, model: nn.Module, loader: DataLoader, device: torch.device):
        model.eval().to(device)
        preds, trues = [], []
        for batch in loader:
            if len(batch) == 3:
                x, mask, y = batch
            elif len(batch) == 2:
                x, y = batch
                mask = None
            else:
                raise ValueError(f"Unexpected batch format with {len(batch)} elements.")
            x = x.to(device)
            if mask is not None:
                logits = model(x, attention_mask=mask)
            else:
                logits = model(x)
            if logits.ndim == 1 or (logits.ndim == 2 and logits.shape[1] == 1):
                p = (torch.sigmoid(logits.view(-1)) > 0.5).long()
            else:
                p = logits.argmax(dim=1)
            preds.extend(p.cpu().tolist())
            trues.extend(y.cpu().tolist())
        return preds, trues

    def _update_reference(self, val):
        self._rolling_scores.append(val)
        if len(self._rolling_scores) > 256:
            self._rolling_scores.pop(0)

    def _normalize_score(self, val):
        if not self._rolling_scores:
            return val
        ref = np.percentile(self._rolling_scores, 50)
        return np.clip(val / (ref + 1e-8), 0, 1)

    # ----------------------------
    # Hybrid Consistency computation
    # ----------------------------
    def _compute_consistency(self, preds, trues) -> float:
        """Compute hybrid semantic consistency score."""
        if len(preds) == 0:
            return 0.0

        scores = []
        for p, t in zip(preds, trues):
            lp = self.label_map.get(p)
            lt = self.label_map.get(t)

            if lp == lt:
                scores.append(1.0)
                continue

            # --- Embedding similarity ---
            emb_sim = 0.0
            if self.entity_embeddings is not None and self.entity2id is not None:
                lp_key = str(lp).lower() if lp is not None else None
                lt_key = str(lt).lower() if lt is not None else None
                if lp_key in self.entity2id and lt_key in self.entity2id:
                    e1 = self.entity_embeddings[self.entity2id[lp_key]]
                    e2 = self.entity_embeddings[self.entity2id[lt_key]]
                    emb_sim = torch.nn.functional.cosine_similarity(e1, e2, dim=0).item()
                    emb_sim = float(np.clip(emb_sim, 0.0, 1.0))

            # --- Symbolic edge consistency ---
            sym_flag = 1.0 if (lp, lt) in self.kg_edges else 0.0

            # --- Combine ---
            hybrid_score = self.alpha * emb_sim + (1 - self.alpha) * sym_flag
            if emb_sim > self.sim_threshold or sym_flag > 0.5:
                scores.append(hybrid_score)
            else:
                scores.append(0.0)

        return float(np.mean(scores))

    # ----------------------------
    # Main compute() function
    # ----------------------------
    def compute(
        self,
        *,
        global_model: Optional[nn.Module] = None,
        client_delta: Optional[Dict[str, torch.Tensor]] = None,
        anchor_loader: Optional[DataLoader] = None,
        precomputed_skg: Optional[float] = None,
        client_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        if precomputed_skg is not None:
            S_KG = float(np.clip(precomputed_skg, 0.0, 1.0))
            return {"S_KG": S_KG, "L_KG": 1.0 - S_KG}

        if global_model is None or client_delta is None or anchor_loader is None:
            return {"S_KG": None, "L_KG": None}

        device = self.device or next(global_model.parameters()).device

        model_clone = self._clone_model(global_model)
        sd = model_clone.state_dict()
        for k, v in client_delta.items():
            if k in sd and sd[k].shape == v.shape:
                sd[k] = sd[k].to(device) + v.to(device)
        model_clone.load_state_dict(sd, strict=False)

        preds, trues = self._predict_labels(model_clone, anchor_loader, device)
        S_KG_raw = self._compute_consistency(preds, trues)
        self._update_reference(S_KG_raw)
        S_KG = self._normalize_score(S_KG_raw)
        L_KG = 1.0 - S_KG

        # EMA tracking
        if self.ema_skg is None:
            self.ema_skg = S_KG
        else:
            self.ema_skg = self.ema_decay * self.ema_skg + (1 - self.ema_decay) * S_KG

        drift_ratio = abs(S_KG - self.ema_skg) / (self.ema_skg + 1e-8)
        flag = "reject" if S_KG < self.reject_threshold else "pass"
        print(f"[KGCheck] Client {client_id}: S_KG={S_KG:.3f}, drift={drift_ratio:.3f}, flag={flag}")

        return {
            "S_KG": float(S_KG),
            "L_KG": float(L_KG),
            "drift_ratio": float(drift_ratio),
            "KG_flag": flag,
        }
