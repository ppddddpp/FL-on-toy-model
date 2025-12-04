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

        if self.entity_embeddings is not None and self.device is not None:
            self.entity_embeddings = self.entity_embeddings.to(self.device)

        # --- normalize entity embeddings ---
        if self.entity_embeddings is not None:
            eps = 1e-6
            norm = torch.norm(self.entity_embeddings, dim=1, keepdim=True) + eps
            self.entity_embeddings = self.entity_embeddings / norm

        # --- normalize edges ---
        if symmetric:
            edges_sym = set()
            for e in kg_edges:
                if len(e) == 3:
                    a, _, b = e
                else:
                    a, b = e

                # Convert both ends to lowercase strings safely
                a = str(a).lower()
                b = str(b).lower()

                edges_sym.add((a, b))
                edges_sym.add((b, a))

            self.kg_edges = edges_sym
        else:
            cleaned = set()
            for e in kg_edges:
                if len(e) == 3:
                    a, _, b = e
                else:
                    a, b = e

                a = str(a).lower()
                b = str(b).lower()

                cleaned.add((a, b))
            self.kg_edges = cleaned
        
        # build adjacency list
        self.neighbors = {}
        for a, b in self.kg_edges:
            self.neighbors.setdefault(a, set()).add(b)

        # normalize entity2id keys to lowercase for robust lookup
        if self.entity2id is not None:
            try:
                self.entity2id = {str(k).lower(): int(v) for k, v in self.entity2id.items()}
            except Exception:
                # keep as-is if conversion fails
                self.entity2id = {str(k).lower(): v for k, v in self.entity2id.items()}

        # precompute mean embedding for fallback (if embeddings exist)
        if self.entity_embeddings is not None and self.entity_embeddings.shape[0] > 0:
            with torch.no_grad():
                mean_emb = self.entity_embeddings.mean(dim=0, keepdim=True)
                mean_emb = mean_emb / (mean_emb.norm(p=2, dim=1, keepdim=True) + 1e-12)
                self._mean_entity_emb = mean_emb.view(-1)
        else:
            self._mean_entity_emb = None

        # smoothing / fallback params
        self.smoothing_alpha = 0.7   # keep 0.7 to S_new = 0.7*prev + 0.3*cur
        self.smoothed_S = None       # will hold running smoothed S_KG
        self.expect_delta = True

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
        if len(self._rolling_scores) < 8:
            return float(np.clip(val, 0.0, 1.0))

        mean = np.mean(self._rolling_scores)
        std = np.std(self._rolling_scores) + 1e-8

        z = (val - mean) / std
        z = float(np.clip(0.5 + 0.1 * z, 0.0, 1.0))
        return z

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

            lp = lp.lower() if lp else None
            lt = lt.lower() if lt else None
            if lp == lt:
                scores.append(1.0)
                continue

            # --- Embedding similarity with fallback ---
            emb_sim = 0.0
            if self.entity_embeddings is not None and self.entity2id is not None:
                lp_key = str(lp).lower() if lp is not None else None
                lt_key = str(lt).lower() if lt is not None else None

                e1 = None
                e2 = None
                if lp_key in self.entity2id:
                    idx1 = self.entity2id[lp_key]
                    e1 = self.entity_embeddings[idx1]
                if lt_key in self.entity2id:
                    idx2 = self.entity2id[lt_key]
                    e2 = self.entity_embeddings[idx2]

                # fallback handling
                if e1 is None and self._mean_entity_emb is not None:
                    e1 = self._mean_entity_emb
                if e2 is None and self._mean_entity_emb is not None:
                    e2 = self._mean_entity_emb

                # if have *both* vectors now, compute cosine
                if e1 is not None and e2 is not None:
                    # ensure both are normalized
                    e1n = e1 / (e1.norm(p=2) + 1e-12)
                    e2n = e2 / (e2.norm(p=2) + 1e-12)
                    emb_sim = float(torch.nn.functional.cosine_similarity(e1n, e2n, dim=0).item())
                    emb_sim = max(0.0, min(1.0, emb_sim))

            # --- Symbolic edge consistency ---
            hop1 = (lp, lt) in self.kg_edges
            hop2 = lp in self.neighbors and any(n2 == lt for n1 in self.neighbors[lp] for n2 in self.neighbors.get(n1, []))

            if hop1:
                sym_flag = 1.0
            elif hop2:
                sym_flag = 0.5
            else:
                sym_flag = 0.0

            # --- Combine ---
            hybrid_score = self.alpha * emb_sim + (1 - self.alpha) * sym_flag
            hybrid_score = float(np.clip(hybrid_score, 0.0, 1.0))

            if emb_sim > self.sim_threshold or sym_flag > 0.5:
                scores.append(hybrid_score)
            else:
                scores.append(0.0)

        return float(np.mean(scores))
    
    def calibrate_baseline(self, baseline_scores: Sequence[float]):
        baseline_scores = np.array(baseline_scores)
        mean = baseline_scores.mean()
        std = baseline_scores.std()
        raw_thr = mean - 2 * std
        self.reject_threshold = float(np.clip(raw_thr, 0.0, 1.0))
        print(f"[KGCheck] Calibrated reject_threshold = {self.reject_threshold:.3f}")

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
            else:
                # accept replacement if shapes differ but sizes compatible etc.
                sd[k] = v.to(device)
        model_clone.load_state_dict(sd, strict=False)

        preds, trues = self._predict_labels(model_clone, anchor_loader, device)
        S_KG_raw = self._compute_consistency(preds, trues)
        self._update_reference(S_KG_raw)
        S_KG = self._normalize_score(S_KG_raw)
        L_KG = 1.0 - S_KG

        # smoothing: combine previous smoothed value and current normalized value
        if self.smoothed_S is None:
            self.smoothed_S = float(S_KG)
        else:
            # S_KG is normalized already (0..1). use smoothing_alpha:
            alpha = float(self.smoothing_alpha)
            self.smoothed_S = alpha * float(self.smoothed_S) + (1.0 - alpha) * float(S_KG)

        # use smoothed_S for downstream decisions / logging
        S_KG_smoothed = float(self.smoothed_S)

        prev_ema = self.ema_skg
        base = max(prev_ema if prev_ema is not None else 0.1, 0.1)
        drift_ratio = abs(S_KG_smoothed - (prev_ema if prev_ema is not None else S_KG_smoothed)) / base

        # EMA tracking
        if self.ema_skg is None:
            self.ema_skg = S_KG_smoothed
        else:
            self.ema_skg = self.ema_decay * self.ema_skg + (1 - self.ema_decay) * S_KG_smoothed

        flag = "reject" if S_KG_smoothed < self.reject_threshold else "pass"
        print(f"[KGCheck] Client {client_id}: S_KG={S_KG_smoothed:.3f}, drift={drift_ratio:.3f}, flag={flag}")

        return {
            "S_KG": float(S_KG_smoothed),
            "L_KG": float(1.0 - S_KG_smoothed),
            "drift_ratio": float(drift_ratio),
            "KG_flag": flag,
        }