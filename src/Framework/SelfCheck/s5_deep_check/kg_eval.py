from typing import Optional, Sequence, Dict, Any, Set, Tuple
from copy import deepcopy
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


class KGConsistencyEvaluator:
    """
    KGConsistencyEvaluator checks whether model predictions respect
    semantic relations defined in a domain Knowledge Graph.
    """

    def __init__(
        self,
        kg_edges: Set[Tuple[str, str]],
        label_map: Dict[int, str],
        device: Optional[torch.device] = None,
        symmetric: bool = True,
        reject_threshold=0.4, 
        ema_decay=0.9
    ):
        """
        Initializes the KGConsistencyEvaluator.

        Parameters
        ----------
        kg_edges : set of (str, str)
            The directed or undirected edges of the Knowledge Graph.
            Each edge (a, b) means "a is semantically consistent with b".
        label_map : dict
            Maps numeric label indices (int) to label names (str) consistent with KG nodes.
        device : torch.device or None
            Device for inference (defaults to model's device).
        symmetric : bool
            If True, treat (a,b) and (b,a) as equivalent relations.
        reject_threshold : float
            Reject threshold for semantic consistency check
        ema_decay : float
            Exponential moving average decay
        """
        self.device = device
        self.label_map = dict(label_map)
        self.symmetric = bool(symmetric)
        self.reject_threshold = reject_threshold
        self.ema_decay = ema_decay
        self.ema_skg = None
        self._rolling_scores = []
        # normalize edges for symmetry
        if symmetric:
            edges_sym = set()
            for (a, b) in kg_edges:
                edges_sym.add((a, b))
                edges_sym.add((b, a))
            self.kg_edges = edges_sym
        else:
            self.kg_edges = set(kg_edges)

    @staticmethod
    def _clone_model(model: nn.Module) -> nn.Module:
        return deepcopy(model)

    @torch.no_grad()
    def _predict_labels(
        self, model: nn.Module, loader: DataLoader, device: torch.device
    ) -> tuple:
        """Run model on anchor dataset to get predictions and true labels."""
        model.eval().to(device)
        preds, trues = [], []
        for x, y in loader:
            x = x.to(device)
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

    def _compute_consistency(self, preds, trues) -> float:
        """Compute semantic consistency rate S_KG."""
        if len(preds) == 0:
            return 0.0
        consistent = 0
        for p, t in zip(preds, trues):
            lp = self.label_map.get(p)
            lt = self.label_map.get(t)
            if lp == lt:
                consistent += 1
            elif (lp, lt) in self.kg_edges:
                consistent += 1
        return consistent / len(preds)

    def compute(
        self,
        *,
        global_model: Optional[nn.Module] = None,
        client_delta: Optional[Dict[str, torch.Tensor]] = None,
        anchor_loader: Optional[DataLoader] = None,
        precomputed_skg: Optional[float] = None,
        client_id: Optional[str] = None,
    ) -> Dict[str, Any]:

        """
        Computes S_KG and L_KG given client delta and anchor data

        Args:
            global_model: global model to evaluate
            client_delta: client-side model update
            anchor_loader: data loader for anchor data
            precomputed_skg: precomputed S_KG value
            client_id: (optional) client ID for logging purposes

        Returns:
            dict with S_KG and L_KG values
        """
        # ---- client-side (precomputed) ----
        if precomputed_skg is not None:
            S_KG = float(np.clip(precomputed_skg, 0.0, 1.0))
            return {"S_KG": S_KG, "L_KG": 1.0 - S_KG}

        # ---- server-side full evaluation ----
        if global_model is None or client_delta is None or anchor_loader is None:
            return {"S_KG": None, "L_KG": None}

        # choose device
        if self.device is None:
            device = next(global_model.parameters()).device
        else:
            device = self.device

        # clone model + apply delta
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

        if self.ema_skg is None:
            self.ema_skg = S_KG
        else:
            self.ema_skg = self.ema_decay * self.ema_skg + (1 - self.ema_decay) * S_KG

        drift_ratio = abs(S_KG - self.ema_skg) / (self.ema_skg + 1e-8)

        flag = "reject" if S_KG < self.reject_threshold else "pass"
        print(f"[KGCheck] Client {client_id}: S_KG={S_KG:.3f}, drift={drift_ratio:.3f}, flag={flag}")

        S_KG_adj = max(0.0, S_KG * np.exp(-3 * drift_ratio))

        return {
            "S_KG": float(S_KG_adj),
            "L_KG": float(1.0 - S_KG),
            "drift_ratio": float(drift_ratio),
            "KG_flag": flag
        }

    def compute_batch(self, skg_list: Sequence[float]) -> Dict[str, Any]:
        """Batch version: normalize a list of precomputed S_KG values."""
        if skg_list is None or len(skg_list) == 0:
            return {"reference": None, "S_KG": np.array([], dtype=float), "L_KG": np.array([], dtype=float)}

        arr = np.asarray(skg_list, dtype=float)
        arr = np.clip(arr, 0.0, 1.0)
        L = 1.0 - arr
        return {"reference": float(np.median(arr)), "S_KG": arr, "L_KG": L}


# For testing
if __name__ == "__main__":
    print("Running KGConsistencyEvaluator self-test...")

    # Toy KG with simple semantic relations
    kg_edges = {("cat", "feline"), ("dog", "canine"), ("feline", "animal"), ("canine", "animal")}
    label_map = {0: "cat", 1: "dog", 2: "feline", 3: "canine", 4: "animal"}

    # Fake model returning deterministic logits (to simulate predictions)
    class ToyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(2, 5)
        def forward(self, x):
            return self.lin(x)

    model = ToyModel()
    delta = {k: 0.01 * torch.randn_like(v) for k, v in model.state_dict().items()}
    loader = [(torch.randn(8, 2), torch.randint(0, 5, (8,))) for _ in range(3)]

    evaluator = KGConsistencyEvaluator(kg_edges, label_map)
    res = evaluator.compute(global_model=model, client_delta=delta, anchor_loader=loader)
    print(res)

    # Batch example
    batch = evaluator.compute_batch([0.8, 0.7, 0.5, 1.0])
    print(batch)
