from typing import Optional, Sequence, Dict, Any, Literal
import numpy as np
import torch

def _normalize(vec: torch.Tensor) -> torch.Tensor:
    """Return unit vector (avoid zero division)."""
    norm = torch.norm(vec)
    if norm.item() < 1e-12:
        return vec * 0.0
    return vec / norm


def _mean_direction(vectors: Sequence[torch.Tensor]) -> Optional[torch.Tensor]:
    """Compute mean direction of a list of tensors."""
    if not vectors:
        return None
    stacked = torch.stack(vectors, dim=0)
    mean_vec = stacked.mean(dim=0)
    return _normalize(mean_vec)


def _median_direction(vectors: Sequence[torch.Tensor]) -> Optional[torch.Tensor]:
    """
    Compute elementwise median of a list of vectors.
    Slightly more robust than mean for skewed sets.
    """
    if not vectors:
        return None
    stacked = torch.stack(vectors, dim=0)
    med = torch.median(stacked, dim=0).values
    return _normalize(med)


class CosineCheck:
    """
    Directional deviation detector.
    """

    def __init__(self, eps: float = 1e-8, direction_agg: Literal["mean","median"] = "mean"):
        """
        Initialize CosineCheck.

        Parameters
        ----------
        eps : float
            Small epsilon to avoid division by zero.
        direction_agg : {"mean","median"}
            How to compute reference direction across clients.
        """
        self.eps = float(eps)
        self.direction_agg = direction_agg

    def compute(
        self,
        delta_i: torch.Tensor,
        all_deltas: Optional[Sequence[torch.Tensor]] = None,
        precomputed_ref: Optional[torch.Tensor] = None,
    ) -> float:
        """
        Compute s_cos for one client.

        Inputs:
            - delta_i: client's flattened update tensor
            - all_deltas: list of all clients' flattened deltas (optional)
            - precomputed_ref: precomputed reference direction (optional)

        Returns:
            - s_cos in [0,1]
        """
        if delta_i is None or delta_i.numel() == 0:
            return 0.0

        # get reference direction
        if precomputed_ref is not None:
            ref = precomputed_ref
        elif all_deltas is not None:
            if self.direction_agg == "median":
                ref = _median_direction(all_deltas)
            else:
                ref = _mean_direction(all_deltas)
        else:
            # no reference, cannot compute deviation
            return 0.0

        if ref is None or torch.norm(ref).item() < 1e-12:
            return 0.0

        # compute cosine
        num = torch.dot(delta_i, ref).item()
        denom = (torch.norm(delta_i).item() * torch.norm(ref).item()) + self.eps
        cos_i = num / denom
        # map to deviation score
        s_cos = 1.0 - (cos_i + 1.0) / 2.0
        s_cos = float(np.clip(s_cos, 0.0, 1.0))
        return s_cos

    def compute_batch(
        self,
        deltas: Sequence[torch.Tensor],
        precomputed_ref: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Compute reference direction and s_cos for all clients.

        Returns dict:
            {
                "reference": ref_vec,
                "s_cos": np.array([...])
            }
        """
        if not deltas:
            return {"reference": None, "s_cos": np.array([], dtype=float)}

        ref = precomputed_ref
        if ref is None:
            if self.direction_agg == "median":
                ref = _median_direction(deltas)
            else:
                ref = _mean_direction(deltas)

        if ref is None:
            return {"reference": None, "s_cos": np.zeros(len(deltas), dtype=float)}

        ref_norm = torch.norm(ref).item() + self.eps
        s_list = []
        for delta_i in deltas:
            if delta_i is None or delta_i.numel() == 0:
                s_list.append(0.0)
                continue
            num = torch.dot(delta_i, ref).item()
            denom = (torch.norm(delta_i).item() * ref_norm) + self.eps
            cos_i = num / denom
            s_cos = 1.0 - (cos_i + 1.0) / 2.0
            s_list.append(float(np.clip(s_cos, 0.0, 1.0)))

        return {"reference": ref, "s_cos": np.array(s_list, dtype=float)}

# for testing
if __name__ == "__main__":
    rng = np.random.RandomState(0)
    # create 10 normal updates around same direction, one flipped
    base = torch.tensor(rng.normal(size=100).astype(np.float32))
    base = base / torch.norm(base)
    deltas = []
    
    for i in range(9):
        noise = torch.tensor(rng.normal(scale=0.01, size=100).astype(np.float32))
        d = base + noise
        deltas.append(d)
    # add a flipped one (opposite direction)
    deltas.append(-base)

    cc = CosineCheck(direction_agg="mean")
    batch = cc.compute_batch(deltas)
    s_cos = batch["s_cos"]
    for idx, s in enumerate(s_cos):
        print(f"client {idx:02d}: s_cos={s:.4f}")
    print("reference norm:", torch.norm(batch["reference"]).item())
