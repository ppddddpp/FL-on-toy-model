from typing import Optional, Sequence, Tuple, Dict, Any
import numpy as np


def _median_mad(arr: np.ndarray, eps: float = 1e-8) -> Tuple[float, float]:
    """
    Compute median and MAD robustly.
    Returns (median, mad) where mad >= eps.
    """
    if arr.size == 0:
        return 0.0, eps
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    if mad < eps:
        mad = eps
    return med, mad


class NormCheck:
    """
    Norm-based deviation detector.
    """

    def __init__(self, z_max: float = 3.0, eps: float = 1e-8):
        """
        Initialize NormCheck with configurable parameters.

        Parameters
        ----------
        z_max : float
            Scaling factor in denominator (controls sensitivity).
        eps : float
            Small epsilon to avoid divide-by-zero.
        """
        self.z_max = float(z_max)
        self.eps = float(eps)

    def compute(self,
                norm_i: float,
                all_norms: Optional[Sequence[float]] = None,
                precomputed_median: Optional[float] = None,
                precomputed_mad: Optional[float] = None) -> float:
        """
        Compute s_norm for a single client.

        Inputs:
            - norm_i: float, L2 norm of client's update
            - all_norms: optional list/array of all clients' norms for this round.
                    If provided, median and MAD are computed from it.
            - precomputed_median, precomputed_mad: if already computed, pass them to avoid recomputation.

        Returns:
            - s_norm in [0,1]
        """
        if precomputed_median is not None and precomputed_mad is not None:
            med = float(precomputed_median)
            mad = float(precomputed_mad)
            if mad < self.eps:
                mad = self.eps
        elif all_norms is not None:
            arr = np.asarray(all_norms, dtype=float)
            med, mad = _median_mad(arr, self.eps)
        else:
            # no context: cannot detect deviation -> return 0.0 (not anomalous)
            return 0.0

        s = abs(float(norm_i) - med) / (self.z_max * (mad + self.eps))
        # Clip to [0,1]
        if s < 0.0:
            s = 0.0
        elif s > 1.0:
            s = 1.0
        return float(s)

    def compute_batch(self, norms: Sequence[float]) -> Dict[str, Any]:
        """
        Compute median, MAD, and s_norms for a batch of norms.

        Returns dict:
            {
                "median": med,
                "mad": mad,
                "s_norms": np.array([...])   # shape (N,)
            }
        """
        arr = np.asarray(norms, dtype=float)
        if arr.size == 0:
            return {"median": 0.0, "mad": self.eps, "s_norms": np.array([], dtype=float)}

        med, mad = _median_mad(arr, self.eps)
        denom = self.z_max * (mad + self.eps)
        # compute vectorized
        s = np.abs(arr - med) / denom
        s = np.clip(s, 0.0, 1.0)
        return {"median": med, "mad": mad, "s_norms": s}

# For testing
if __name__ == "__main__":
    # Simulate normals and an outlier
    rng = np.random.RandomState(0)
    normal = rng.normal(loc=1.0, scale=0.05, size=20)  # typical norms ~1.0
    outlier_large = 10.0
    outlier_small = 0.01
    norms = np.concatenate([normal, [outlier_large, outlier_small]])
    nc = NormCheck(z_max=3.0)

    batch = nc.compute_batch(norms)
    print("median:", batch["median"], "mad:", batch["mad"])
    for idx, n in enumerate(norms):
        s = nc.compute(n, all_norms=norms)
        print(f"client {idx:02d} norm={n:.4f} -> s_norm={s:.4f}")

    # single-case: insufficient context -> returns 0 (safe fallback)
    print("no-context fallback:", nc.compute(2.0, all_norms=None))
