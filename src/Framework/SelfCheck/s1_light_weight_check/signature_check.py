from typing import Optional, Sequence, Dict, Any
import numpy as np
import torch


class SignatureCheck:
    """
    Random-projection based signature distance check.
    """

    def __init__(self, sig_dim: int = 64, seed: int = 42,
                    k_tolerance: float = 3.0, eps: float = 1e-8):
        """
        Initialize SignatureCheck instance.

        Parameters
        ----------
        sig_dim : int, optional
            Dimensionality of compressed signatures (typically 64-256).
            Defaults to 64.
        seed : int, optional
            Random seed for projection matrix reproducibility.
            Defaults to 42.
        k_tolerance : float, optional
            Multiplier controlling deviation tolerance (2-4 typical).
            Defaults to 3.0.
        eps : float, optional
            Small epsilon for numerical stability.
            Defaults to 1e-8.
        """
        self.sig_dim = int(sig_dim)
        self.seed = int(seed)
        self.k_tolerance = float(k_tolerance)
        self.eps = float(eps)
        self._projection_matrix_cache: Optional[np.ndarray] = None
        self._proj_dim_in: Optional[int] = None

    def _get_projection_matrix(self, dim_in: int) -> np.ndarray:
        """
        Return a cached random projection matrix for input dimension `dim_in`.
        Regenerates if dimension changes or cache empty.
        """
        if (self._projection_matrix_cache is None) or (self._proj_dim_in != dim_in):
            rng = np.random.RandomState(self.seed)
            W = rng.normal(size=(self.sig_dim, dim_in)).astype(np.float32)
            W /= np.sqrt(self.sig_dim)  # Johnson–Lindenstrauss scaling
            self._projection_matrix_cache = W
            self._proj_dim_in = dim_in
        return self._projection_matrix_cache

    def compute_signature(self, delta: torch.Tensor) -> Optional[np.ndarray]:
        """
        Compute compressed signature for a given flattened delta tensor.
        Returns None if delta is empty. Otherwise, returns sigma_i belonging to R^{sig_dim}.
        """
        if delta is None or delta.numel() == 0:
            return None
        v = delta.detach().cpu().numpy().astype(np.float32)
        W = self._get_projection_matrix(v.shape[0])
        sig = np.dot(W, v)  # shape (sig_dim,)
        return sig

    def compute(
        self,
        sig_i: np.ndarray,
        all_sigs: Optional[Sequence[np.ndarray]] = None,
        precomputed_median_sig: Optional[np.ndarray] = None,
        precomputed_median_dist: Optional[float] = None,
    ) -> float:
        """
        Compute s_sig for a single client.

        Inputs:
            - sig_i: compressed signature vector for this client.
            - all_sigs: list of all clients' signatures (optional).
            - precomputed_median_sig: if already computed, supply it.
            - precomputed_median_dist: precomputed median distance value (optional).

        Returns:
            s_sig in [0,1]
        """
        if sig_i is None:
            return 0.0

        # determine median signature and median distance
        if precomputed_median_sig is not None:
            med_sig = precomputed_median_sig
        elif all_sigs is not None and len(all_sigs) > 0:
            med_sig = np.median(np.stack(all_sigs, axis=0), axis=0)
        else:
            return 0.0

        # compute distance of this sig to median sig
        d_i = float(np.linalg.norm(sig_i - med_sig))

        if precomputed_median_dist is not None:
            med_d = max(precomputed_median_dist, self.eps)
        elif all_sigs is not None and len(all_sigs) > 0:
            all_d = np.linalg.norm(np.stack(all_sigs, axis=0) - med_sig, axis=1)
            med_d = max(float(np.median(all_d)), self.eps)
        else:
            med_d = self.eps

        s_sig = d_i / (self.k_tolerance * med_d)
        s_sig = float(np.clip(s_sig, 0.0, 1.0))
        return s_sig

    def compute_batch(
        self, signatures: Sequence[np.ndarray]
    ) -> Dict[str, Any]:
        """
        Compute median signature and all s_sig scores for a batch.

        Returns dict:
            {
                "median_sig": np.ndarray,
                "median_dist": float,
                "s_sig": np.ndarray  # shape (N,)
            }
        """
        if not signatures:
            return {
                "median_sig": None,
                "median_dist": self.eps,
                "s_sig": np.array([], dtype=float)
            }

        sigs = np.stack(signatures, axis=0)
        med_sig = np.median(sigs, axis=0)
        dists = np.linalg.norm(sigs - med_sig[None, :], axis=1)
        med_d = max(float(np.median(dists)), self.eps)
        s = dists / (self.k_tolerance * med_d)
        s = np.clip(s, 0.0, 1.0)

        return {"median_sig": med_sig, "median_dist": med_d, "s_sig": s}


# For testing
if __name__ == "__main__":
    rng = np.random.RandomState(0)
    # simulate 10 clients with similar updates + 1 abnormal
    base = rng.normal(scale=0.01, size=(512,)).astype(np.float32)

    deltas = []
    for i in range(9):
        noise = rng.normal(scale=0.001, size=(512,)).astype(np.float32)
        deltas.append(torch.tensor(base + noise))
    # add outlier
    deltas.append(torch.tensor(rng.normal(scale=0.05, size=(512,)).astype(np.float32)))

    sc = SignatureCheck(sig_dim=64, seed=42, k_tolerance=3.0)
    sigs = [sc.compute_signature(d) for d in deltas]
    batch = sc.compute_batch(sigs)

    print("median_dist:", batch["median_dist"])
    for idx, s in enumerate(batch["s_sig"]):
        print(f"client {idx:02d}: s_sig={s:.4f}")
