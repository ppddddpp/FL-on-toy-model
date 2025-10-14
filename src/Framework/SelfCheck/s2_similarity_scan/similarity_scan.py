import numpy as np
from typing import Dict, List, Tuple

class SimilarityScanDetector:
    """
    Sampled Top-k Similarity Scan.
    Finds clients whose updates are tightly aligned with a few others
    but deviate from the global/population direction.
    """

    def __init__(self,
                    sketch_dim: int = 256,
                    sample_size: int = 100,
                    topk: int = 10,
                    sim_threshold: float = 0.85,
                    pop_threshold: float = 0.70,
                    random_state: int = 42):
        """
        Initialize SimilarityScanDetector.

        Parameters
        ----------
        sketch_dim : int, optional
            Dimensionality of compressed sketches (default: 256).
        sample_size : int, optional
            Number of random samples for each client (default: 100).
        topk : int, optional
            Number of top similarity values to consider (default: 10).
        sim_threshold : float, optional
            Threshold for considering two clients similar (default: 0.85).
        pop_threshold : float, optional
            Threshold for considering a client's update aligned with population (default: 0.70).
        random_state : int, optional
            Random seed for reproducibility (default: 42).
        """
        self.sketch_dim = sketch_dim
        self.sample_size = sample_size
        self.topk = topk
        self.sim_threshold = sim_threshold
        self.pop_threshold = pop_threshold
        self.rng = np.random.default_rng(random_state)
        # deterministic projection matrix
        self.proj_matrix = self.rng.standard_normal((sketch_dim,))

    def run(self,
            client_deltas: Dict[str, np.ndarray],
            candidate_ids: List[str] = None
            ) -> Tuple[List[str], Dict]:
        """
        Perform sampled similarity scan.

        Parameters
        ----------
        client_deltas : dict {client_id: flattened_update}
        candidate_ids : optional list of ids from Stage 1 (limit scope)

        Returns
        -------
        flagged_clients : list[str]
        stats : dict with per-client metrics
        """
        ids = candidate_ids or list(client_deltas.keys())
        sketches = self._make_sketches(client_deltas, ids)
        pop_ref = self._median_vector(sketches)

        sims_pop = {cid: self._cosine(sketches[cid], pop_ref) for cid in ids}
        topk_mean = self._sampled_topk(sketches, ids)

        # decision rule
        flagged = [cid for cid in ids
                    if topk_mean[cid] > self.sim_threshold
                    and sims_pop[cid] < self.pop_threshold]

        stats = {
            "topk_mean": topk_mean,
            "pop_similarity": sims_pop,
            "flagged": flagged,
            "sketch_dim": self.sketch_dim
        }
        return flagged, stats

    def _make_sketches(self, deltas, ids):
        """Random projection compression — use the same indices (or projection) for every client in one run call."""
        sketches = {}
        # determine base vector size
        first_v = next(iter(deltas.values()))
        dim = first_v.reshape(-1).size

        if self.sketch_dim >= dim:
            # no compression possible; return normalized full vectors (may be costly)
            idx = np.arange(dim)
        else:
            # sample index set once per call so sketches are comparable
            idx = self.rng.choice(dim, size=self.sketch_dim, replace=False)

        for cid in ids:
            v = deltas[cid].reshape(-1)
            sketch = v[idx].astype(np.float32)
            norm = np.linalg.norm(sketch) + 1e-12
            sketches[cid] = sketch / norm
        return sketches

    def _median_vector(self, sketches):
        X = np.stack(list(sketches.values()))
        return np.median(X, axis=0) / (np.linalg.norm(np.median(X, axis=0)) + 1e-12)

    def _cosine(self, a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-12))

    def _sampled_topk(self, sketches, ids):
        """Estimate each client's mean of top-k similarities via random sampling."""
        N = len(ids)
        topk_mean = {}
        for i, cid in enumerate(ids):
            others = [j for j in ids if j != cid]
            samp = self.rng.choice(others, size=min(self.sample_size, N-1), replace=False)
            sims = [self._cosine(sketches[cid], sketches[o]) for o in samp]
            sims.sort(reverse=True)
            topk_mean[cid] = float(np.mean(sims[:self.topk]))
        return topk_mean
