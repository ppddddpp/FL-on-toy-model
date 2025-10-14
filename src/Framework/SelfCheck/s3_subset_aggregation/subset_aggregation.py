import numpy as np
from typing import Dict, List, Tuple

class SubsetAggregationDetector:
    """
    Randomized Subset-Aggregation Divergence Detector.
    Detects subsets whose aggregate updates deviate strongly
    from the population mean update.
    """

    def __init__(self,
                    num_subsets: int = 6,
                    subset_size: int = 0,
                    distance_metric: str = "cosine",
                    outlier_kappa: float = 2.5,
                    random_state: int = 2709):
        """
        Initialize SubsetAggregationDetector.

        Parameters
        ----------
        num_subsets : int, optional
            Number of subsets to sample from the population.
            Defaults to 6.
        subset_size : int, optional
            Size of each subset, or 0 to automatically set to
            max(10, N // num_subsets) where N is the population size.
            Defaults to 0.
        distance_metric : str, optional
            Distance metric to use when computing pairwise distances.
            Supports "cosine" or any other valid distance metric.
            Defaults to "cosine".
        outlier_kappa : float, optional
            Multiplier for detecting outliers in the subset distances.
            Defaults to 2.5.
        random_state : int, optional
            Seed for random number generation.
            Defaults to 2709.
        """
        self.num_subsets = num_subsets
        self.subset_size = subset_size
        self.distance_metric = distance_metric
        self.outlier_kappa = outlier_kappa
        self.rng = np.random.default_rng(random_state)

    # ---------- Core Public API ----------
    def run(self,
            client_deltas: Dict[str, np.ndarray],
            client_weights: Dict[str, float] = None
            ) -> Tuple[List[str], Dict]:
        """
        Perform subset aggregation divergence test.

        Parameters
        ----------
        client_deltas : dict {client_id: flattened_update_vector}
        client_weights : dict {client_id: num_samples} (optional)

        Returns
        -------
        flagged_clients : list of str
            Clients belonging to suspicious subsets.
        stats : dict
            Detailed subset statistics for ledger/debugging.
        """
        client_ids = list(client_deltas.keys())
        N = len(client_ids)
        if self.subset_size == 0:
            self.subset_size = max(10, N // self.num_subsets)

        # 1. Random partitioning
        subsets = self._sample_subsets(client_ids, N)

        # 2. Compute subset aggregates
        subset_updates = []
        for subset in subsets:
            agg = self._aggregate_subset(subset, client_deltas, client_weights)
            subset_updates.append(agg)

        # 3. Measure divergence
        distances = self._pairwise_distances(subset_updates)
        outlier_indices = self._detect_outliers(distances)

        # 4. Collect flagged clients
        flagged = [cid for i in outlier_indices for cid in subsets[i]]

        stats = {
            "num_subsets": self.num_subsets,
            "subset_size": self.subset_size,
            "distances": distances,
            "outliers": outlier_indices,
        }
        return list(set(flagged)), stats

    # ---------- Internal Helpers ----------
    def _sample_subsets(self, ids: List[str], N: int) -> List[List[str]]:
        # Randomly partition IDs into num_subsets groups (possibly overlapping)
        self.rng.shuffle(ids)
        subsets = []
        for i in range(self.num_subsets):
            start = i * self.subset_size % N
            end = start + self.subset_size
            subset = [ids[j % N] for j in range(start, end)]
            subsets.append(subset)
        return subsets

    def _aggregate_subset(self, subset, deltas, weights):
        vecs = [deltas[cid].reshape(-1) for cid in subset]
        if weights is None:
            agg = np.mean(vecs, axis=0)
        else:
            ws = np.array([weights.get(cid, 1.0) for cid in subset])
            ws = ws / np.sum(ws)
            agg = np.sum([w * v for w, v in zip(ws, vecs)], axis=0)
        return agg / np.linalg.norm(agg)  # normalize for cosine

    def _pairwise_distances(self, subset_updates):
        m = len(subset_updates)
        D = np.zeros((m, m))
        for i in range(m):
            for j in range(i+1, m):
                if self.distance_metric == "cosine":
                    sim = np.dot(subset_updates[i], subset_updates[j])
                    D[i,j] = D[j,i] = 1 - sim
                else:  # L2 distance
                    diff = subset_updates[i] - subset_updates[j]
                    D[i,j] = D[j,i] = np.linalg.norm(diff)
        return D

    def _detect_outliers(self, D):
        # Compute mean distance of each subset to others
        mean_dist = np.mean(D, axis=1)
        median = np.median(mean_dist)
        mad = np.median(np.abs(mean_dist - median)) + 1e-12
        thresh = median + self.outlier_kappa * mad
        return np.where(mean_dist > thresh)[0].tolist()
