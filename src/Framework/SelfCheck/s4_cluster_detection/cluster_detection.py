import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict, deque
import hashlib

class ClusterDetector:
    """
    Graph / Cluster Anomaly Detection.
    Builds a similarity graph of candidate clients (sketches) and
    detects compact collusive clusters. Tracks persistence over rounds.
    """

    def __init__(self,
                    edge_threshold: float = 0.80,
                    min_cluster_size: int = 2,
                    max_cluster_size: int = 10,
                    persistence_window: int = 10,
                    persistence_threshold: float = 0.3,
                    intra_sim_thresh: float = 0.85,
                    pop_sim_thresh: float = 0.70,
                    random_state: int = 42):
        """
        Initialize the ClusterDetector instance.

        Parameters
        ----------
        edge_threshold : float
            Similarity threshold for connecting clients in the graph.
        min_cluster_size : int
            Minimum size of a cluster to be considered.
        max_cluster_size : int
            Maximum size of a cluster to be considered.
        persistence_window : int
            Number of recent rounds to track persistence.
        persistence_threshold : float
            Minimum similarity threshold for a client to be considered
            persistent.
        intra_sim_thresh : float
            Minimum similarity threshold for a client to be considered
            part of a cluster.
        pop_sim_thresh : float
            Minimum similarity threshold for a client to be considered
            similar to the population.
        random_state : int
            Random seed for reproducibility.

        Returns
        -------
        None
        """
        self.edge_threshold = edge_threshold
        self.min_cluster_size = min_cluster_size
        self.max_cluster_size = max_cluster_size
        self.persistence_window = persistence_window
        self.persistence_threshold = persistence_threshold
        self.intra_sim_thresh = intra_sim_thresh
        self.pop_sim_thresh = pop_sim_thresh
        self.rng = np.random.default_rng(random_state)
        # persistence tracking: cluster_signature -> deque of round_ids
        self.history = defaultdict(lambda: deque(maxlen=self.persistence_window))

    # ---------- Public API ----------
    def run(self,
            sketches: Dict[str, np.ndarray],
            candidate_ids: List[str],
            round_id: int,
            pop_ref: np.ndarray = None
            ) -> Tuple[List[Dict], Dict]:
        """
        Run Stage 3 clustering.

        Parameters
        ----------
        sketches : dict {client_id: compressed vector}
        candidate_ids : list of client IDs to analyze
        round_id : int, current round
        pop_ref : optional np.ndarray, population reference vector

        Returns
        -------
        clusters : list[dict]  each containing cluster metrics and action
        stats : dict  debug info for logging
        """
        if not candidate_ids:
            return [], {"info": "no candidates"}

        # Build similarity matrix
        S, ids = self._pairwise_sim_matrix(sketches, candidate_ids)

        # Build adjacency
        A = (S >= self.edge_threshold).astype(float) * S

        # Cluster extraction (simple DBSCAN-like grouping)
        clusters_idx = self._find_clusters(A)

        # Compute metrics
        clusters_info = []
        pop_ref = self._compute_pop_ref(sketches, ids, pop_ref)
        for cluster in clusters_idx:
            members = [ids[i] for i in cluster]
            intra_sim = self._mean_pairwise(S, cluster)
            pop_sim = np.mean([self._cosine(sketches[c], pop_ref) for c in members])
            mean_vec = np.mean([sketches[c] for c in members], axis=0)
            directional_dev = 1 - self._cosine(mean_vec, pop_ref)

            # persistence
            signature = self._signature(members)
            self.history[signature].append(round_id)
            persistence = len(self.history[signature]) / self.persistence_window

            # decision
            action, conf = self._decide(intra_sim, pop_sim, persistence)

            clusters_info.append({
                "cluster_id": signature,
                "members": members,
                "size": len(members),
                "intra_sim": float(intra_sim),
                "pop_sim": float(pop_sim),
                "directional_deviation": float(directional_dev),
                "persistence": float(persistence),
                "confidence": float(conf),
                "action_reco": action
            })

        stats = {"adjacency": A, "sim_matrix": S, "num_clusters": len(clusters_info)}
        return clusters_info, stats

    def _pairwise_sim_matrix(self, sketches, ids):
        n = len(ids)
        S = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                s = self._cosine(sketches[ids[i]], sketches[ids[j]])
                S[i,j] = S[j,i] = s
        np.fill_diagonal(S, 1.0)
        return S, ids

    def _find_clusters(self, A):
        """
        Simple DBSCAN-style grouping using adjacency threshold.
        """
        visited = set()
        clusters = []
        n = A.shape[0]
        for i in range(n):
            if i in visited:
                continue
            # neighbors above threshold
            neigh = set(np.where(A[i] >= self.edge_threshold)[0])
            if len(neigh) < self.min_cluster_size:
                continue
            # expand cluster
            cluster = set(neigh)
            added = True
            while added:
                added = False
                for j in list(cluster):
                    new = set(np.where(A[j] >= self.edge_threshold)[0])
                    new_to_add = new - cluster
                    if new_to_add:
                        cluster |= new_to_add
                        added = True
            for j in cluster:
                visited.add(j)
            if self.min_cluster_size <= len(cluster) <= self.max_cluster_size:
                clusters.append(sorted(list(cluster)))
        return clusters

    def _compute_pop_ref(self, sketches, ids, pop_ref):
        if pop_ref is not None:
            return pop_ref / (np.linalg.norm(pop_ref) + 1e-12)
        X = np.stack([sketches[i] for i in ids])
        v = np.median(X, axis=0)
        return v / (np.linalg.norm(v) + 1e-12)

    def _cosine(self, a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-12))

    def _mean_pairwise(self, S, idxs):
        if len(idxs) < 2:
            return 1.0
        vals = []
        for i in range(len(idxs)):
            for j in range(i+1, len(idxs)):
                vals.append(S[idxs[i], idxs[j]])
        return np.mean(vals)

    def _signature(self, members):
        m_sorted = sorted(members)
        return hashlib.sha1("_".join(m_sorted).encode()).hexdigest()[:12]

    def _decide(self, intra_sim, pop_sim, persistence):
        conf = (
            0.5 * intra_sim +
            0.3 * (1 - pop_sim) +
            0.2 * persistence
        )
        if intra_sim >= self.intra_sim_thresh and pop_sim <= self.pop_sim_thresh and persistence >= self.persistence_threshold:
            return "escalate", conf
        elif intra_sim >= 0.75 and persistence >= 0.2:
            return "monitor", conf
        else:
            return "ignore", conf
