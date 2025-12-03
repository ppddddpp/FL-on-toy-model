import numpy as np
import copy

class SybilAmplificationAttack:
    """
    Unified Sybil attack class supporting 3 modes:
        1. static          - amplify own gradient
        2. leader          - copy strongest attacker gradient
        3. coordinated     - evolving shared direction
    """

    def __init__(
        self,
        amplification_factor=5.0,
        sybil_mode="static",       # static | leader | coordinated
        shared_vector=None,        # for collusion strategies
        fake_data_size=500,
        alpha=0.8,                 # smoothing for coordinated mode
        collusion=True
    ):
        self.amplification_factor = amplification_factor
        self.sybil_mode = sybil_mode
        self.shared_vector = shared_vector
        self.fake_data_size = fake_data_size
        self.alpha = alpha
        self.collusion = collusion

        # For coordinated evolving attack
        self.prev_shared = None

    def update_shared_vector(self, sybil_updates: list):
        """
        Called only for leader_mode or coordinated_mode.
        sybil_updates = list of dict gradients from Sybils this round
        """

        if self.sybil_mode == "leader":
            # pick the one with largest L2-norm
            best = None
            best_norm = -1
            for g in sybil_updates:
                norm = sum(np.linalg.norm(v) for v in g.values())
                if norm > best_norm:
                    best_norm = norm
                    best = g
            self.shared_vector = best

        elif self.sybil_mode == "coordinated":
            # mean of sybil gradients
            avg = {}
            keys = sybil_updates[0].keys()
            for k in keys:
                avg[k] = np.mean([g[k] for g in sybil_updates], axis=0)

            # smooth over time
            if self.prev_shared is None:
                self.prev_shared = avg
            else:
                for k in avg:
                    avg[k] = self.alpha * self.prev_shared[k] + (1 - self.alpha) * avg[k]
                self.prev_shared = avg

            self.shared_vector = self.prev_shared

    def apply(self, benign_update, client_metadata=None):
        """
        Apply attack to a single Sybil update.
        """

        update = copy.deepcopy(benign_update)
        metadata = client_metadata.copy() if client_metadata else {}
        metadata["num_samples"] = self.fake_data_size

        # ---- Mode 1: static (non-collusive) ----
        if self.sybil_mode == "static":
            for k in update:
                update[k] = update[k] * self.amplification_factor
            return update, metadata

        # ---- Mode 2 & 3 require shared_vector ----
        if self.shared_vector is None:
            raise ValueError("shared_vector is required for leader/coordinated sybil modes")

        # ---- Mode 2: leader-follower ----
        if self.sybil_mode == "leader":
            for k in update:
                update[k] = self.shared_vector[k] * self.amplification_factor
            return update, metadata

        # ---- Mode 3: coordinated evolving ----
        if self.sybil_mode == "coordinated":
            for k in update:
                update[k] = self.shared_vector[k] * self.amplification_factor
            return update, metadata

        raise ValueError(f"Unknown sybil_mode: {self.sybil_mode}")
