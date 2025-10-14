from typing import Optional, Dict, Any
import numpy as np
from collections import defaultdict, deque
import torch

class TemporalCheck:
    """
    Temporal instability detector.
    """

    def __init__(self, window_size: int = 5, V_max: float = 0.1, eps: float = 1e-12):
        """
        Parameters
        ----------
        window_size : int
            Number of recent rounds (K) to consider.
        V_max : float
            Maximum expected variance for normal clients.
        eps : float
            Small epsilon to prevent divide-by-zero.
        """
        self.window_size = int(window_size)
        self.V_max = float(V_max)
        self.eps = float(eps)
        # rolling history: client_id -> deque of recent norms
        self.history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.window_size))

    # -----------------------
    # Update / compute logic
    # -----------------------
    def update_history(self, client_id: str, norm_value: float):
        """
        Append a new norm value to the client's history buffer.
        """
        self.history[client_id].append(float(norm_value))

    def compute(
        self,
        client_id: str,
        new_norm: Optional[float] = None,
        precomputed_var: Optional[float] = None,
    ) -> float:
        """
        Compute s_temp for one client.

        Inputs:
            - client_id: str, unique ID.
            - new_norm: optional, new norm value to append before computation.
            - precomputed_var: if you already computed variance externally.

        Returns:
            s_temp in [0,1]
        """
        if new_norm is not None:
            self.update_history(client_id, new_norm)

        hist = list(self.history.get(client_id, []))
        if len(hist) < 2:
            # not enough data to estimate variance
            return 0.0

        if precomputed_var is not None:
            var = float(precomputed_var)
        else:
            var = float(np.var(np.asarray(hist, dtype=np.float64)))

        s_temp = var / (self.V_max + self.eps)
        s_temp = float(np.clip(s_temp, 0.0, 1.0))
        return s_temp

    def compute_batch(
        self,
        client_norms: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Update histories for multiple clients and compute all s_temp values.

        Inputs:
            - client_norms: dict mapping {client_id: latest_update or norm tensor}

        Returns dict:
            {
                "s_temp": {client_id: s_temp_value},
                "variances": {client_id: variance_value}
            }
        """
        s_temp_dict = {}
        var_dict = {}

        for cid, update in client_norms.items():
            # --- Ensure scalar norm ---
            if isinstance(update, torch.Tensor):
                norm_val = torch.norm(update).item()
            elif isinstance(update, (list, np.ndarray)):
                norm_val = float(np.linalg.norm(update))
            elif isinstance(update, (float, int)):
                norm_val = float(update)
            else:
                raise TypeError(f"Unsupported update type for client {cid}: {type(update)}")

            # --- Update and compute variance ---
            self.update_history(cid, norm_val)
            hist = list(self.history[cid])
            if len(hist) >= 2:
                var = float(np.var(np.asarray(hist, dtype=np.float64)))
                s_temp = float(np.clip(var / (self.V_max + self.eps), 0.0, 1.0))
            else:
                var = 0.0
                s_temp = 0.0

            s_temp_dict[cid] = s_temp
            var_dict[cid] = var

        return {"s_temp": s_temp_dict, "variances": var_dict}

# For testing
if __name__ == "__main__":
    tc = TemporalCheck(window_size=5, V_max=0.02)

    # Simulate 3 clients
    # - client A: stable norms
    # - client B: gradually increasing
    # - client C: erratic / spiky
    rounds = 10
    rng = np.random.RandomState(0)
    for t in range(rounds):
        norms = {
            "A": 1.0 + rng.normal(scale=0.005),
            "B": 1.0 + 0.02 * t + rng.normal(scale=0.005),
            "C": 1.0 + (0.5 if t % 3 == 0 else 0.0) + rng.normal(scale=0.005),
        }
        result = tc.compute_batch(norms)
        print(f"Round {t:02d}")
        for cid, s in result["s_temp"].items():
            print(f"  {cid}: s_temp={s:.3f} (var={result['variances'][cid]:.5f})")
        print("-" * 40)
