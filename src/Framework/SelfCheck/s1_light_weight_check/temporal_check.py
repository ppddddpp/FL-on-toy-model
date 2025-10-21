from typing import Optional, Dict, Any
import numpy as np
from collections import defaultdict, deque
import torch
from Helpers.Helpers import log_and_print
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[4]

class TemporalCheck:
    """
    Temporal instability detector (adaptive + cross-round difference version).
    """

    def __init__(self, window_size: int = 5, V_max: float = 0.1, 
                    eps: float = 1e-12, adaptive: bool = False,
                    log_dir: Path = BASE_DIR / "logs" / "run.txt"):
        """
        Parameters
        ----------
        window_size : int
            Number of recent rounds (K) to consider.
        V_max : float
            Maximum expected variance for normal clients.
        eps : float
            Small epsilon to prevent divide-by-zero.
        adaptive : bool
            If True, adapt V_max each round using median variance across clients.
        log_dir : Path
            Directory for logging.
        """
        self.window_size = int(window_size)
        self.V_max = float(V_max)
        self.eps = float(eps)
        self.adaptive = bool(adaptive)
        self.history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.window_size))
        self.prev_round: Dict[str, torch.Tensor] = {}
        self.initial_Vmax = V_max
        self.log_dir = log_dir

    # -----------------------
    # Update / compute logic
    # -----------------------
    def update_history(self, client_id: str, value: float):
        """Append a new scalar (norm or diff) to client's rolling history."""
        self.history[client_id].append(float(value))

    def compute(
        self,
        client_id: str,
        new_value: Optional[float] = None,
        precomputed_var: Optional[float] = None,
    ) -> float:
        """Compute s_temp for one client."""
        if new_value is not None:
            self.update_history(client_id, new_value)

        hist = list(self.history.get(client_id, []))
        if len(hist) < 2:
            return 0.0

        var = float(precomputed_var) if precomputed_var is not None else float(np.var(hist))
        s_temp = var / (self.V_max + self.eps)
        return float(np.clip(s_temp, 0.0, 1.0))

    def compute_batch(self, client_updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update histories for multiple clients and compute temporal scores.
        Measures norm differences vs. previous round if available.
        """
        s_temp_dict, var_dict = {}, {}

        for cid, update in client_updates.items():
            # compute scalar measure (either norm or difference vs. prev round)
            if isinstance(update, torch.Tensor):
                val = torch.norm(update - self.prev_round[cid]).item() if cid in self.prev_round else torch.norm(update).item()
            elif isinstance(update, (list, np.ndarray)):
                arr = np.array(update, dtype=float)
                val = np.linalg.norm(arr - self.prev_round[cid].cpu().numpy()) if cid in self.prev_round else np.linalg.norm(arr)
            elif isinstance(update, (float, int)):
                val = float(update)
            else:
                raise TypeError(f"Unsupported update type for client {cid}: {type(update)}")

            self.update_history(cid, val)
            hist = list(self.history[cid])

            if len(hist) >= 2:
                var = float(np.var(hist))
                s_temp = float(np.clip(var / (self.V_max + self.eps), 0.0, 1.0))
            else:
                var = 0.0
                s_temp = 0.0

            s_temp_dict[cid] = s_temp
            var_dict[cid] = var

        # --- Adaptive threshold update (optional)
        if self.adaptive and var_dict:
            median_var = np.median(list(var_dict.values()))
            alpha = 0.2
            new_vmax = (1 - alpha) * self.V_max + alpha * median_var
            new_vmax = max(new_vmax, self.eps, 0.5 * self.initial_Vmax)
            if abs(new_vmax - self.V_max) / (self.V_max + 1e-12) > 0.2:
                log_and_print(f"[TemporalCheck] Adjusting V_max: {self.V_max:.5f} -> {new_vmax:.5f}", log_dir=self.log_dir)
            self.V_max = new_vmax

        # --- Save current updates for next round comparison
        self.prev_round = {cid: update.clone().detach() if isinstance(update, torch.Tensor) else update for cid, update in client_updates.items()}

        return {"s_temp": s_temp_dict, "variances": var_dict}


# --- Testing
if __name__ == "__main__":
    tc = TemporalCheck(window_size=5, V_max=0.02, adaptive=True)
    rng = np.random.RandomState(0)

    for t in range(10):
        updates = {
            "A": torch.tensor(rng.normal(1.0, 0.01, 128)).float(),
            "B": torch.tensor(rng.normal(1.0 + 0.02 * t, 0.01, 128)).float(),
            "C": torch.tensor(rng.normal(1.0 + (0.5 if t % 3 == 0 else 0.0), 0.01, 128)).float(),
        }
        result = tc.compute_batch(updates)
        print(f"Round {t:02d}")
        for cid, s in result["s_temp"].items():
            print(f"  {cid}: s_temp={s:.3f} (var={result['variances'][cid]:.5f})")
        print("-" * 40)
