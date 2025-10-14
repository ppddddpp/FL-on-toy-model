import json
from pathlib import Path
from typing import Dict, Any

class TemporalMemory:
    """
    Persistent multi-metric EMA memory across rounds for each client.
    Tracks cumulative drift (temporal consistency) over rounds.
    """

    def __init__(self, memory_file: Path, alpha: float = 0.2):
        self.memory_file = memory_file
        self.alpha = alpha
        self.state: Dict[str, Dict[str, float]] = {}  # {client_id: {metric: ema_val}}
        if self.memory_file.exists():
            try:
                with open(self.memory_file, "r") as f:
                    self.state = json.load(f)
                print(f"[TemporalMemory] Loaded EMA state from {self.memory_file}")
            except Exception as e:
                print(f"[TemporalMemory] Warning: failed to load: {e}")

    def update(self, client_id: str, metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Update EMA for each metric for a given client.
        Returns drift magnitudes (|ema - new_value|).
        """
        prev = self.state.get(client_id, {})
        new_ema = {}
        drifts = {}

        for key, val in metrics.items():
            old = prev.get(key, val)
            ema_val = (1 - self.alpha) * old + self.alpha * val
            new_ema[key] = float(ema_val)
            drifts[key] = abs(ema_val - val)

        self.state[client_id] = new_ema
        return drifts

    def save(self):
        try:
            with open(self.memory_file, "w") as f:
                json.dump(self.state, f, indent=2)
        except Exception as e:
            print(f"[TemporalMemory] Warning: save failed: {e}")
