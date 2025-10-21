from typing import Dict, Sequence, Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from Helpers.Helpers import log_and_print

BASE_DIR = Path(__file__).resolve().parents[4]

class LogisticScoring(nn.Module):
    """
    Logistic regression suspicion scorer.
    """
    def __init__(
        self,
        feature_keys: Optional[Sequence[str]] = None,
        lr: float = 1e-2,
        T_flag: float = 0.1,
        device: str = "cpu",
        pretrained_path: Optional[str] = None,
        auto_load: bool = True,
        log_dir: Path = BASE_DIR / "logs" / "run.txt"
    ):
        """
        Initialize the LogisticScoring model.

        Parameters
        ----------
        feature_keys : Optional[Sequence[str]]
            Names of expected pre-check metrics.
            Default: ["norm", "cos", "sig", "chal", "temp"]
        lr : float
            Learning rate for online fine-tuning.
        T_flag : float
            Default suspicion threshold.
        device : str
            'cpu' or 'cuda' (default: 'cpu')
        pretrained_path : Optional[str]
            Path to a saved model to load from.
            auto_load : bool
            If True, try to load a saved model from the given path.
        """
        super().__init__()
        self.feature_keys = list(feature_keys or ["norm", "cos", "sig", "chal", "temp"])
        self.dim = len(self.feature_keys)
        self.linear = nn.Linear(self.dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.T_flag = float(T_flag)
        self.device = torch.device(device)
        self.to(self.device)
        self.optimizer = optim.SGD(self.parameters(), lr=lr)
        self.log_dir = log_dir

        # Try auto-loading pretrained weights
        if auto_load:
            # default: look in project checkpoints folder
            default_path = Path(pretrained_path) if pretrained_path else \
                    BASE_DIR / "checkpoints" / "logistic_scoring" / "logistic_trained.pt"
            if default_path.exists():
                try:
                    self.load_state_dict(torch.load(default_path, map_location=self.device))
                    log_and_print(f"[LogisticScoring] Loaded pretrained weights from: {default_path}", log_file=self.log_dir)
                except Exception as e:
                    log_and_print(f"[LogisticScoring] Warning: failed to load pretrained model at {default_path}: {e}", log_file=self.log_dir)
            else:
                log_and_print(f"[LogisticScoring] No pretrained model found at {default_path} — using random init.", log_file=self.log_dir)

    # Forward / compute logic
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass — compute suspicion probabilities.
        """
        return self.sigmoid(self.linear(x))

    def compute(
        self,
        s_features: Dict[str, float]
    ) -> float:
        """
        Compute suspicion probability for one client.

        Input:
            s_features: dict mapping {metric_name: score}
        Returns:
            float suspicion score ∈ [0,1]
        """
        x = np.array([float(s_features.get(k, 0.0)) for k in self.feature_keys], dtype=np.float32)
        x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            score = self.forward(x_tensor).item()
        return float(score)

    def compute_batch(
        self,
        s_all: Dict[str, Dict[str, float]]
    ) -> Dict[str, Any]:
        """
        Compute suspicion probabilities for multiple clients.

        Inputs:
            s_all: dict mapping {client_id: {metric_name: score}}

        Returns:
            {
                "scores": {client_id: prob},
                "flags": {client_id: bool}
            }
        """
        scores = {}
        flags = {}
        for cid, feats in s_all.items():
            s = self.compute(feats)
            scores[cid] = s
            flags[cid] = (s > self.T_flag)
        return {"scores": scores, "flags": flags}

    # Optional training logic
    def train_step(
        self,
        batch_features: np.ndarray,
        batch_labels: np.ndarray
    ) -> float:
        """
        Perform one SGD training step.

        Inputs:
            batch_features: shape (N, D)
            batch_labels: shape (N,), values ∈ {0,1}
        Returns:
            loss value
        """
        self.train()
        x = torch.tensor(batch_features, dtype=torch.float32, device=self.device)
        y = torch.tensor(batch_labels, dtype=torch.float32, device=self.device).unsqueeze(1)

        pred = self.forward(x)
        loss = nn.BCELoss()(pred, y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return float(loss.item())

    def report(self, results: Dict[str, Any]):
        """
        Print a simple summary of anomaly scores and flags.
        """
        scores = results.get("anomaly_scores", {})
        flags = results.get("flags", {})
        print("==== Lightweight Pre-check Summary ====")
        for cid, score in scores.items():
            mark = "[LOGISTIC-FLAG-BAD]" if flags.get(cid, False) else "[LOGISTIC-FLAG-GOOD]"
            print(f"Client {cid:>8s} | score={score:.3f} | {mark}")
        print(f"Flag threshold: {self.T_flag:.2f}")
        print("=======================================")


# For testing
if __name__ == "__main__":
    print("Running self-test for LogisticScoring...")

    model = LogisticScoring(T_flag=0.15, lr=0.1)

    # Simulated pre-check scores for 5 clients
    clients = {
        "c1": {"norm": 0.05, "cos": 0.03, "sig": 0.04, "chal": 0.02, "temp": 0.01},
        "c2": {"norm": 0.1,  "cos": 0.1,  "sig": 0.08, "chal": 0.07, "temp": 0.03},
        "c3": {"norm": 0.25, "cos": 0.15, "sig": 0.12, "chal": 0.1,  "temp": 0.04},
        "c4": {"norm": 0.5,  "cos": 0.45, "sig": 0.3,  "chal": 0.25, "temp": 0.15},
        "c5": {"norm": 0.9,  "cos": 0.85, "sig": 0.6,  "chal": 0.55, "temp": 0.3},
    }

    # Compute initial suspicion probabilities
    results = model.compute_batch(clients)
    model.report(results)
