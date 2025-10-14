from typing import Dict, Any, Optional
import numpy as np

class RuleBasedScoring:
    """
    Simple rule-based triage scoring model.
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None, T_flag: float = 0.1):
        """
        Initialize the RuleBasedScoring model.

        Parameters
        ----------
        weights : Optional[Dict[str, float]], default=None
            Weight dictionary for each pre-check component.
            Example:
                {"norm": 0.25, "cos": 0.25, "sig": 0.2, "chal": 0.2, "temp": 0.1}
            Automatically normalized to sum to 1.
        T_flag : float, default=0.1
            Threshold for flagging clients as suspicious.
            Typical range: [0.05, 0.2].
        """
        default_w = {
            "norm": 0.25,
            "cos": 0.25,
            "sig": 0.2,
            "chal": 0.2,
            "temp": 0.1
        }
        self.weights = weights or default_w
        total = sum(self.weights.values())
        if total <= 0:
            raise ValueError("Weights must sum to a positive value.")
        self.weights = {k: v / total for k, v in self.weights.items()}
        self.T_flag = float(T_flag)

    # Core scoring logic
    def compute(
        self,
        s_features: Dict[str, float]
    ) -> float:
        """
        Compute suspicion score for one client.

        Input:
            s_features: dict of pre-check scores
                        keys in {"norm","cos","sig","chal","temp"}
                        values in [0,1]

        Returns:
            anomaly_score in [0,1]
        """
        total = 0.0
        for k, w in self.weights.items():
            total += w * float(s_features.get(k, 0.0))
        return float(np.clip(total, 0.0, 1.0))

    def compute_batch(
        self,
        s_all: Dict[str, Dict[str, float]]
    ) -> Dict[str, Any]:
        """
        Compute rule-based suspicion scores for multiple clients.

        Inputs:
            s_all: dict mapping {client_id: {metric_name: score_value}}

        Returns dict:
            {
                "scores": {client_id: anomaly_score},
                "flags":  {client_id: bool}
            }
        """
        scores = {}
        flags = {}
        for cid, feats in s_all.items():
            score = self.compute(feats)
            scores[cid] = score
            flags[cid] = (score > self.T_flag)
        return {"scores": scores, "flags": flags}
    
    def report(self, results: Dict[str, Any]):
        """
        Print a simple summary of anomaly scores and flags.
        """
        scores = results.get("anomaly_scores", {})
        flags = results.get("flags", {})
        print("==== Lightweight Pre-check Summary ====")
        for cid, score in scores.items():
            mark = "[RULEBASED-FLAG-BAD]" if flags.get(cid, False) else "[RULEBASED-FLAG-GOOD]"
            print(f"Client {cid:>8s} | score={score:.3f} | {mark}")
        print(f"Flag threshold: {self.T_flag:.2f}")
        print("=======================================")


# For testing
if __name__ == "__main__":
    print("Running self-test for RuleBasedScoring...")

    # Simulated pre-check feature vectors for 5 clients
    clients = {
        "c1": {"norm": 0.05, "cos": 0.03, "sig": 0.04, "chal": 0.02, "temp": 0.01},
        "c2": {"norm": 0.1,  "cos": 0.1,  "sig": 0.08, "chal": 0.07, "temp": 0.03},
        "c3": {"norm": 0.25, "cos": 0.15, "sig": 0.12, "chal": 0.1,  "temp": 0.04},
        "c4": {"norm": 0.5,  "cos": 0.45, "sig": 0.3,  "chal": 0.25, "temp": 0.15},
        "c5": {"norm": 0.9,  "cos": 0.85, "sig": 0.6,  "chal": 0.55, "temp": 0.3},
    }

    model = RuleBasedScoring(T_flag=0.15)
    results = model.compute_batch(clients)
    model.report(results)
