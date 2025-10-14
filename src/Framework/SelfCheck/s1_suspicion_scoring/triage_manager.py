from pathlib import Path
import sys
try:
    BASE_DIR = Path(__file__).resolve().parents[4]
except NameError:
    BASE_DIR = Path.cwd().parents[3]
sys.path.append(str(BASE_DIR))
import json, os
import numpy as np

from typing import Dict, Any, Optional

from .rule_based import RuleBasedScoring
from .adaptive_threshold import AdaptiveThreshold
from .smoothing import SuspicionSmoothing
from .logistic_model import LogisticScoring
from .temporal_memory import TemporalMemory

BASE_DIR = Path(__file__).resolve().parents[4]
EMA_DIR = BASE_DIR / "ema_memory"
EMA_DIR.mkdir(exist_ok=True)  # create folder if not exists

class TriageManager:
    """
    Suspicion scoring orchestrator for the FL self-check pipeline.
    """

    def __init__(
        self,
        scorer: Optional[Any] = None,
        adaptive: Optional[Any] = None,
        smoother: Optional[Any] = None,
        base_threshold: float = 0.1,
        use_adaptive: bool = True,
        ema_dir: Optional[str] = None,
        alpha_ema: float = 0.2
    ):
        """
        Initializes a TriageManager instance.

        Parameters
        ----------
        scorer : object, optional
            A scoring component instance (e.g., RuleBasedScoring or LogisticScoring).
            If None, defaults to LogisticScoring().
        adaptive : object, optional
            An AdaptiveThreshold instance. Defaults to AdaptiveThreshold().
        smoother : object, optional
            A SuspicionSmoothing instance. Defaults to SuspicionSmoothing().
        base_threshold : float, default=0.1
            The baseline anomaly threshold before adaptation.
        use_adaptive : bool, default=True
            Whether to enable adaptive thresholding.
        ema_dir : str, optional
            The directory path to store the EMA memory file. Defaults to "ema_memory" under the package root.
        alpha_ema : float, default=0.2
            The EMA decay factor.
        Notes
        -----
        The EMA memory file is stored as a JSON file in the specified directory. If the file does not exist, it will be created. If the file exists, the TriageManager will try to load the existing EMA state.
        """
        # Component injection with fallback defaults
        self.scorer = scorer if scorer is not None else LogisticScoring()
        self.adaptive = adaptive if adaptive is not None else AdaptiveThreshold()
        self.smoother = smoother if smoother is not None else SuspicionSmoothing()

        self.use_adaptive = use_adaptive
        self.base_threshold = float(base_threshold)

        # ---- EMA memory setup ----
        self.alpha_ema = alpha_ema
        self.ema_file = Path(ema_dir) if ema_dir else EMA_DIR / "client_ema_latest.json"
        self.client_ema = {}
        self.temporal_memory = TemporalMemory(EMA_DIR / "multi_metric_ema.json", alpha=self.alpha_ema)

        # Try loading existing EMA file ONCE
        if self.ema_file.exists():
            try:
                with open(self.ema_file, "r") as f:
                    self.client_ema = json.load(f)
                print(f"[TriageManager] Loaded previous EMA state from {self.ema_file}")
            except Exception as e:
                print(f"[TriageManager] Warning: failed to load EMA memory: {e}")

        print(
            f"[TriageManager] Initialized | "
            f"Scorer={self.scorer.__class__.__name__}, "
            f"Adaptive={'ON' if self.use_adaptive else 'OFF'}, "
            f"BaseThreshold={self.base_threshold:.3f}"
        )

    def step(
        self,
        client_features: Dict[str, Dict[str, float]],
        round_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Execute full suspicion scoring pipeline for one FL round.

        Parameters
        ----------
        client_features : Dict[str, Dict[str, float]]
            {
                client_id: {
                    "s_norm": float,
                    "s_cos": float,
                    "s_sig": float,
                    "s_chal": float,
                    "s_temp": float
                }
            }
        round_id : int, optional
            Current FL round number (for logging).

        Returns
        -------
        Dict[str, Any]
            {
                "scores": {client_id: float},
                "threshold": float,
                "smoothed": {client_id: float},
                "flags": {client_id: bool},
                "streaks": {client_id: int}
            }
        """

        # Update multi-metric EMA per client
        for cid, feats in client_features.items():
            metrics = {
                "s_cos": 1.0 - feats.get("s_cos", 0.0),  # drift magnitude
                "s_norm": feats.get("s_norm", 0.0),
                "s_sig": feats.get("s_sig", 0.0),
                "s_chal": feats.get("s_chal", 0.0),
                "s_temp": feats.get("s_temp", 0.0),
            }

            # Update persistent EMA
            ema_updated = self.temporal_memory.update(cid, metrics)
            feats.update({f"ema_{k}": v for k, v in ema_updated.items()})

            if round_id is not None:
                print(f"[TemporalMemory] Round {round_id:03d} | {cid:<10s} | {ema_updated}")

        # Save updated EMA state
        self.temporal_memory.save()

        # Compute suspicion scores
        score_result = self.scorer.compute_batch(client_features)
        scores: Dict[str, float] = score_result["scores"]

        # Blend EMA drift info into the main score
        for cid in scores:
            ema_cos = client_features[cid].get("ema_s_cos", 0.0)
            ema_norm = client_features[cid].get("ema_s_norm", 0.0)
            ema_sig = client_features[cid].get("ema_s_sig", 0.0)
            # Custom weight: small impact but persistent drift memory
            scores[cid] += 0.05 * (ema_cos + ema_norm + ema_sig)

        # Compute adaptive threshold + smoothing
        if self.use_adaptive:
            thresh_result = self.adaptive.compute_batch(scores)
            T_flag = thresh_result["adaptive_threshold"]
        else:
            T_flag = self.base_threshold

        smooth_result = self.smoother.update(scores, T_flag)

        # Determine anomaly flags (adaptive + smoothed)
        adaptive_flags = {cid: (s > T_flag) for cid, s in scores.items()}
        final_flags = {
            cid: (adaptive_flags[cid] or smooth_result["flags"].get(cid, False))
            for cid in scores
        }

        combined = {
            "scores": scores,
            "threshold": T_flag,
            "smoothed": smooth_result["smoothed_scores"],
            "flags": final_flags,
            "streaks": smooth_result["streaks"],
        }

        if round_id is not None:
            print(f"[TriageManager] Round {round_id} processed — Adaptive T={T_flag:.3f}")

        return combined

if __name__ == "__main__":
    # Example input from lightweight pre-check
    sample_features = {
        "client_1": {"s_norm": 0.1, "s_cos": 0.2, "s_sig": 0.05, "s_chal": 0.1, "s_temp": 0.1},
        "client_2": {"s_norm": 0.3, "s_cos": 0.4, "s_sig": 0.35, "s_chal": 0.25, "s_temp": 0.2},
        "client_3": {"s_norm": 0.5, "s_cos": 0.7, "s_sig": 0.6, "s_chal": 0.55, "s_temp": 0.5},
    }

    triage = TriageManager(base_threshold=0.25)
    for r in range(1, 5):
        res = triage.step(sample_features, round_id=r)

