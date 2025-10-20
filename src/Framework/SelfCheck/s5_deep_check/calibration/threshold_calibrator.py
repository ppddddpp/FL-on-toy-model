import numpy as np
import torch
from tqdm import tqdm
from typing import Any

class ThresholdCalibrator:
    """
    Automatically calibrates thresholds for DeepCheckManager detectors
    using benign and synthetic adversarial updates.
    """

    def __init__(self, deepcheck, target_fpr=0.05):
        """
        deepcheck: DeepCheckManager instance
        target_fpr: target false-positive rate for benign clients
        """
        from ..deep_check_eval import DeepCheckManager
        self.deepcheck = deepcheck or DeepCheckManager()
        self.target_fpr = target_fpr

    def simulate_adversarial(self, client_delta):
        """
        Create a synthetic 'adversarial' version of a client delta by injecting noise.
        """
        adv = {}
        for k, v in client_delta.items():
            noise = torch.randn_like(v) * 0.5  # heavy perturbation
            adv[k] = v + noise
        return adv

    def calibrate(self, global_model, benign_deltas, anchor_loader=None):
        """
        Run DeepCheck on benign and adversarial updates, estimate thresholds.
        Returns: dict of recommended threshold values.
        """
        benign_scores = []
        malicious_scores = []

        print("[Calibrator] Running DeepCheck on benign deltas...")
        for cid, delta in tqdm(benign_deltas.items()):
            res = self.deepcheck.compute(global_model=global_model,
                                            client_delta=delta,
                                            client_id=f"{cid}_benign",
                                            anchor_loader=anchor_loader)
            benign_scores.append(res["S_final"])

        print("[Calibrator] Running DeepCheck on simulated adversarial deltas...")
        for cid, delta in tqdm(benign_deltas.items()):
            adv_delta = self.simulate_adversarial(delta)
            res = self.deepcheck.compute(global_model=global_model,
                                            client_delta=adv_delta,
                                            client_id=f"{cid}_adv",
                                            anchor_loader=anchor_loader)
            malicious_scores.append(res["S_final"])

        benign_scores = np.array(benign_scores)
        malicious_scores = np.array(malicious_scores)

        # compute recommended threshold for activation rejection or sandbox
        thresh = np.percentile(benign_scores, 100 * self.target_fpr)
        print(f"[Calibrator] Recommended S_final rejection threshold = {thresh:.3f}")

        return {
            "activation_reject_threshold": thresh,
            "anchor_drop_tolerance": float(np.clip(np.mean(1 - benign_scores), 0.01, 0.2))
        }


class SafeThresholdCalibrator(ThresholdCalibrator):
    """
    Safer extension of ThresholdCalibrator with defenses
    against calibration poisoning or manipulation.
    """

    def __init__(self, deepcheck: Any, target_fpr=0.05,
                    min_samples=20, max_delta=0.05, ema_alpha=0.1):
        from ..deep_check_eval import DeepCheckManager
        deepcheck = deepcheck or DeepCheckManager()
        super().__init__(deepcheck, target_fpr)
        self.min_samples = min_samples
        self.max_delta = max_delta
        self.ema_alpha = ema_alpha  # smoothing for updates

    def robust_threshold(self, scores):
        """Compute robust threshold with trimming to avoid outlier influence."""
        if len(scores) < 5:
            return float(np.percentile(scores, 100 * self.target_fpr))
        trim = 0.1
        scores = np.sort(scores)
        lo = int(len(scores) * trim)
        hi = int(len(scores) * (1 - trim))
        trimmed = scores[lo:hi]
        return float(np.percentile(trimmed, 100 * self.target_fpr))

    def calibrate(self, global_model, benign_deltas, anchor_loader=None):
        """
        Calibrate thresholds safely using robust estimators and change limits.
        """
        if len(benign_deltas) < self.min_samples:
            print("[SafeCalibrator] Skipped calibration — insufficient trusted samples.")
            return {}

        benign_scores = []
        malicious_scores = []

        print("[SafeCalibrator] Evaluating benign updates...")
        for cid, delta in tqdm(benign_deltas.items()):
            res = self.deepcheck.compute(global_model=global_model,
                                            client_delta=delta,
                                            client_id=f"{cid}_benign",
                                            anchor_loader=anchor_loader)
            benign_scores.append(res.get("S_final", 0.0))

        print("[SafeCalibrator] Simulating adversarial updates...")
        for cid, delta in tqdm(benign_deltas.items()):
            adv_delta = self.simulate_adversarial(delta)
            res = self.deepcheck.compute(global_model=global_model,
                                            client_delta=adv_delta,
                                            client_id=f"{cid}_adv",
                                            anchor_loader=anchor_loader)
            malicious_scores.append(res.get("S_final", 0.0))

        benign_scores = np.array(benign_scores)
        malicious_scores = np.array(malicious_scores)

        # Check separation between benign and adversarial distributions
        sep = np.mean(benign_scores) - np.mean(malicious_scores)
        if sep < 0.15:
            print("[SafeCalibrator] Warning: weak separation between benign/adversarial data — skip calibration.")
            return {}

        # Compute new threshold robustly
        candidate_thresh = self.robust_threshold(benign_scores)

        # Clamp the change from current threshold
        old_thresh = self.deepcheck.activation_reject_threshold
        delta = np.clip(candidate_thresh - old_thresh, -self.max_delta, self.max_delta)
        new_thresh = old_thresh + delta

        # Apply EMA smoothing
        smoothed = (1 - self.ema_alpha) * old_thresh + self.ema_alpha * new_thresh
        self.deepcheck.activation_reject_threshold = smoothed

        print(f"[SafeCalibrator] Activation threshold {old_thresh:.3f} → {smoothed:.3f} (sep={sep:.3f})")

        # Conservative anchor tolerance update (slowly adapting)
        old_anchor_tol = self.deepcheck.anchor_drop_tolerance
        new_anchor_tol = float(np.clip(np.mean(1 - benign_scores), 0.01, 0.2))
        smoothed_anchor = (1 - self.ema_alpha) * old_anchor_tol + self.ema_alpha * new_anchor_tol
        self.deepcheck.anchor_drop_tolerance = smoothed_anchor

        return {
            "activation_reject_threshold": smoothed,
            "anchor_drop_tolerance": smoothed_anchor,
            "sep": float(sep)
        }