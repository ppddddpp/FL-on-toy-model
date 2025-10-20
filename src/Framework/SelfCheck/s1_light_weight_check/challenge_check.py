from typing import Optional, Sequence, Dict, Any
import numpy as np

class ChallengeCheck:
    """
    Challenge-response deviation detector.
    """

    def __init__(self, A_max: float = 0.2, use_median_ref: bool = True, eps: float = 1e-8):
        """
        Parameters
        ----------
        A_max : float
            Maximum tolerated accuracy degradation.
            (e.g., 0.1 means 10% accuracy drop yields s_chal ≈ 1)
        use_median_ref : bool
            If True, use median(a_i) of all clients as reference instead of fixed baseline a_0.
        eps : float
            Small epsilon for numerical stability.
        """
        self.A_max = float(A_max)
        self.use_median_ref = bool(use_median_ref)
        self.eps = float(eps)

    def compute(
        self,
        a_i: float,
        all_accs: Optional[Sequence[float]] = None,
        baseline_acc: Optional[float] = None,
        precomputed_ref: Optional[float] = None,
    ) -> float:
        """
        Compute s_chal for a single client.

        Inputs:
            - a_i: client's anchor accuracy.
            - all_accs: optional list of all clients' accuracies (if using median reference).
            - baseline_acc: accuracy of global model before this round (if using fixed baseline).
            - precomputed_ref: precomputed reference accuracy value (median or baseline).

        Returns:
            s_chal in [0,1]
        """
        if a_i is None:
            return 0.0

        # Determine reference
        if precomputed_ref is not None:
            ref = float(precomputed_ref)
        elif self.use_median_ref and all_accs is not None and len(all_accs) > 0:
            ref = float(np.median(np.asarray(all_accs, dtype=float)))
        elif (not self.use_median_ref) and baseline_acc is not None:
            ref = float(baseline_acc)
        else:
            # insufficient context
            return 0.0

        delta_a = max(0.0, ref - float(a_i))
        s_chal = delta_a / (self.A_max + self.eps)
        s_chal = float(np.clip(s_chal, 0.0, 1.0))
        return s_chal

    def compute_batch(
        self,
        accs: Sequence[float],
        baseline_acc: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Compute reference accuracy and s_chal for a batch of clients.

        Returns dict:
            {
                "reference": reference_accuracy,
                "s_chal": np.array([...])
            }
        """
        if accs is None or len(accs) == 0:
            return {"reference": None, "s_chal": np.array([], dtype=float)}

        arr = np.asarray(accs, dtype=float)
        if self.use_median_ref:
            ref = float(np.median(arr))
            if baseline_acc is not None:
                ref = 0.5 * (ref + float(baseline_acc))  # smooth reference
        elif baseline_acc is not None:
            ref = float(baseline_acc)
        else:
            ref = float(np.median(arr))

        if np.all(arr < 0.2):  # e.g. all bad
            ref = max(ref, getattr(self, "baseline_acc", 0.5))

        deltas = np.maximum(0.0, np.abs(arr - ref))
        s = deltas / (self.A_max + self.eps)
        s = np.clip(s, 0.0, 1.0)

        return {"reference": ref, "s_chal": s}


# For testing
if __name__ == "__main__":
    print("Running self-test for ChallengeCheck...")
    rng = np.random.RandomState(0)

    # Simulate 10 normal clients with similar accuracy, 2 malicious drops
    accs = list(0.8 + rng.normal(scale=0.01, size=10))
    accs += [0.6, 0.5]  # two bad ones

    cc = ChallengeCheck(A_max=0.2, use_median_ref=True)
    batch = cc.compute_batch(accs)
    print(f"reference median acc = {batch['reference']:.3f}")
    for idx, s in enumerate(batch["s_chal"]):
        print(f"client {idx:02d}: acc={accs[idx]:.3f}, s_chal={s:.4f}")

    # Example with fixed baseline
    cc_fixed = ChallengeCheck(A_max=0.2, use_median_ref=False)
    baseline = 0.8
    batch_fixed = cc_fixed.compute_batch(accs, baseline_acc=baseline)
    print(f"\nFixed baseline={baseline}")
    for idx, s in enumerate(batch_fixed["s_chal"]):
        print(f"client {idx:02d}: acc={accs[idx]:.3f}, s_chal={s:.4f}")
