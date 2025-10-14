from typing import Dict, Any
from collections import defaultdict

class SuspicionSmoothing:
    """
    Multi-round smoothing of per-client suspicion scores.
    """

    def __init__(self, beta: float = 0.9, r_min: int = 2):
        """
        Initializes the SuspicionSmoothing instance.

        Parameters
        ----------
        beta : float, optional
            EMA coefficient in [0,1].
            Higher beta means longer memory, smoother but slower reaction.
            Default is 0.9.
        r_min : int, optional
            Minimum consecutive round a client must exceed the threshold
            before being flagged as persistently suspicious.
            Default is 2.
        """
        self.beta = float(beta)
        self.r_min = int(r_min)

        # maintain per-client state
        self.smoothed_scores: Dict[str, float] = defaultdict(float)
        self.flag_streaks: Dict[str, int] = defaultdict(int)

    # Update and compute methods
    def update(
        self,
        scores: Dict[str, float],
        T_flag: float
    ) -> Dict[str, Any]:
        """
        Update smoothed suspicion scores for all clients and compute flags.

        Inputs:
            - scores: dict {client_id: current suspicion_score}
            - T_flag: threshold for flagging

        Returns dict:
            {
                "smoothed_scores": {client_id: value},
                "flags": {client_id: bool},
                "streaks": {client_id: int}
            }
        """
        smoothed = {}
        flags = {}
        streaks = {}

        for cid, s in scores.items():
            prev = self.smoothed_scores.get(cid, 0.0)
            new = self.beta * prev + (1 - self.beta) * float(s)
            self.smoothed_scores[cid] = new
            smoothed[cid] = new

            # check threshold crossing
            if new > T_flag:
                self.flag_streaks[cid] = self.flag_streaks.get(cid, 0) + 1
            else:
                self.flag_streaks[cid] = 0

            streaks[cid] = self.flag_streaks[cid]
            flags[cid] = (self.flag_streaks[cid] >= self.r_min)

        return {
            "smoothed_scores": smoothed,
            "flags": flags,
            "streaks": streaks
        }

    def reset_client(self, client_id: str):
        """
        Clear stored history for a specific client.
        """
        if client_id in self.smoothed_scores:
            del self.smoothed_scores[client_id]
        if client_id in self.flag_streaks:
            del self.flag_streaks[client_id]

    def report(self, results: Dict[str, Any]):
        """
        Print a simple summary of anomaly scores and flags.
        """
        scores = results.get("anomaly_scores", {})
        flags = results.get("flags", {})
        print("==== Lightweight Pre-check Summary ====")
        for cid, score in scores.items():
            mark = "[SMOOTHED-FLAG-BAD]" if flags.get(cid, False) else "[SMOOTHED-FLAG-GOOD]"
            print(f"Client {cid:>8s} | score={score:.3f} | {mark}")
        print(f"Flag threshold: {self.T_flag:.2f}")
        print("=======================================")


# For testing
if __name__ == "__main__":
    print("Running self-test for SuspicionSmoothing...")

    smoother = SuspicionSmoothing(beta=0.8, r_min=2)

    # simulate suspicion scores over 5 rounds for 3 clients
    rounds = [
        {"c1": 0.1, "c2": 0.05, "c3": 0.2},
        {"c1": 0.15, "c2": 0.2, "c3": 0.6},
        {"c1": 0.2, "c2": 0.4, "c3": 0.7},
        {"c1": 0.25, "c2": 0.3, "c3": 0.8},
        {"c1": 0.18, "c2": 0.35, "c3": 0.9},
    ]
    T = 0.3

    for t, scores in enumerate(rounds, start=1):
        print(f"\nRound {t}")
        result = smoother.update(scores, T)
        smoother.report(result, T)
