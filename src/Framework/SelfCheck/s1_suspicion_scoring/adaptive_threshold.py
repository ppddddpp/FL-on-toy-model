from typing import Sequence, Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
from Helpers.Helpers import log_and_print
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[4]

# --- Residual neural corrector ------------------------------------------------
class ThresholdMLP(nn.Module):
    """Tiny MLP that predicts a residual correction for threshold."""
    def __init__(self, input_dim=4, hidden_dim=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# --- Adaptive + semi-auto class ----------------------------------------------
class AdaptiveThreshold:
    """
    Adaptive + learnable threshold calculator for suspicion scores.
    """
    def __init__(
        self,
        base_T: float = 0.1,
        lambda_scale: float = 1.0,
        clip_range: Optional[Sequence[float]] = (0.02, 0.5),
        ema_beta: float = 0.0,
        lr: float = 1e-3,
        device: str = "cpu",
        max_adjust: float = 0.05,
        semi_mode: bool = True,
        freeze_drift: float = 0.25,
        patience: int = 3,
        checkpoint_path: Optional[str] = None,
        log_dir: Path = BASE_DIR / "logs" / "run.txt",
    ):
        """
        Initialize the adaptive thresholding component.

        Parameters
        ----------
        base_T : float, default=0.1
            The baseline anomaly threshold before adaptation.
        lambda_scale : float, default=1.0
            Sensitivity multiplier (typical range 0.5 to 2.0).
            Lower lambda means more sensitive, more clients flagged.
            Higher lambda means more conservative, fewer flags.
        clip_range : tuple(float, float), default=(0.02, 0.5)
            Optional range to bound the threshold (min, max).
        ema_beta : float, default=0.0
            Optional smoothing coefficient for threshold evolution across rounds.
            0 disables smoothing, >0 retains previous threshold memory.
        lr : float, default=1e-3
            Learning rate for the MLP residual corrector.
        device : str, default="cpu"
            Device to run the model on.
        max_adjust : float, default=0.05
            Maximum residual correction allowed.
        semi_mode : bool, default=True
            If True, use semi-auto learning with pseudo-labels.
        freeze_drift : float, default=0.25
            Drift threshold beyond which the model is frozen.
        patience : int, default=3
            Number of consecutive rounds to wait before unfreezing.
        checkpoint_path : str, optional
            Path to load/store model checkpoints. If None, uses the default checkpoint path.
        log_dir : Path, default=BASE_DIR / "logs" / "run.txt"
            Path to the log file.
        """
        # Adaptive threshold
        self.base_T = float(base_T)
        self.lambda_scale = float(lambda_scale)
        self.clip_range = clip_range
        self.ema_beta = float(ema_beta)
        self.prev_T: Optional[float] = None
        self.device = torch.device(device)

        # Semi-auto learning
        self.max_adjust = float(max_adjust)
        self.semi_mode = semi_mode
        self.freeze_drift = freeze_drift
        self.patience = patience
        self.freeze_counter = 0
        self.is_frozen = False

        # Load checkpoint if available
        self.checkpoint_path = checkpoint_path if checkpoint_path is not None else BASE_DIR / "checkpoints" / "adaptive_threshold"
        self.checkpoint_path.mkdir(exist_ok=True, parents=True)
        self.model_path = self.checkpoint_path / "mlp_residual.pt"
        self.state_path = self.checkpoint_path / "state.npy"

        # MLP residual corrector
        self.model = ThresholdMLP(input_dim=4).to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        # Try to restore from previous checkpoint
        self._try_load()

        # Logging
        self.log_dir = log_dir

    def _rule_threshold(self, scores: Sequence[float]) -> float:
        """
        Compute the rule-based adaptive threshold.

        This function computes the rule-based adaptive threshold by
        calculating the mean and standard deviation of the input scores,
        and then applying the lambda scale to the standard deviation.

        If the input scores are None or have less than 2 elements, this
        function returns the base threshold.

        Returns
        -------
        float
            The computed rule-based adaptive threshold.
        float
            The mean of the input scores.
        float
            The standard deviation of the input scores.
        """
        if scores is None or len(scores) < 2:
            return self.base_T
        
        # rule-based threshold
        arr = np.asarray(scores, dtype=np.float32)
        mu, sigma = float(np.mean(arr)), float(np.std(arr))
        T = mu + self.lambda_scale * sigma
        if self.clip_range is not None:
            T = float(np.clip(T, self.clip_range[0], self.clip_range[1]))
        return T, mu, sigma

    def compute(self, scores: Sequence[float]) -> float:
        """
        Compute the adaptive threshold with optional neural correction.

        If the model is not frozen, it computes the neural correction
        delta_T and adds it to the rule-based threshold T_rule. The
        resulting threshold is then clipped to the specified range and
        smoothed using exponential moving average (EMA).

        If the model is frozen, it skips the neural correction and returns
        the rule-based threshold directly.

        Parameters
        ----------
        scores : Sequence[float]
            Input suspicion scores for adaptive thresholding.

        Returns
        -------
        float
            The computed adaptive threshold.
        """        
        T_rule, mu, sigma = self._rule_threshold(scores)
        if self.is_frozen:
            # if frozen, skip neural correction
            return T_rule

        T_in = torch.tensor(
            [[mu, sigma, self.lambda_scale, self.prev_T or T_rule]],
            dtype=torch.float32, device=self.device
        )
        with torch.no_grad():
            dT = self.model(T_in).item()

        # semi-auto limiter
        if self.semi_mode:
            dT = float(np.clip(dT, -self.max_adjust, self.max_adjust))
        T = T_rule + dT

        # safe clip + EMA
        if self.clip_range is not None:
            T = float(np.clip(T, self.clip_range[0], self.clip_range[1]))
        if self.ema_beta > 0 and self.prev_T is not None:
            T = self.ema_beta * self.prev_T + (1 - self.ema_beta) * T

        # drift detection (auto freeze)
        if self.prev_T is not None:
            drift = abs(T - self.prev_T)
            if drift > self.freeze_drift:
                self.freeze_counter += 1
                if self.freeze_counter >= self.patience:
                    log_and_print(f"[AutoFreeze] Drift {drift:.3f} exceeds limit. Model frozen.", log_file=self.log_dir)
                    self.is_frozen = True
            else:
                self.freeze_counter = 0
        self.prev_T = T
        return float(T)

    def train_step(self, scores: Sequence[float], target_T: float) -> float:
        """
        Train the MLP residual corrector for one step.

        If the model is frozen, it skips the training step and returns 0.0.

        Parameters
        ----------
        scores : Sequence[float]
            Input suspicion scores for adaptive thresholding.
        target_T : float
            Target adaptive threshold for training.

        Returns
        -------
        float
            The computed loss for the training step.
        """
        if self.is_frozen:
            return 0.0
        T_rule, mu, sigma = self._rule_threshold(scores)
        x = torch.tensor([[mu, sigma, self.lambda_scale, self.prev_T or T_rule]],
                            dtype=torch.float32, device=self.device)
        y = torch.tensor([[target_T - T_rule]], dtype=torch.float32, device=self.device)
        self.opt.zero_grad()
        pred = self.model(x)
        loss = self.loss_fn(pred, y)
        loss.backward()
        self.opt.step()
        return float(loss.item())
    
    def manual_train(self, data: Sequence[Dict[str, float]], epochs: int = 10):
        """
        Offline supervised training for the MLP residual corrector.
        Each item in data should be:
            {
                "mu": float,           # mean suspicion score
                "sigma": float,        # std of suspicion scores
                "lambda": float,       # lambda_scale used
                "prev_T": float,       # previous threshold
                "target_T": float      # desired next threshold (label)
            }
        """
        if not data:
            log_and_print("[ManualTrain][Error] No data provided.", log_file=self.log_dir)
            return

        self.model.train()
        X, Y = [], []
        for sample in data:
            mu = sample["mu"]
            sigma = sample["sigma"]
            lam = sample.get("lambda", self.lambda_scale)
            prev_T = sample["prev_T"]
            target_T = sample["target_T"]
            rule_T = mu + lam * sigma
            X.append([mu, sigma, lam, prev_T])
            Y.append([target_T - rule_T])
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        Y = torch.tensor(Y, dtype=torch.float32, device=self.device)

        for epoch in range(epochs):
            pred = self.model(X)
            loss = self.loss_fn(pred, Y)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            if epoch % 2 == 0 or epoch == epochs - 1:
                log_and_print(f"[ManualTrain] Epoch {epoch+1}/{epochs} | Loss={loss.item():.6f}")

        self.save()
        log_and_print(f"[ManualTrain] Finished training on {len(data)} samples.", log_file=self.log_dir)
    
    def semi_auto_update(self, scores: Sequence[float], T_applied: float):
        """
        Semi-auto learning with pseudo-label.
        The model learns gradually from observed stable rounds,
        weighted by a safety confidence term.
        """
        if self.is_frozen or not self.semi_mode:
            return
        T_rule, mu, sigma = self._rule_threshold(scores)
        pseudo_label = (T_applied - T_rule)
        # confidence = low if recent drift high
        stability = 1.0 - min(abs(pseudo_label) / (self.max_adjust * 2), 1.0)
        weight = torch.tensor(stability, dtype=torch.float32, device=self.device)
        x = torch.tensor([[mu, sigma, self.lambda_scale, self.prev_T or T_rule]],
                            dtype=torch.float32, device=self.device)
        y = torch.tensor([[pseudo_label]], dtype=torch.float32, device=self.device)
        pred = self.model(x)
        loss = self.loss_fn(pred, y) * weight
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        log_and_print(f"[SemiAuto] pseudo delta_T={pseudo_label:.4f} | weight={stability:.2f}", log_file=self.log_dir)
        return float(loss.item())

    def compute_batch(self, scores_dict: Dict[str, float]) -> Dict[str, Any]:
        """
        Compute adaptive threshold and flags for a batch of clients.

        Parameters
        ----------
        scores_dict : Dict[str, float]
            {client_id: suspicion_score}

        Returns
        -------
        Dict[str, Any]
            {
                "adaptive_threshold": float,
                "flags": {client_id: bool}
            }
        """
        if not scores_dict:
            return {"adaptive_threshold": self.base_T, "flags": {}}
        scores = list(scores_dict.values())
        T = self.compute(scores)
        flags = {cid: (s > T) for cid, s in scores_dict.items()}
        # optional semi-auto refinement
        self.semi_auto_update(scores, T)
        return {"adaptive_threshold": T, "flags": flags}

    def save(self):
        """Save model weights and state."""
        torch.save(self.model.state_dict(), self.model_path)
        state = {
            "prev_T": self.prev_T,
            "is_frozen": self.is_frozen,
            "freeze_counter": self.freeze_counter,
        }
        np.save(self.state_path, state)
        log_and_print(f"[Checkpoint] Saved model to {self.model_path}", log_file=self.log_dir)

    def _try_load(self):
        """Try loading previous checkpoint if available."""
        if self.model_path.exists():
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            log_and_print(f"[Checkpoint] Loaded MLP from {self.model_path}")
        if self.state_path.exists():
            state = np.load(self.state_path, allow_pickle=True).item()
            self.prev_T = state.get("prev_T", None)
            self.is_frozen = state.get("is_frozen", False)
            self.freeze_counter = state.get("freeze_counter", 0)
            log_and_print(f"[Checkpoint] Loaded state (prev_T={self.prev_T}, frozen={self.is_frozen})", log_file=self.log_dir)

    def report(self, results: Dict[str, Any]):
        print("==== Adaptive Thresholding (Hybrid) ====")
        T = results.get("adaptive_threshold", self.base_T)
        flags = results.get("flags", {})
        flagged = [cid for cid, f in flags.items() if f]
        print(f"Adaptive threshold: {T:.3f} | Frozen: {self.is_frozen}")
        print(f"Flagged clients: {len(flagged)}")
        if flagged:
            print("  " + ", ".join(flagged))
        print("========================================")

if __name__ == "__main__":
    print("Running Semi-Auto AdaptiveThreshold test with drift & poisoning simulation...")

    # ====== Synthetic FL-like pretraining ======
    def generate_fl_like_data(num_rounds=300):
        rng = np.random.default_rng(123)
        data = []
        for _ in range(num_rounds):
            mu = float(np.clip(rng.normal(0.15, 0.03), 0.05, 0.3))
            sigma = float(np.clip(rng.normal(0.05, 0.02), 0.01, 0.1))
            lam = 1.0
            prev_T = mu + lam * sigma + rng.normal(0, 0.01)
            target_T = mu + lam * sigma + rng.normal(0, 0.015)
            data.append({
                "mu": mu,
                "sigma": sigma,
                "lambda": lam,
                "prev_T": prev_T,
                "target_T": target_T
            })
        return data

    print("\n[Pretraining] Synthetic data for MLP residual corrector...")
    pretrain_data = generate_fl_like_data()
    at = AdaptiveThreshold(
        base_T=0.1, lambda_scale=1.0,
        clip_range=(0.05, 0.5),
        ema_beta=0.5, semi_mode=True,
        freeze_drift=0.25, patience=2
    )
    at.manual_train(pretrain_data, epochs=15)

    # ====== Initialize FL-like simulation ======
    rng = np.random.default_rng(42)
    num_clients = 10
    base_scores = {f"c{i+1}": float(rng.uniform(0.05, 0.35)) for i in range(num_clients)}

    poisoned_client = "c9"  # designate one attacker
    history_T = []
    history_mu = []
    history_sigma = []

    print("\n[Simulation] Running adaptive threshold test with injected attack...\n")
    for t in range(12):
        print(f"\n--- Round {t+1} ---")

        # Natural random evolution of scores
        scores = {
            k: float(np.clip(v + rng.normal(0, 0.05), 0, 1))
            for k, v in base_scores.items()
        }

        # Simulate poisoning (rounds 6–8)
        if 5 <= t <= 7:
            print(f"[Poisoning] {poisoned_client} spiking during round {t+1}")
            scores[poisoned_client] = float(np.clip(scores[poisoned_client] * 3.0, 0, 1))

        # Compute adaptive threshold
        result = at.compute_batch(scores)
        at.report(result)

        # Record metrics for analysis
        arr = np.array(list(scores.values()))
        mu, sigma = float(np.mean(arr)), float(np.std(arr))
        history_mu.append(mu)
        history_sigma.append(sigma)
        history_T.append(result["adaptive_threshold"])

        # Update base scores for next round
        base_scores = scores.copy()

    # ====== Optional Visualization ======
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 4))
        plt.plot(history_T, label="Adaptive Threshold", lw=2)
        plt.plot(np.array(history_mu) + np.array(history_sigma), '--', label="μ + σ (rule baseline)")
        plt.xlabel("Round")
        plt.ylabel("Threshold value")
        plt.title("Adaptive vs Rule-based Threshold Evolution")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    except ImportError:
        print("\n[Note] matplotlib not installed — skipping plot.")
