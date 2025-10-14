import torch
import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import zscore
import copy
import time
import hashlib

class ActivationOutlierDetector:
    """
    Detect activation-space anomalies that may indicate backdoor or trigger behavior.
    Collects internal activations on an anchor dataset and computes statistical drift.
    """

    def __init__(
        self,
        layer_name: str = "features",
        n_components: int = 2,
        outlier_threshold: float = 3.0,
        max_samples: int = 128,
        ema_decay: float = 0.9,
        ema_enabled: bool = True, 
    ):
        """
        Parameters
        ----------
        layer_name : str
            Layer name (substring match) to hook for activation extraction.
        n_components : int
            PCA dimensionality for compression.
        outlier_threshold : float
            Z-score threshold for flagging outliers.
        max_samples : int
            Max number of anchor samples to analyze to avoid memory overflow.
        """
        self.layer_name = layer_name
        self.n_components = n_components
        self.outlier_threshold = outlier_threshold
        self.max_samples = max_samples

        self.ema_decay = ema_decay
        self.ema_enabled = ema_enabled

        # Rolling EMA state
        self._ema_mean = None
        self._ema_std = None
        self._ema_zmax = None

    def reset_ema(self):
        """Reset EMA drift tracking to its initial (None) state."""
        self._ema_mean = None
        self._ema_std = None
        self._ema_zmax = None
        print("[ActivationOutlierDetector] EMA state reset.")

    def compute(
        self,
        *,
        global_model: torch.nn.Module,
        client_delta: dict,
        anchor_loader,
        client_id: str = None,
    ) -> dict:
        """
        Apply delta, collect activations, and compute anomaly metrics.

        Returns
        -------
        dict:
            {
                "S_activation": float,
                "activation_zmax": float,
                "activation_mean": float,
                "pca_var_ratio": float,
                "act_hash": str,
                "activation_flag": str,
                "round_id": int
            }
        """
        model = copy.deepcopy(global_model)
        model.eval()

        # --- Apply client delta to model parameters ---
        with torch.no_grad():
            for name, param in model.named_parameters():
                if isinstance(client_delta, dict) and name in client_delta:
                    try:
                        delta_tensor = client_delta[name].to(param.device)
                        if delta_tensor.shape == param.shape:
                            param.add_(delta_tensor)
                    except Exception as e:
                        print(f"[ActivationCheck] {client_id}: delta apply failed on {name} ({e})")
                elif "flatten" in client_delta:
                    # If flattened delta: skip, or handle with future adapter
                    break

        # --- Hook activations ---
        activations = []
        handle = None

        def hook_fn(_, __, output):
            if isinstance(output, torch.Tensor):
                activations.append(output.detach().cpu().flatten(1))

        for n, m in model.named_modules():
            if self.layer_name in n:
                handle = m.register_forward_hook(hook_fn)
                break

        if handle is None:
            print(f"[ActivationCheck] {client_id}: No layer matching '{self.layer_name}' found.")
            return {"S_activation": 1.0, "activation_flag": "pass"}

        # --- Collect activations on anchor dataset ---
        model_device = next(model.parameters()).device
        sample_count = 0
        with torch.no_grad():
            for x, _ in anchor_loader:
                x = x.to(model_device)
                model(x)
                sample_count += x.size(0)
                if sample_count >= self.max_samples:
                    break

        handle.remove()

        if not activations:
            print(f"[ActivationCheck] {client_id}: No activations captured.")
            return {"S_activation": 1.0, "activation_flag": "pass"}

        acts = torch.cat(activations, dim=0).numpy()
        acts = np.nan_to_num(acts)

        # --- Z-score drift detection ---
        acts_z = np.abs(zscore(acts, axis=0))
        zmax = float(np.nan_to_num(acts_z.max(), nan=0.0))
        mean_activation = float(np.mean(acts))
        std_activation = float(np.std(acts))

        # --- EMA-based drift ---
        if self.ema_enabled:
            if self._ema_mean is None:
                self._ema_mean = mean_activation
                self._ema_std = std_activation
                self._ema_zmax = zmax
            else:
                self._ema_mean = (
                    self.ema_decay * self._ema_mean + (1 - self.ema_decay) * mean_activation
                )
                self._ema_std = (
                    self.ema_decay * self._ema_std + (1 - self.ema_decay) * std_activation
                )
                self._ema_zmax = (
                    self.ema_decay * self._ema_zmax + (1 - self.ema_decay) * zmax
                )

            drift_ema = abs(zmax - self._ema_zmax) / (self._ema_zmax + 1e-8)
        else:
            drift_ema = 0.0


        # --- PCA variance ratio (proxy for complexity drift) ---
        try:
            pca = PCA(n_components=min(self.n_components, acts.shape[1]))
            pca.fit(acts)
            var_ratio = float(np.sum(pca.explained_variance_ratio_))
        except Exception as e:
            print(f"[ActivationCheck] {client_id}: PCA failed ({e})")
            var_ratio = 0.0

        # --- Drift-based trust score ---
        drift_score = min(1.0, (zmax + drift_ema) / self.outlier_threshold)
        S_activation = max(0.0, float(1.0 - drift_score))

        print(f"[ActivationCheck] {client_id}: zmax={zmax:.3f}, var_ratio={var_ratio:.2f}, S_act={S_activation:.3f}")

        # --- Privacy-safe activation hash ---
        try:
            act_vector = np.concatenate([
                acts.mean(axis=0).flatten(),
                acts.std(axis=0).flatten()
            ])
            act_hash = hashlib.sha256(act_vector.tobytes()).hexdigest()[:16]
        except Exception:
            act_hash = None

        activation_flag = "reject" if S_activation < 0.3 else "pass"

        return {
            "S_activation": S_activation,
            "activation_mean": mean_activation,
            "activation_std": std_activation,
            "activation_zmax": zmax,
            "ema_zmax": self._ema_zmax,
            "drift_ema": drift_ema,
            "pca_var_ratio": var_ratio,
            "act_hash": act_hash,
            "activation_flag": activation_flag,
            "round_id": int(time.time())
        }
