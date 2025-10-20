import torch
import numpy as np
from sklearn.decomposition import PCA

import copy
import os
import time
import hashlib
from torch import nn
from scipy.stats import median_abs_deviation as mad
from torch.utils.hooks import RemovableHandle

class ActivationOutlierDetector:
    """
    Detect activation-space anomalies that may indicate backdoor or trigger behavior.
    Collects internal activations on an anchor dataset and computes statistical drift.

    This version auto-detects a sensible module to hook if the configured `layer_name`
    substring is not present in the model.
    """

    def __init__(
        self,
        layer_name: str = "encoder",
        n_components: int = 2,
        outlier_threshold: float = 15.0,
        max_samples: int = 128,
        ema_decay: float = 0.9,
        ema_enabled: bool = True,
        activation_reject_threshold: float = 0.3,
        eps: float = 1e-6,
    ):
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

        # hook handle / bookkeeping for repeated runs
        self._current_handle = None
        self._current_hooked_module = None

        self.activation_reject_threshold = activation_reject_threshold
        self.eps = float(eps)

    def reset_ema(self):
        """Reset EMA drift tracking to its initial (None) state."""
        self._ema_mean = None
        self._ema_std = None
        self._ema_zmax = None
        print("[ActivationOutlierDetector] EMA state reset.")

    def _reset_hook_state(self):
        """Remove any previously-registered hook (defensive), reset transient hook bookkeeping."""
        try:
            if getattr(self, "_current_handle", None) is not None:
                try:
                    self._current_handle.remove()
                except Exception:
                    pass
        finally:
            self._current_handle = None
            self._current_hooked_module = None

    @staticmethod
    def _find_module_name_for_hook(model: torch.nn.Module, prefer_substr: str = None):
        """
        Return an exact module name (string) suitable for substring matching when registering hook.
        Preference order:
          1. Any module name containing prefer_substr (if provided)
          2. Any module name containing a preferred substring from list
          3. First module instance that is Linear/Conv2d/Embedding/LayerNorm
          4. None
        """
        names_and_modules = list(model.named_modules())
        names = [n for n, _ in names_and_modules]

        if prefer_substr:
            for n in names:
                if prefer_substr in n:
                    return n  # return actual module name that contains substring

        preferred = ["encoder", "backbone", "token_embeddings", "embed", "features", "classifier"]
        for p in preferred:
            for n in names:
                if p in n:
                    return n

        for n, m in names_and_modules:
            if isinstance(m, (nn.Linear, nn.Conv2d, nn.Embedding, nn.LayerNorm)):
                return n

        return None

    def compute_zmax_distribution(self, global_model, anchor_loader, layer_name=None, max_samples=512):
        """
        Run the model on anchor_loader and collect zmax for each batch/mini-collection.
        Returns dict with summary stats and array of observed zmax values.
        (This is for offline calibration / debugging.)
        """
        layer_name = layer_name or self.layer_name
        model = copy.deepcopy(global_model)
        model.eval()

        zmax_list = []

        # similar hook_fn but we compute one zmax per batch
        def collect_zmax_for_batch(x):
            # forward, hook into same module, collect activations and compute zmax
            activations = []
            def hook_fn(_, __, output):
                out = output
                if isinstance(output, (tuple, list)) and len(output) > 0:
                    out = output[0]
                if isinstance(out, torch.Tensor):
                    try:
                        activations.append(out.detach().cpu().flatten(1))
                    except Exception:
                        activations.append(out.detach().cpu().reshape(out.size(0), -1))

            chosen = self._find_module_name_for_hook(model, prefer_substr=layer_name)
            handle = None
            if chosen:
                for n, m in model.named_modules():
                    if n == chosen or chosen in n:
                        handle = m.register_forward_hook(hook_fn)
                        break
            if handle is None:
                return None

            with torch.no_grad():
                try:
                    model(x)
                except Exception:
                    try:
                        handle.remove()
                    except Exception:
                        pass
                    return None

            try:
                handle.remove()
            except Exception:
                pass

            if not activations:
                return None
            
            acts = torch.cat(activations, dim=0).numpy()
            acts = np.nan_to_num(acts)
            if acts.size == 0 or acts.shape[1] == 0:
                return None
            
            # stable zscore per column
            col_mean = acts.mean(axis=0)
            col_std = acts.std(axis=0)
            col_std = np.maximum(col_std, self.eps)
            acts_z = np.abs((acts - col_mean) / col_std)
            zmax = float(np.nan_to_num(np.nanmax(acts_z), nan=0.0))
            return zmax

        model_device = next(model.parameters()).device
        count = 0
        with torch.no_grad():
            for batch in anchor_loader:
                if isinstance(batch, (tuple, list)) and len(batch) >= 1:
                    x = batch[0]
                else:
                    x = batch
                x = x.to(model_device)
                z = collect_zmax_for_batch(x)
                if z is not None:
                    zmax_list.append(z)
                count += x.size(0)
                if count >= max_samples:
                    break

        if not zmax_list:
            return {"zmax": np.array([]), "summary": {}}

        arr = np.array(zmax_list)
        summary = {
            "count": int(arr.size),
            "min": float(arr.min()),
            "median": float(np.median(arr)),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "p75": float(np.percentile(arr, 75)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "mad": float(mad(arr)) if arr.size > 0 else 0.0
        }
        return {"zmax": arr, "summary": summary}

    def set_threshold_from_baseline(self, baseline_path=None, method="percentile", p=95, k_mad=6.0, layer_name=None):
        """
        Set self.outlier_threshold using either:
            - method="percentile": threshold = p-th percentile of baseline zmax
            - method="mad": threshold = median(zmax) + k_mad * MAD
        If baseline_path provided, try to load saved baseline stats (and maybe zmax array you saved).
        """
        # If baseline file contains a saved array of zmax values, you can load it; otherwise user should
        # call compute_zmax_distribution() and pass results.
        # For convenience, look if baseline_path is an npz with 'zmax' array
        baseline = None
        if baseline_path and os.path.exists(baseline_path):
            try:
                data = np.load(baseline_path, allow_pickle=True)
                # support both saved array under 'zmax' or whole array
                if "zmax" in data:
                    baseline = np.asarray(data["zmax"])
                else:
                    # if stored as first array
                    try:
                        baseline = np.asarray(data[list(data.files)[0]])
                    except Exception:
                        baseline = None
            except Exception:
                baseline = None

        if baseline is None:
            print("[ActivationOutlierDetector] No baseline zmax array provided or found; please call compute_zmax_distribution() first.")
            return False

        arr = np.asarray(baseline)
        if arr.size == 0:
            print("[ActivationOutlierDetector] Baseline zmax array empty.")
            return False

        if method == "percentile":
            thr = float(np.percentile(arr, p))
            bootstrap = float(np.percentile(arr, 50))  # median as a good EMA seed
        else:  # mad
            med = float(np.median(arr))
            m = float(mad(arr))
            thr = med + float(k_mad) * m
            bootstrap = med

        thr = max(thr, 1.0)
        self.outlier_threshold = thr

        # Initialize ema_zmax to a robust baseline point (median/percentile)
        self._ema_zmax = max(bootstrap, 1.0)
        print(f"[ActivationOutlierDetector] outlier_threshold set -> {thr:.3f} (method={method}), ema_zmax init -> {self._ema_zmax:.3f}")
        return True

    # -----------------------
    # Compute (main)
    # -----------------------
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
        Returns a dict with activation metrics and a trust score S_activation.
        """
        # Ensure no stale hooks
        self._reset_hook_state()

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
                elif isinstance(client_delta, dict) and "_flat_update" in client_delta:
                    # If flattened delta: skip or handle with an adapter (not implemented here)
                    break

        # --- Hook activations ---
        activations = []
        handle = None
        hooked_module_name = None

        def hook_fn(_, __, output):
            # support Tensor or tuple/list outputs (take first Tensor)
            out = output
            if isinstance(output, (tuple, list)) and len(output) > 0:
                out = output[0]
            if isinstance(out, torch.Tensor):
                # flatten from dim=1 onward (keep batch dim)
                try:
                    activations.append(out.detach().cpu().flatten(1))
                except Exception:
                    # fallback: flatten entire tensor
                    activations.append(out.detach().cpu().reshape(out.size(0), -1))

        chosen_name = self._find_module_name_for_hook(model, prefer_substr=self.layer_name)
        if chosen_name is None:
            print(f"[ActivationCheck] {client_id}: No suitable hook candidate found (tried '{self.layer_name}').")
            return {"S_activation": 1.0, "activation_flag": "pass"}

        # Register hook on the first module whose name exactly matches chosen_name
        for n, m in model.named_modules():
            if n == chosen_name:
                handle = m.register_forward_hook(hook_fn)
                hooked_module_name = n
                break

        if handle is None:
            # defensive fallback: try substring match for the chosen_name
            for n, m in model.named_modules():
                if chosen_name in n:
                    handle = m.register_forward_hook(hook_fn)
                    hooked_module_name = n
                    break

        if handle is None:
            print(f"[ActivationCheck] {client_id}: No layer matching '{self.layer_name}'/{chosen_name} found.")
            return {"S_activation": 1.0, "activation_flag": "pass"}

        # store on self so reset is deterministic
        self._current_handle = handle
        self._current_hooked_module = hooked_module_name

        print(f"[ActivationCheck] {client_id}: hooked module '{hooked_module_name}' (using preference '{self.layer_name}')")

        # --- Collect activations on anchor dataset (ensure hook removal on all exits) ---
        try:
            model_device = next(model.parameters()).device
        except StopIteration:
            # empty model? treat as pass
            self._reset_hook_state()
            print(f"[ActivationCheck] {client_id}: model has no parameters; skipping activation check.")
            return {"S_activation": 1.0, "activation_flag": "pass"}

        sample_count = 0
        try:
            with torch.no_grad():
                for batch in anchor_loader:
                    # support (x, y) or x-only loaders
                    if isinstance(batch, (tuple, list)) and len(batch) >= 1:
                        x = batch[0]
                    else:
                        x = batch
                    x = x.to(model_device)
                    try:
                        model(x)
                    except Exception as e:
                        # forward failed; return but ensure finally will cleanup hook
                        print(f"[ActivationCheck] {client_id}: model forward failed during activation collection ({e})")
                        return {"S_activation": 1.0, "activation_flag": "pass"}

                    sample_count += x.size(0)
                    if sample_count >= self.max_samples:
                        break
        finally:
            # guaranteed removal of the hook handle even on early return/exception
            self._reset_hook_state()

        # activations should have been populated by the hook
        if not activations:
            print(f"[ActivationCheck] {client_id}: No activations captured.")
            return {"S_activation": 1.0, "activation_flag": "pass"}

        acts = torch.cat(activations, dim=0).numpy()
        acts = np.nan_to_num(acts)

        # --- Z-score drift detection ---
        if acts.size == 0 or acts.shape[1] == 0:
            print(f"[ActivationCheck] {client_id}: activations empty after concat.")
            return {"S_activation": 1.0, "activation_flag": "pass"}

        # stable per-feature zscore
        col_mean = acts.mean(axis=0)
        col_std = acts.std(axis=0)
        col_std = np.maximum(col_std, self.eps)   # floor for numerical stability
        acts_z = np.abs((acts - col_mean) / col_std)
        zmax = float(np.nan_to_num(np.nanmax(acts_z), nan=0.0))
        mean_activation = float(np.mean(acts))
        std_activation = float(np.std(acts))

        # --- EMA-based drift ---
        if self.ema_enabled:
            try:
                if self._ema_mean is None:
                    self._ema_mean = mean_activation
                    self._ema_std = std_activation
                    self._ema_zmax = float(zmax)
                else:
                    # EMA: keep previous scale (decay * old + (1-decay)*new)
                    d = float(self.ema_decay)
                    self._ema_mean = (d * self._ema_mean) + ((1.0 - d) * mean_activation)
                    self._ema_std = (d * self._ema_std) + ((1.0 - d) * std_activation)
                    self._ema_zmax = (d * float(self._ema_zmax)) + ((1.0 - d) * float(zmax))
            except Exception as e:
                print(f"[ActivationCheck] Warning: EMA update failed: {e}")

            drift_ema = abs(zmax - self._ema_zmax) / (self._ema_zmax + 1e-8)
        else:
            drift_ema = 0.0

        # --- PCA variance ratio (proxy for complexity drift) ---
        try:
            pca = PCA(n_components=min(self.n_components, max(1, acts.shape[1])))
            pca.fit(acts)
            var_ratio = float(np.sum(pca.explained_variance_ratio_))
        except Exception as e:
            print(f"[ActivationCheck] {client_id}: PCA failed ({e})")
            var_ratio = 0.0

        # --- Drift-based trust score ---
        ratio = (zmax + drift_ema) / (self.outlier_threshold + 1e-12)

        # smooth activation trust using Gaussian decay — avoids hard 0 cutoff
        if ratio <= 1.0:
            S_activation = 1.0
        else:
            S_activation = float(np.exp(-0.25 * (ratio - 1.0)**2))

        # safety clamp
        S_activation = max(0.0, min(1.0, S_activation))

        # debug: print final, after all computed
        print(
            f"[ActivationCheck] {client_id}: zmax={zmax:.3f}, outlier_thr={self.outlier_threshold:.3f}, "
            f"ema_zmax={self._ema_zmax:.3f}, drift_ema={drift_ema:.3f}, var_ratio={var_ratio:.3f}, S_act={S_activation:.3f}"
        )

        # --- Privacy-safe activation hash ---
        try:
            act_vector = np.concatenate([
                acts.mean(axis=0).flatten(),
                acts.std(axis=0).flatten()
            ])
            act_hash = hashlib.sha256(act_vector.tobytes()).hexdigest()[:16]
        except Exception:
            act_hash = None

        activation_flag = "reject" if S_activation < self.activation_reject_threshold else "pass"

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

    # -----------------------
    # Baseline helpers
    # -----------------------
    @staticmethod
    def build_baseline(global_model, anchor_loader, layer_name="features",
                       max_samples=512, save_path="baseline_activation_stats.npz"):
        """
        Compute and save baseline activation statistics from trusted (clean) data.
        Returns True on success, False if no suitable hook was found.
        """
        model = copy.deepcopy(global_model)
        model.eval()

        activations = []
        handle = None
        hooked_module_name = None

        def hook_fn(_, __, output):
            out = output
            if isinstance(output, (tuple, list)) and len(output) > 0:
                out = output[0]
            if isinstance(out, torch.Tensor):
                try:
                    activations.append(out.detach().cpu().flatten(1))
                except Exception:
                    activations.append(out.detach().cpu().reshape(out.size(0), -1))

        chosen_name = ActivationOutlierDetector._find_module_name_for_hook(model, prefer_substr=layer_name)
        if chosen_name is None:
            print(f"[ActivationOutlierDetector] No layer containing '{layer_name}' found in model; baseline creation aborted.")
            return False

        for n, m in model.named_modules():
            if n == chosen_name:
                handle = m.register_forward_hook(hook_fn)
                hooked_module_name = n
                break
        if handle is None:
            for n, m in model.named_modules():
                if chosen_name in n:
                    handle = m.register_forward_hook(hook_fn)
                    hooked_module_name = n
                    break

        if handle is None:
            print(f"[ActivationOutlierDetector] No layer matching '{layer_name}'/{chosen_name} found; baseline creation aborted.")
            return False

        print(f"[ActivationOutlierDetector] Building baseline using hooked module '{hooked_module_name}' (pref '{layer_name}')")
        layer_name = hooked_module_name
        print(f"[ActivationOutlierDetector] Stored baseline layer_name -> '{layer_name}'")

        model_device = next(model.parameters()).device
        sample_count = 0
        with torch.no_grad():
            for batch in anchor_loader:
                if isinstance(batch, (tuple, list)) and len(batch) >= 1:
                    x = batch[0]
                else:
                    x = batch
                x = x.to(model_device)
                model(x)
                sample_count += x.size(0)
                if sample_count >= max_samples:
                    break

        try:
            handle.remove()
        except Exception:
            pass

        if not activations:
            print("[ActivationOutlierDetector] No activations captured for baseline; aborting.")
            return False

        acts = torch.cat(activations, dim=0).numpy()
        # For baselin save per-batch zmax array for calibration convenience:
        # compute zmax per-slice (coarse: here compute single zmax across all activations)
        col_mean = acts.mean(axis=0)
        col_std = acts.std(axis=0)
        col_std = np.maximum(col_std, 1e-6)
        acts_z = np.abs((acts - col_mean) / col_std)
        zmax_all = float(np.nan_to_num(np.nanmax(acts_z), nan=0.0))

        # Save baseline summary + optionally the zmax array (single value here).
        np.savez(save_path,
                    mean=np.mean(acts, axis=0),
                    std=np.std(acts, axis=0),
                    cov=np.cov(acts, rowvar=False),
                    zmax=np.array([zmax_all], dtype=np.float32))
        print(f"[ActivationOutlierDetector] Baseline saved to {save_path} (zmax={zmax_all:.3f})")
        return True

    def load_baseline(self, path="baseline_activation_stats.npz"):
        """
        Load precomputed baseline mean/std into EMA state. If zmax array is saved, use it to init ema_zmax.
        """
        if not os.path.exists(path):
            print(f"[ActivationOutlierDetector] Baseline file not found: {path}")
            return False

        data = np.load(path, allow_pickle=True)
        self._ema_mean = float(np.mean(data["mean"])) if "mean" in data else None
        self._ema_std = float(np.mean(data["std"])) if "std" in data else None

        # if a zmax array was stored, init ema_zmax from its median
        if "zmax" in data:
            arr = np.asarray(data["zmax"])
            if arr.size > 0:
                self._ema_zmax = float(np.median(arr))
            else:
                self._ema_zmax = 1.0
        else:
            # conservative default
            self._ema_zmax = 1.0

        print(f"[ActivationOutlierDetector] Loaded baseline stats from {path} (ema_zmax={self._ema_zmax:.3f})")
        return True