from typing import Optional, Sequence, Dict, Any
from copy import deepcopy
import math

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import copy
from collections import deque
import logging

DEFAULT_LMAX_PERCENTILE = 95

class AnchorEvaluator:
    """
    AnchorEvaluator performs public/synthetic dataset validation of a client's model update.
    """

    def __init__(self,
                    anchor_loader: DataLoader,
                    device: Optional[torch.device] = None,
                    running_std: float = 0.01,
                    kappa: float = 2.0,
                    use_original: bool = False,
                    eps: float = 1e-8,
                    rolling_window: int = 512,
                    lmax_percentile: int = DEFAULT_LMAX_PERCENTILE,
                    bn_calibration_loader: Optional[DataLoader] = None):
        """
        Initializes the AnchorEvaluator.

        Parameters
        ----------
        anchor_loader: DataLoader
            DataLoader for the anchor dataset (public / synthetic).
        device: torch.device or None
            Device for evaluation (defaults to model device).
        running_std: float
            estimated standard deviation of anchor accuracy (for z-score normalization).
        kappa: float
            Softness / steepness parameter for softPlus scaling.
        use_original: bool
            If True, use max(0, -delta_acc) instead of the smooth z-score+softPlus variant.
        eps: float
            Small epsilon for numeric stability.
        rolling_window: int
            Window size for tracking the rolling max of L_anchor.
        lmax_percentile: int
            Percentile for computing L_max (default is 95th percentile).
        bn_calibration_loader: Optional[DataLoader]
            Optional DataLoader for calibrating BN stats on clone using bn_calibration_loader.
        """

        self.anchor_loader = anchor_loader
        self.device = device
        self.running_std = float(running_std)
        self.kappa = float(kappa)
        self.use_original = bool(use_original)
        self.eps = float(eps)

        self._rolling_L = deque(maxlen=int(rolling_window))
        self._lmax_percentile = int(lmax_percentile)
        self._bn_calibration_loader = bn_calibration_loader
        self.logger = logging.getLogger("AnchorEvaluator")

    # ------------------ low-level helpers ------------------

    @staticmethod
    def _clone_model(model: nn.Module) -> nn.Module:
        return deepcopy(model)

    @staticmethod
    @torch.no_grad()
    def _evaluate_accuracy(model: nn.Module,
                            dataloader: DataLoader,
                            device: torch.device) -> float:
        model = model.eval().to(device)
        correct, total = 0, 0
        for batch in dataloader:
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            if logits.ndim == 1 or (logits.ndim == 2 and logits.shape[1] == 1):
                preds = (torch.sigmoid(logits.view(-1)) > 0.5).long()
            else:
                preds = logits.argmax(dim=1)
            correct += int((preds == y).sum().item())
            total += y.size(0)
        return float(correct / total) if total > 0 else 0.0

    @staticmethod
    def _softplus(x: float) -> float:
        return x if x > 20 else math.log1p(math.exp(x))

    def _compute_L_anchor(self, acc_g: float, acc_p: float) -> float:
        if self.use_original:
            delta = acc_p - acc_g
            return float(max(0.0, -delta))
        sigma = max(self.running_std, self.eps)
        z = (acc_g - acc_p) / sigma
        return float(self._softplus(z * self.kappa))

    def _safe_apply_delta(self, state_dict, client_delta, device):
        """Apply additive delta but only for matching keys/shapes. Returns warnings list."""
        warnings = []
        for k, v in client_delta.items():
            if k not in state_dict:
                warnings.append(f"missing_key:{k}")
                continue
            if state_dict[k].shape != v.shape:
                warnings.append(f"shape_mismatch:{k}:{state_dict[k].shape}!={v.shape}")
                continue
            # do arithmetic on device to avoid extra moves; we'll place back in state_dict as CPU for load_state_dict
            state_dict[k] = (state_dict[k].to(device) + v.to(device))
        return warnings

    def _bn_calibration(self, model_clone, device):
        """Optional short forward pass to calibrate BN stats on clone using bn_calibration_loader."""
        if self._bn_calibration_loader is None:
            return
        model_clone.train()
        with torch.no_grad():
            for idx, batch in enumerate(self._bn_calibration_loader):
                x, _ = batch
                model_clone(x.to(device))
                # limit the number of batches to keep calibration cheap
                if idx >= 5:
                    break
        model_clone.eval()

    def compute_L_max(self, percentile: Optional[int] = None, floor: float = 1e-6) -> float:
        """Compute adaptive L_max from the rolling buffer; fallback to floor."""
        if percentile is None:
            percentile = self._lmax_percentile
        arr = np.asarray(self._rolling_L, dtype=float)
        if arr.size == 0:
            return float(floor)
        val = float(np.percentile(arr, percentile))
        return max(val, float(floor))

    def _record_L(self, L_val: float):
        self._rolling_L.append(float(L_val))

    def get_normalized_score(self, L_anchor: float) -> float:
        """Convert L_anchor -> S_anchor in [0,1] using adaptive L_max."""
        L_max = self.compute_L_max()
        if L_max <= 0:
            return 1.0  # nothing observed yet; be permissive
        S = 1.0 - min(max(L_anchor / (L_max + self.eps), 0.0), 1.0)
        return float(S)

    def sandbox_validate(self, global_model, client_delta, client_id=None):
        if self.anchor_loader is None:
            self.logger.debug("[AnchorSandbox] Skipped: No anchor dataset available.")
            return None

        # clone model
        model_clone = self._clone_model(global_model)

        # determine device
        try:
            dev = next(model_clone.parameters()).device
        except StopIteration:
            dev = torch.device("cpu")

        # apply delta safely using same helper as compute()
        sd = model_clone.state_dict()
        warnings = self._safe_apply_delta(sd, client_delta, dev)

        # load modified state_dict back into clone (ensure tensors are CPU tensors)
        cpu_sd = {k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v) for k, v in sd.items()}
        model_clone.load_state_dict(cpu_sd)

        # optional BN calibration
        try:
            self._bn_calibration(model_clone, dev)
        except Exception as e:
            self.logger.warning(f"bn_calibration failed in sandbox client={client_id}: {e}")

        # evaluate baseline and updated model on anchor (use existing helpers)
        baseline_loss, baseline_acc = self._evaluate_model(global_model)
        new_loss, new_acc = self._evaluate_model(model_clone)

        return {
            "baseline_loss": baseline_loss,
            "new_loss": new_loss,
            "baseline_acc": baseline_acc,
            "new_acc": new_acc,
            "warnings": warnings
        }

    def _evaluate_model(self, model):
        model.eval()
        total_loss, total_correct, total = 0.0, 0, 0
        criterion = torch.nn.CrossEntropyLoss()

        with torch.no_grad():
            for x, y in self.anchor_loader:
                x, y = x.to(next(model.parameters()).device), y.to(next(model.parameters()).device)
                out = model(x)
                loss = criterion(out, y)
                total_loss += loss.item() * len(x)
                preds = out.argmax(dim=1)
                total_correct += (preds == y).sum().item()
                total += len(x)
        return total_loss / total, total_correct / total


    def compute(self, a_i: Optional[float] = None, *,
                global_model: Optional[nn.Module] = None,
                client_delta: Optional[Dict[str, torch.Tensor]] = None,
                precomputed_ref: Optional[float] = None,
                baseline_acc: Optional[float] = None,
                client_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate the anchor check score (L_anchor) for a client by computing the accuracy of the global model with and without the client's delta on the anchor dataset.
        
        Modes:
            (1) Server-side: provide global_model + client_delta
            (2) Client-side: provide precomputed_ref (median/baseline) and a_i (client's local anchor accuracy)
        
        Returns:
            dict with all component losses and the unified trust score.
        """
        
        # Evaluate global model on anchor dataset
        if global_model is not None and client_delta is not None:
            with torch.no_grad():
                if self.device is None:
                    try:
                        dev = next(global_model.parameters()).device
                    except StopIteration:
                        raise ValueError("global_model has no parameters")
                    device = dev
                else:
                    device = self.device
                
                acc_g = self._evaluate_accuracy(global_model, self.anchor_loader, device)

                model_clone = self._clone_model(global_model).to(device)
                sd = model_clone.state_dict()
                warnings = self._safe_apply_delta(sd, client_delta, device)

                # convert state dict tensors to CPU before load to be safe/consistent
                cpu_sd = {k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v) for k, v in sd.items()}
                model_clone.load_state_dict(cpu_sd)

                # optional bn calibration
                try:
                    self._bn_calibration(model_clone, device)
                except Exception as e:
                    self.logger.warning(f"bn_calibration failed client={client_id}: {e}")

                acc_p = self._evaluate_accuracy(model_clone, self.anchor_loader, device)
                L_anchor = self._compute_L_anchor(acc_g, acc_p)
                delta_acc = acc_p - acc_g

                # record for adaptive L_max
                self._record_L(L_anchor)

                # normalized score
                S_anchor = self.get_normalized_score(L_anchor)

                # structured log
                self.logger.info({
                    "client_id": client_id,
                    "acc_g": float(acc_g),
                    "acc_p": float(acc_p),
                    "L_anchor": float(L_anchor),
                    "S_anchor": float(S_anchor),
                    "delta_acc": float(delta_acc),
                    "warnings": warnings
                })

                return {"L_anchor": L_anchor, "acc_g": acc_g, "acc_p": acc_p,
                        "s_anchor": S_anchor, "delta_acc": delta_acc, "warnings": warnings}

        # client-side / precomputed mode
        # client provides its local anchor accuracy a_i and a reference value (median/baseline)
        if a_i is not None:
            if precomputed_ref is not None:
                ref = float(precomputed_ref)
            elif baseline_acc is not None:
                ref = float(baseline_acc)
            else:
                # insufficient info: return neutral safe result
                return {"L_anchor": None, "acc_g": None, "acc_p": None, "s_anchor": 0.0, "delta_acc": None, "warnings": []}

            delta = max(0.0, ref - float(a_i))
            # use running_std as sensitivity / scale (configurable)
            A_max = max(self.running_std, self.eps)
            s_anchor = float(np.clip(delta / (A_max + self.eps), 0.0, 1.0))
            # No model eval here, so L_anchor/acc values are None
            return {"L_anchor": None, "acc_g": None, "acc_p": None, "s_anchor": s_anchor, "delta_acc": -delta, "warnings": []}

        # fallback: nothing to do
        return {"L_anchor": None, "acc_g": None, "acc_p": None, "s_anchor": None, "delta_acc": None, "warnings": []}

    def compute_batch(self,
                        client_deltas: Optional[Sequence[Dict[str, torch.Tensor]]] = None,
                        precomputed_accs: Optional[Sequence[float]] = None,
                        baseline_acc: Optional[float] = None
                        ) -> Dict[str, Any]:
        """
        Batch compute for multiple clients.

        Modes:
            - Server-side: provide client_deltas list and global_model must be set per-call (not supported here).
                (This function focuses on batch post-hoc scoring using precomputed accs or deltas.)
            - Client-side: provide precomputed_accs (list of a_i) and baseline_acc.
                Returns reference (median or baseline) and array of s_anchor (float array).

        NOTE: For server-side full eval of many client_deltas use compute() in a loop (so each gets a clone).
        """
        if precomputed_accs is not None:
            arr = np.asarray(precomputed_accs, dtype=float)
            # choose reference: baseline if provided else median
            ref = float(baseline_acc) if baseline_acc is not None else float(np.median(arr))
            deltas = np.maximum(0.0, ref - arr)
            A_max = max(self.running_std, self.eps)
            s = deltas / (A_max + self.eps)
            s = np.clip(s, 0.0, 1.0)
            return {"reference": ref, "s_anchor": s}
        # fallback empty
        return {"reference": None, "s_anchor": np.array([], dtype=float)}


# For testing
if __name__ == "__main__":
    # small synthetic test: linear classifier on 2D points (sign(x0 + x1) -> label)
    print("Running AnchorEvaluator self-test (synthetic)...")

    torch.manual_seed(0)
    N = 200
    X = torch.randn(N, 2)
    y = ((X.sum(dim=1) + 0.2 * torch.randn(N)) > 0).long()

    dataset = TensorDataset(X, y)
    anchor_loader = DataLoader(dataset, batch_size=64, shuffle=False)

    # simple linear model
    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(2, 2)

        def forward(self, x):
            return self.lin(x)

    # create global model and a "benign" and "malicious" delta
    global_model = SimpleNet()
    # train global model slightly to have some baseline accuracy
    opt = torch.optim.SGD(global_model.parameters(), lr=0.1)
    global_model.train()
    for _ in range(10):
        for xb, yb in anchor_loader:
            logits = global_model(xb)
            loss = nn.CrossEntropyLoss()(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
    global_model.eval()

    # benign delta: small gaussian noise (should not catastrophically drop acc)
    benign_delta = {k: 0.01 * torch.randn_like(v) for k, v in global_model.state_dict().items()}
    # malicious delta: large negative weight perturbation (likely to reduce acc)
    malicious_delta = {k: (-0.5) * torch.randn_like(v) for k, v in global_model.state_dict().items()}

    ae = AnchorEvaluator(anchor_loader=anchor_loader, running_std=0.02, kappa=2.0, use_original=False)

    print("\nEvaluating benign delta...")
    out_b = ae.compute(global_model=global_model, client_delta=benign_delta)
    print("L_anchor:", out_b["L_anchor"], "acc_g:", out_b["acc_g"], "acc_p:", out_b["acc_p"], "s_anchor:", out_b["s_anchor"])

    print("\nEvaluating malicious delta...")
    out_m = ae.compute(global_model=global_model, client_delta=malicious_delta)
    print("L_anchor:", out_m["L_anchor"], "acc_g:", out_m["acc_g"], "acc_p:", out_m["acc_p"], "s_anchor:", out_m["s_anchor"])

    # Example client-side precomputed mode (clients report local anchor accuracy a_i)
    print("\nClient-side precomputed examples (a_i):")
    accs = [out_b["acc_p"], out_m["acc_p"], out_b["acc_g"]]
    batch_res = ae.compute_batch(precomputed_accs=accs, baseline_acc=out_b["acc_g"])
    print("reference:", batch_res["reference"])
    for idx, s in enumerate(batch_res["s_anchor"]):
        print(f"client {idx}: acc={accs[idx]:.4f}, s_anchor={s:.4f}")
