from typing import Optional, Dict, Any
import numpy as np
import torch
import json
import os
import time
from pathlib import Path

from .anchor_eval import AnchorEvaluator
from .kg_eval import KGConsistencyEvaluator
from .signature_eval import SignatureEvaluator
from .activation_eval import ActivationOutlierDetector

BASE_DIR = Path(__file__).resolve().parents[4]
DL_DIR = BASE_DIR / "deepcheck_ledger"
if not DL_DIR.exists():
    DL_DIR.mkdir()  

class DeepCheckManager:
    """
    DeepCheckManager performs the full multi-level trust evaluation
    for a client update in Federated Learning.
    """

    def __init__(
        self,
        anchor_eval=None,
        kg_eval=None,
        sig_eval=None,
        activation_eval=None,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 1.0,
        delta: float = 1.0,
        lmax_percentile: float = 95.0,
        eps: float = 1e-8,
        anchor_drop_tolerance: float = 0.05,
        anchor_loss_tolerance: float = 0.1,
        activation_reject_threshold: float = 0.3
    ):  
        """
        Initializes DeepCheckManager.

        Parameters
        ----------
        anchor_eval: AnchorEvaluator
            AnchorEvaluator instance for evaluating client updates on the anchor dataset.
        kg_eval: KGConsistencyEvaluator
            KGConsistencyEvaluator instance for evaluating client updates on the Knowledge Graph.
        sig_eval: Optional[SignatureEvaluator]
            SignatureEvaluator instance for evaluating client updates on the signature dataset.
        activation_eval: Optional[ActivationOutlierDetector]
            ActivationOutlierDetector instance for evaluating client updates on the signature dataset.
        alpha: float
            Weight for AnchorEvaluator's output.
        beta: float
            Weight for KGConsistencyEvaluator's output.
        gamma: float
            Weight for SignatureEvaluator's output.
        delta: float
            Weight for ActivationOutlierDetector's output.
        lmax_percentile: float
            Percentile for computing L_max.
        eps: float
            Small epsilon for numeric stability.
        anchor_drop_tolerance: float
            Max allowed drop in anchor accuracy (default 5%).
        anchor_loss_tolerance: float
            Max allowed increase in anchor loss (default 10%).
        activation_reject_threshold: float
            Threshold for activation outlier detection (default 30%).
        """
        self.anchor_eval = anchor_eval or AnchorEvaluator()
        self.kg_eval = kg_eval or KGConsistencyEvaluator()
        self.sig_eval = sig_eval or SignatureEvaluator()
        self.activation_eval = activation_eval or ActivationOutlierDetector()

        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.delta = float(delta)

        self.eps = float(eps)
        self._rolling_L = []
        self.lmax_percentile = float(lmax_percentile)
        self.anchor_drop_tolerance = float(anchor_drop_tolerance)
        self.anchor_loss_tolerance = float(anchor_loss_tolerance)
        self.activation_reject_threshold = activation_reject_threshold

    def _update_Lmax(self, L_check: float):
        self._rolling_L.append(L_check)
        if len(self._rolling_L) > 512:
            self._rolling_L.pop(0)

    def _compute_Lmax(self) -> float:
        if len(self._rolling_L) == 0:
            return 1.0
        return max(1e-6, float(np.percentile(self._rolling_L, self.lmax_percentile)))
    
    def reset_all_ema(self):
        """Reset EMA-related states for detectors (activation, etc.)."""
        if hasattr(self.activation_eval, "reset_ema"):
            self.activation_eval.reset_ema()
            print("[DeepCheckManager] Activation EMA reset.")

    def _get_ledger_path(self, round_id: int) -> Path:
        """Return ledger file path for the given round."""
        return DL_DIR / f"ledger_round_{round_id}.json"

    def compute(
        self,
        *,
        global_model: torch.nn.Module,
        client_delta: Dict[str, torch.Tensor],
        client_sig: Optional[torch.Tensor] = None,
        anchor_loader=None,
        client_id: Optional[str] = None,
        ref_sig: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Perform deep-check evaluation for a single client.

        Returns:
            dict with all component losses and the unified trust score.
        """
        results = {}
        sig_res, act_res = None, None

        # Anchor Sandbox Validation
        sandbox_res = self.anchor_eval.sandbox_validate(
            global_model=global_model,
            client_delta=client_delta,
            client_id=client_id,
        )

        if sandbox_res is not None:
            loss_baseline = sandbox_res["baseline_loss"]
            loss_after = sandbox_res["new_loss"]
            acc_baseline = sandbox_res["baseline_acc"]
            acc_after = sandbox_res["new_acc"]

            loss_increase = (loss_after - loss_baseline) / (loss_baseline + 1e-8)
            acc_drop = (acc_baseline - acc_after) / (acc_baseline + 1e-8)

            if loss_increase > self.anchor_loss_tolerance or acc_drop > self.anchor_drop_tolerance:
                results["sandbox_flag"] = "reject"
                results["sandbox_reason"] = {
                    "loss_increase": float(loss_increase),
                    "acc_drop": float(acc_drop)
                }
                print(f"[AnchorSandbox] Client {client_id} rejected — ΔLoss={loss_increase:.3f}, ΔAcc={acc_drop:.3f}")
                return {**results, "S_deep": 0.0}
            else:
                results["sandbox_flag"] = "accept"
                print(f"[AnchorSandbox] Client {client_id} passed sandbox — ΔLoss={loss_increase:.3f}, ΔAcc={acc_drop:.3f}")

        # Core Anchor Check
        anchor_res = self.anchor_eval.compute(
            global_model=global_model,
            client_delta=client_delta,
            client_id=client_id,
        )
        results.update(anchor_res)

        # KG Consistency
        if self.kg_eval is not None:
            kg_res = self.kg_eval.compute(
                global_model=global_model,
                client_delta=client_delta,
                anchor_loader=anchor_loader,
                client_id=client_id,
            )
            results.update(kg_res)
        else:
            results.update({"L_KG": None, "S_KG": None})

        # Activation-based Outlier Detection
        if self.activation_eval is not None and anchor_loader is not None:
            act_res = self.activation_eval.compute(
                global_model=global_model,
                client_delta=client_delta,
                anchor_loader=anchor_loader,
                client_id=client_id,
            )
            results.update(act_res)
        else:
            results.update({"S_activation": None})

        # Signature-based Trust
        if self.sig_eval is not None and client_sig is not None and ref_sig is not None:
            sig_res = self.sig_eval.compute(client_sig=client_sig, reference_sig=ref_sig)
            results.update(sig_res)
        else:
            results.update({"L_sig": None, "S_sig": None})

        # Combine Loss & Scores
        L_anchor = results.get("L_anchor", 0.0) or 0.0
        L_KG = results.get("L_KG", 0.0) or 0.0
        L_sig = results.get("L_sig", 0.0) or 0.0
        L_check = self.alpha * L_anchor + self.beta * L_KG + self.gamma * L_sig

        self._update_Lmax(L_check)
        L_max = self._compute_Lmax()
        S_deep = 1.0 - min(L_check / (L_max + self.eps), 1.0)

        S_activation = results.get("S_activation", 1.0)
        if S_activation < self.activation_reject_threshold:
            results["activation_flag"] = "reject"
            print(f"[ActivationCheck] Client {client_id} rejected — S_act={S_activation:.3f}")
            return {**results, "S_final": 0.0}

        S_combined = (
            (self.alpha * results.get("S_anchor", 1.0)) +
            (self.beta * results.get("S_KG", 1.0)) +
            (self.gamma * results.get("S_sig", 1.0)) +
            (self.delta * S_activation)
        ) / (self.alpha + self.beta + self.gamma + self.delta)

        results.update({
            "L_check": L_check,
            "L_max": L_max,
            "S_deep": S_deep,
            "S_activation": S_activation,
            "S_final": S_combined
        })

        # Unified Ledger Entry
        try:
            round_id = (
                (sig_res.get("round_id") if sig_res else None)
                or (act_res.get("round_id") if act_res else None)
                or int(time.time())
            )
            ledger_entry = {
                "client_id": client_id,
                "round_id": round_id,
                "sig_hash": sig_res.get("sig_hash", None) if sig_res else None,
                "act_hash": act_res.get("act_hash", None) if act_res else None,
                "S_sig": sig_res.get("S_sig", None) if sig_res else None,
                "S_activation": act_res.get("S_activation", None) if act_res else None,
                "activation_flag": act_res.get("activation_flag", "pass") if act_res else None,
                "ema_zmax": act_res.get("ema_zmax", None) if act_res else None,
                "drift_ema": act_res.get("drift_ema", None) if act_res else None,
                "S_final": results.get("S_final"),
                "timestamp": time.time(),
            }

            # Round-based ledger file
            ledger_file = self._get_ledger_path(round_id)
            existing = []
            if ledger_file.exists():
                with open(ledger_file, "r") as f:
                    existing = json.load(f)
            existing.append(ledger_entry)

            with open(ledger_file, "w") as f:
                json.dump(existing, f, indent=2)
            print(f"[Ledger] Logged entry for client {client_id} (round {round_id})")

        except Exception as e:
            print(f"[Ledger] Warning: failed to write ledger for {client_id}: {e}")



        return results
    
    def run_batch(
        self,
        *,
        global_model: torch.nn.Module,
        client_deltas: Dict[str, Any],
        candidate_clients: Optional[list] = None,
        client_sigs: Optional[Dict[str, torch.Tensor]] = None,
        ref_sigs: Optional[Dict[str, torch.Tensor]] = None,
        anchor_loader=None,
    ) -> Dict[str, Any]:
        """
        Run deep-check.compute() for each candidate client and return dict of per-client results.

        Parameters
        ----------
        global_model : torch.nn.Module
            Current global model used for sandbox evaluation.
        client_deltas : Dict[str, Any]
            Mapping client_id -> update (dict of tensors OR flattened array).
        candidate_clients : list, optional
            List of client IDs to evaluate. If None, all clients are processed.
        client_sigs : Dict[str, torch.Tensor], optional
            Client-side signature embeddings.
        ref_sigs : Dict[str, torch.Tensor], optional
            Reference signatures for comparison.
        anchor_loader : optional
            Dataloader for anchor sandbox evaluation.

        Returns
        -------
        Dict[str, Any]
            Mapping client_id -> result dict from DeepCheckManager.compute().
        """
        results = {}
        candidate_clients = candidate_clients or list(client_deltas.keys())

        for cid in candidate_clients:
            delta = client_deltas.get(cid)
            if delta is None:
                results[cid] = {"error": "no_delta"}
                continue

            # ---- Normalize delta format ----
            if isinstance(delta, dict):
                # Structured (layer-wise) delta: ensure tensor conversion
                client_delta = {
                    k: (torch.as_tensor(v, dtype=torch.float32)
                        if not isinstance(v, torch.Tensor) else v)
                    for k, v in delta.items()
                }
            else:
                # Flattened numpy/tensor vector: enforce 1D conversion
                arr = np.asarray(delta)
                if arr.ndim > 2:
                    print(f"[WARN] Client {cid}: delta has shape {arr.shape}, flattening.")
                    arr = arr.reshape(-1)
                client_delta = {"flatten": torch.as_tensor(arr, dtype=torch.float32)}
                print(f"[DeepCheck] Client {cid}: using fallback 'flatten' delta structure.")

            # ---- Optional signatures ----
            client_sig = client_sigs.get(cid) if client_sigs else None
            ref_sig = ref_sigs.get(cid) if ref_sigs else None

            # ---- Compute deep check ----
            try:
                res = self.compute(
                    global_model=global_model,
                    client_delta=client_delta,
                    client_sig=client_sig,
                    anchor_loader=anchor_loader,
                    client_id=cid,
                    ref_sig=ref_sig,
                )
                results[cid] = res
            except Exception as e:
                results[cid] = {"error": str(e)}
                print(f"[DeepCheckManager] Client {cid} failed: {e}")

        return results
