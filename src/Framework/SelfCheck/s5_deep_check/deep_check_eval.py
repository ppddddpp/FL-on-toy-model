from typing import Optional, Dict, Any
import numpy as np
import torch
import json
import warnings
import time
from pathlib import Path
import tempfile
import os

from .anchor_eval import AnchorEvaluator
from .kg_eval import KGConsistencyEvaluator
from .signature_eval import SignatureEvaluator
from .activation_eval import ActivationOutlierDetector
from .calibration import SafeThresholdCalibrator
from Helpers.Helpers import log_and_print, log_round_summary

BASE_DIR = Path(__file__).resolve().parents[4]
DL_DIR = BASE_DIR / "deepcheck_ledger"
if not DL_DIR.exists():
    DL_DIR.mkdir()  

class DeepCheckManager:
    """
    DeepCheckManager performs the full multi-level trust evaluation for a client update.
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
        activation_reject_threshold: float = 0.3,
        anchor_loader = None,
        log_dir: Path = BASE_DIR / "logs" / "run.txt",
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
        anchor_loader: DataLoader
            DataLoader for the anchor dataset (public / synthetic).
        log_dir : Path
            Path to log file
        """
        self.anchor_eval = anchor_eval or AnchorEvaluator(anchor_loader)
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

        # per-client reference signatures
        self.ref_sigs = {}             # client_id -> torch.Tensor
        self.ref_update_alpha = 0.1    # EMA update rate for baseline
        self.ref_update_min_Ssig = 0.75 # only update when S_sig >= this

        # dirty flag used when commit_ref_sigs mutates ref_sigs
        self._ref_sigs_dirty = False
        _ref_path = DL_DIR / "ref_sigs.pt"
        if _ref_path.exists():
            try:
                self.load_ref_sigs(str(_ref_path))
            except Exception:
                pass
        
        self.log_dir = log_dir


    def _update_Lmax(self, L_check: float):
        self._rolling_L.append(L_check)
        if len(self._rolling_L) > 512:
            self._rolling_L.pop(0)

    def normalize_client_delta(self, delta):
        """
        Return dict[str, torch.Tensor] or {} if unable to coerce.
        Accepts:
        - dict: {k: tensor_or_array}
        - list/tuple of (k,v) pairs
        - list/tuple of tensors (positional) -> param_0, param_1, ...
        - single tensor/ndarray -> {'_flat_update': tensor}
        """
        def to_tensor(x):
            if isinstance(x, torch.Tensor):
                return x
            try:
                return torch.as_tensor(x, dtype=torch.float32)
            except Exception:
                return None

        if delta is None:
            return {}

        # already dict-like: map keys -> tensors (keep only values convertible to tensors)
        if isinstance(delta, dict):
            out = {}
            for k, v in delta.items():
                t = to_tensor(v)
                if t is not None:
                    out[str(k)] = t
            return out

        # list/tuple of pairs
        if isinstance(delta, (list, tuple)):
            # prefer explicit (k,v) pairs
            if all(isinstance(x, (list, tuple)) and len(x) == 2 for x in delta):
                out = {}
                for k, v in delta:
                    t = to_tensor(v)
                    if t is not None:
                        out[str(k)] = t
                return out
            # otherwise treat as list of tensors -> generate names
            if all(isinstance(x, (torch.Tensor, np.ndarray, list)) for x in delta):
                out = {}
                for i, v in enumerate(delta):
                    t = to_tensor(v)
                    if t is not None:
                        out[f"param_{i}"] = t
                return out
            # fallback: single-element list
            if len(delta) == 1:
                t = to_tensor(delta[0])
                return {"_flat_update": t} if t is not None else {}

        # single tensor/ndarray
        if isinstance(delta, (torch.Tensor, np.ndarray, list)):
            t = to_tensor(delta)
            return {"_flat_update": t} if t is not None else {}

        # unknown type
        return {}


    def _compute_Lmax(self) -> float:
        if len(self._rolling_L) == 0:
            return 1.0
        return max(1e-6, float(np.percentile(self._rolling_L, self.lmax_percentile)))
    
    def reset_all_ema(self):
        """Reset EMA-related states for detectors (activation, etc.)."""
        if hasattr(self.activation_eval, "reset_ema"):
            # self.activation_eval.reset_ema()
            log_and_print("[DeepCheckManager] Activation EMA reset.", log_file=self.log_dir)

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

        # --- Normalize delta format (handle dict / list / tensor) ---
        def _to_tensor(v):
            try:
                return v.detach().cpu().float() if torch.is_tensor(v) else torch.as_tensor(v, dtype=torch.float32)
            except Exception:
                return torch.tensor(v, dtype=torch.float32)

        log_and_print(f"[DeepCheckManager] Normalizing client_delta for {client_id} (type={type(client_delta)})", log_file=self.log_dir)

        # Safe coercion for all possible formats
        try:
            if hasattr(client_delta, "items"):
                # Regular dict-like object
                client_delta = {str(k): _to_tensor(v) for k, v in client_delta.items()}
            elif isinstance(client_delta, (list, tuple)):
                # Only use entries that are (k,v) pairs
                valid_pairs = [p for p in client_delta if isinstance(p, (tuple, list)) and len(p) == 2]
                if valid_pairs:
                    client_delta = {str(k): _to_tensor(v) for k, v in valid_pairs}
                else:
                    # Fallback for list of tensors
                    client_delta = {f"param_{i}": _to_tensor(v) for i, v in enumerate(client_delta)}
            elif torch.is_tensor(client_delta) or isinstance(client_delta, (np.ndarray, list)):
                # Single tensor or ndarray
                client_delta = {"_flat_update": _to_tensor(client_delta)}
            else:
                # Unknown type fallback
                client_delta = {}
        except Exception as e:
            log_and_print(f"[DeepCheckManager] Warning: failed to normalize delta for {client_id}: {e}", log_file=self.log_dir)
            client_delta = {}

        log_and_print(f"[DeepCheckManager] Normalized client_delta for {client_id} (type={type(client_delta)})", log_file=self.log_dir)

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
                log_and_print(f"[AnchorSandbox] Client {client_id} rejected — delta_Loss={loss_increase:.3f}, delta_Acc={acc_drop:.3f}",
                        log_file=self.log_dir
                    )
                return {**results, "S_deep": 0.0}
            else:
                results["sandbox_flag"] = "accept"
                log_and_print(f"[AnchorSandbox] Client {client_id} passed sandbox — delta_Loss={loss_increase:.3f}, delta_Acc={acc_drop:.3f}",
                        log_file=self.log_dir
                    )
        else:
            raise RuntimeError("Anchor sandbox validation failed.")
        
        # Core Anchor Check
        if self.anchor_eval is not None:
            anchor_res = self.anchor_eval.compute(
                global_model=global_model,
                client_delta=client_delta,
                client_id=client_id,
            )
            if anchor_res is None:
                warnings.warn("Anchor check failed.")
                raise RuntimeError("Anchor check failed.")
            results.update(anchor_res)
        else:
            warnings.warn("Anchor check failed due to missing anchor eval config.")
            results.update({"L_anchor": None, "S_anchor": None})

        # KG Consistency
        if self.kg_eval is not None:
            kg_res = self.kg_eval.compute(
                global_model=global_model,
                client_delta=client_delta,
                anchor_loader=anchor_loader,
                client_id=client_id,
            )
            results.update(kg_res)
            if kg_res is None:
                warnings.warn("KG check failed.")
                raise RuntimeError("KG check failed.")
        else:
            warnings.warn("KG check failed due to misssing KG eval config.")
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
            if act_res is None:
                warnings.warn("Activation check failed.")
                raise RuntimeError("Activation check failed.")
        else:
            warnings.warn("Activation check failed due to missing activation eval config.")
            results.update({"S_activation": None})

        # --- Signature-based Trust ---
        if self.sig_eval is not None:
            # Ensure client_sig exists (auto-generate from client_delta if needed)
            if client_sig is None:
                try:
                    if hasattr(self.sig_eval, "encode"):
                        client_sig = self.sig_eval.encode(client_delta, dim=256, device="cpu")
                    elif hasattr(self.sig_eval, "make_signature_from_delta"):
                        client_sig = self.sig_eval.make_signature_from_delta(client_delta, dim=256, device="cpu")
                    else:
                        raise AttributeError("sig_eval has no encode/make_signature_from_delta")
                    log_and_print(f"[SignatureCheck] Auto-generated client_sig for {client_id}", log_file=self.log_dir)
                except Exception as e:
                    log_and_print(f"[SignatureCheck] Failed to generate client_sig for {client_id}: {e}", log_file=self.log_dir)
                    client_sig = None


            # Choose ref_sig: prefer explicit arg -> stored per-client baseline -> dev fallback
            if ref_sig is None:
                stored_ref = self.ref_sigs.get(client_id)
                if stored_ref is not None:
                    ref_sig = stored_ref
                    log_and_print(f"[SignatureCheck] Using stored ref_sig for {client_id}", log_file=self.log_dir)
                else:
                    # DEV fallback (do not rely on this in production)
                    ref_sig = client_sig.clone() if client_sig is not None else None
                    if ref_sig is not None:
                        log_and_print(f"[SignatureCheck] Using fallback self-reference for {client_id} (dev only)", log_file=self.log_dir)

            # Compute similarity if both available
            if client_sig is not None and ref_sig is not None:
                sig_res = self.sig_eval.compute(client_sig=client_sig, reference_sig=ref_sig)
                if sig_res is None:
                    warnings.warn("Signature check failed.")
                    raise RuntimeError("Signature check failed.")
                results.update(sig_res)

                # Prepare candidate ref_sig update
                try:
                    S_sig = float(results.get("S_sig", 0.0))
                    sandbox_ok = results.get("sandbox_flag") == "accept"
                    if S_sig >= self.ref_update_min_Ssig and sandbox_ok and client_sig is not None:
                        # propose either initial or EMA-updated ref to be applied if client accepted by server
                        old = self.ref_sigs.get(client_id)
                        if old is None:
                            candidate = client_sig.detach().cpu().clone()
                        else:
                            alpha = float(self.ref_update_alpha)
                            candidate = (alpha * client_sig.detach().cpu()) + ((1.0 - alpha) * old.detach().cpu())
                        
                        # Stage candidate ref_sig
                        results["candidate_ref_sig"] = candidate.clone()
                        results["candidate_ref_score"] = S_sig
                        results["candidate_ref_ok"] = True
                        log_and_print(f"[SignatureCheck] Candidate ref_sig computed for {client_id} (S_sig={S_sig:.3f})", log_file=self.log_dir)
                    else:
                        results["candidate_ref_ok"] = False
                except Exception as e:
                    log_and_print(f"[SignatureCheck] Warning: failed to compute candidate ref_sig for {client_id}: {e}", log_file=self.log_dir)
                    results["candidate_ref_ok"] = False

            else:
                warnings.warn("Signature check skipped — unable to form valid signatures.")
                results.update({"L_sig": None, "S_sig": None})
        else:
            warnings.warn("Signature check failed due to missing signature eval config.")
            results.update({"L_sig": None, "S_sig": None})

        # Combine Loss & Scores
        L_anchor = results.get("L_anchor", 0.0) or 0.0
        L_KG = results.get("L_KG", 0.0) or 0.0
        L_sig = results.get("L_sig", 0.0) or 0.0
        L_check = self.alpha * L_anchor + self.beta * L_KG + self.gamma * L_sig

        self._update_Lmax(L_check)
        L_max = self._compute_Lmax()
        S_deep = 1.0 - min(L_check / (L_max + self.eps), 1.0)

        results["L_check"] = float(L_check)
        results["L_max"] = float(L_max)
        results["S_deep"] = float(S_deep)

        S_activation = results.get("S_activation", 1.0)
        if S_activation < self.activation_reject_threshold:
            results["activation_flag"] = "reject"
            log_and_print(f"[ActivationCheck] Client {client_id} rejected — S_act={S_activation:.3f}", log_file=self.log_dir)
            return {**results, "S_final": 0.0}

        S_combined = (
            (self.alpha * results.get("S_anchor", 1.0)) +
            (self.beta * results.get("S_KG", 1.0)) +
            (self.gamma * results.get("S_sig", 1.0)) +
            (self.delta * S_activation)
        ) / (self.alpha + self.beta + self.gamma + self.delta)

        log_and_print(
            f"[DeepCheck DEBUG] client={client_id} "
            f"L_sig={L_sig:.6f} L_check={L_check:.6f} L_max={L_max:.6f} "
            f"S_sig={results.get('S_sig', None)} S_activation={S_activation:.3f}",
            log_file=self.log_dir
        )

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
            log_and_print(f"[Ledger] Logged entry for client {client_id} (round {round_id})", log_file=self.log_dir)

        except Exception as e:
            log_and_print(f"[Ledger] Warning: failed to write ledger for {client_id}: {e}", log_file=self.log_dir)

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

        # Normalize all deltas format
        normalized_deltas = {}
        for cid in candidate_clients:
            normalized_deltas[cid] = self.normalize_client_delta(client_deltas.get(cid))

        precomputed_sigs = {}
        if self.sig_eval is not None:
            for cid in candidate_clients:
                # use normalized delta if possible (we'll normalize below per-client)
                raw_delta = client_deltas.get(cid)
                try:
                    norm = self.normalize_client_delta(raw_delta)
                    if hasattr(self.sig_eval, "encode"):
                        precomputed_sigs[cid] = self.sig_eval.encode(norm, dim=256, device="cpu")
                    else:
                        precomputed_sigs[cid] = self.sig_eval.make_signature_from_delta(norm, dim=256, device="cpu")
                except Exception as e:
                    log_and_print(f"[DeepCheckManager] Warning: failed to precompute sig for {cid}: {e}", log_file=self.log_dir)
                    precomputed_sigs[cid] = None

        accepted_count = 0
        rejected_count = 0
        rejected_reasons = {}

        for cid in candidate_clients:
            delta = client_deltas.get(cid)
            if delta is None:
                results[cid] = {"error": "no_delta"}
                rejected_count += 1
                continue
            
            client_delta = normalized_deltas[cid]

            # ---- Optional signatures ----
            client_sig = client_sigs.get(cid) if client_sigs else precomputed_sigs.get(cid)
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

                flags = {k: v for k, v in res.items() if isinstance(k, str) and k.endswith("_flag")}
                if any(v == "reject" for v in flags.values()):
                    rejected_count += 1
                    # pick a primary reason
                    primary = next((k for k, v in flags.items() if v == "reject"), "unknown")
                    rejected_reasons[cid] = primary
                else:
                    accepted_count += 1

            except Exception as e:
                results[cid] = {"error": str(e)}
                rejected_count += 1
                rejected_reasons[cid] = str(e)
                log_and_print(f"[DeepCheckManager] Client {cid} failed: {e}", log_file=self.log_dir)

        # === Build structured per-client decisions ===
        accepted_clients = []
        rejected_clients = []
        decisions = {}

        for cid, res in results.items():
            if cid == "_summary":
                continue
            flags = {k: v for k, v in res.items() if isinstance(k, str) and k.endswith("_flag")}
            if any(v == "reject" for v in flags.values()):
                decisions[cid] = {
                    "action": "reject",
                    "reason": next((k for k, v in flags.items() if v == "reject"), "unknown")
                }
                rejected_clients.append(cid)
            else:
                decisions[cid] = {"action": "accept"}
                accepted_clients.append(cid)

        summary = {
            "num_clients": len(candidate_clients),
            "accepted": accepted_clients,
            "rejected": rejected_clients,
            "rejected_reasons": rejected_reasons,
            "timestamp": time.time(),
        }

        log_round_summary(summary, log_dir=BASE_DIR / "logs")
        log_and_print(f"[DeepCheckManager] Batch summary: {summary}", log_file=self.log_dir)

        # Combine both detailed and summary outputs
        for cid, dec in decisions.items():
            if cid in results and isinstance(results[cid], dict):
                results[cid].update(dec)  # keep S_final etc, just add 'action'/'reason'
            else:
                results[cid] = dec        # fallback if no detailed entry yet

        results["accepted"] = accepted_clients
        results["rejected"] = rejected_clients
        results["_summary"] = summary

        return results

    def auto_calibrate_thresholds(
        self,
        global_model: torch.nn.Module,
        trusted_deltas: Dict[str, Dict[str, torch.Tensor]],
        anchor_loader=None,
        safe_calibrator=None,
        round_id: Optional[int] = None,
    ):
        """
        Safely recalibrate DeepCheck thresholds using trusted client deltas.
        Should be called only periodically (e.g., every 10 rounds).

        Parameters
        ----------
        global_model : torch.nn.Module
            Current global model.
        trusted_deltas : dict
            Mapping of trusted client_id -> delta (dict of tensors).
        anchor_loader : optional
            DataLoader for anchor-based checks.
        safe_calibrator : SafeThresholdCalibrator
            A safe calibrator instance. If None, a default one will be created.
        round_id : int, optional
            Current round id (for ledger logging).

        Returns
        -------
        dict : New thresholds or empty if skipped.
        """

        if safe_calibrator is None:
            safe_calibrator = SafeThresholdCalibrator(self)

        log_and_print(f"[DeepCheckManager] Auto-calibration triggered — {len(trusted_deltas)} trusted deltas.", log_file=self.log_dir)

        result = safe_calibrator.calibrate(
            global_model=global_model,
            benign_deltas=trusted_deltas,
            anchor_loader=anchor_loader,
        )

        if not result:
            log_and_print("[DeepCheckManager] Calibration skipped — insufficient separation or samples.", log_file=self.log_dir)
            return {}

        # Log calibration results
        try:
            log_entry = {
                "round_id": round_id or int(time.time()),
                "timestamp": time.time(),
                "new_activation_reject_threshold": result["activation_reject_threshold"],
                "new_anchor_drop_tolerance": result["anchor_drop_tolerance"],
                "sep": result.get("sep", None),
            }
            ledger_file = DL_DIR / "threshold_calibration_log.json"
            existing = []
            if ledger_file.exists():
                with open(ledger_file, "r") as f:
                    existing = json.load(f)
            existing.append(log_entry)
            with open(ledger_file, "w") as f:
                json.dump(existing, f, indent=2)
            log_and_print(f"[DeepCheckManager] Logged calibration event (round={round_id}).", log_file=self.log_dir)
        except Exception as e:
            log_and_print(f"[DeepCheckManager] Failed to log calibration: {e}", log_file=self.log_dir)

        return result

    def run_deep_checks(self, *, global_model, client_delta, client_sig=None,
                    ref_sig=None, anchor_loader=None, client_id=None, round_id=None) -> Dict[str,Any]:
        """
        Canonical single-entry call for per-update deterministic scoring.
        Returns dict with standardized fields.
        """
        res = self.compute(
            global_model=global_model,
            client_delta=client_delta,
            client_sig=client_sig,
            ref_sig=ref_sig,
            anchor_loader=anchor_loader,
            client_id=client_id,
        )
        # normalize keys / fill defaults
        out = {
            "client_id": client_id,
            "round_id": round_id or res.get("round_id") or int(time.time()),
            "S_anchor": float(res.get("S_anchor") or 1.0),
            "S_KG": None if res.get("S_KG") is None else float(res.get("S_KG")),
            "S_sig": None if res.get("S_sig") is None else float(res.get("S_sig")),
            "S_activation": None if res.get("S_activation") is None else float(res.get("S_activation")),
            "S_deep": float(res.get("S_deep", 1.0)),
            "S_final": float(res.get("S_final", 1.0)),
            "L_check": float(res.get("L_check", 0.0)),
            "flags": {k:v for k,v in res.items() if k.endswith("_flag")}
        }
        out.update(res)
        return out

    def _ref_sigs_equal_on_disk(self, path: str) -> bool:
        """
        Return True if on-disk ref_sigs are equivalent to self.ref_sigs.
        Comparison: same keys and torch.equal for tensors.
        """
        try:
            if not os.path.exists(path):
                return False
            disk = torch.load(path, map_location="cpu")
            if not isinstance(disk, dict):
                return False
            if set(disk.keys()) != set(self.ref_sigs.keys()):
                return False
            for k in self.ref_sigs.keys():
                a = self.ref_sigs[k].detach().cpu()
                b = disk[k].detach().cpu()
                if not torch.equal(a, b):
                    return False
            return True
        except Exception:
            # if any failure, assume not equal to be safe (so we'll overwrite)
            return False

    def save_ref_sigs(self, path: str):
        try:
            # If nothing changed on-disk, skip saving
            if os.path.exists(path) and self._ref_sigs_equal_on_disk(path):
                log_and_print(f"[DeepCheckManager] ref_sigs unchanged on disk -> skipping save ({path})", log_file=self.log_dir)
                return True

            dirpath = os.path.dirname(path)
            os.makedirs(dirpath, exist_ok=True)

            tmp = None
            try:
                tmpf = tempfile.NamedTemporaryFile(delete=False, dir=dirpath, suffix=".pt")
                tmp = tmpf.name
                tmpf.close()  # close before torch.save
                torch.save(self.ref_sigs, tmp)
                os.replace(tmp, path)  # atomic replace
                log_and_print(f"[DeepCheckManager] Saved ref_sigs -> {path} (atomic)", log_file=self.log_dir)
                self._ref_sigs_dirty = False
                return True
            finally:
                if tmp is not None and os.path.exists(tmp):
                    try:
                        os.remove(tmp)
                    except Exception:
                        pass
        except Exception as e:
            log_and_print(f"[DeepCheckManager] Failed to save ref_sigs: {e}", log_file=self.log_dir)
            return False

    def load_ref_sigs(self, path: str):
        import os
        if not os.path.exists(path):
            log_and_print(f"[DeepCheckManager] ref_sigs file not found: {path}", log_file=self.log_dir)
            return False
        try:
            data = torch.load(path, map_location="cpu")
            if isinstance(data, dict):
                self.ref_sigs = data
                self._ref_sigs_dirty = False
                log_and_print(f"[DeepCheckManager] Loaded ref_sigs from {path}", log_file=self.log_dir)
                return True
        except Exception as e:
            log_and_print(f"[DeepCheckManager] Failed to load ref_sigs: {e}", log_file=self.log_dir)
        return False

    def commit_ref_sigs(self, accepted_candidates: Dict[str, torch.Tensor], path: Optional[str]=None):
        """
        Commit staged candidate ref_sigs after the server has decided which clients to trust.
        - accepted_candidates: mapping client_id -> tensor (the candidate ref sig to commit)
        - path: optional path to save (defaults to DL_DIR/"ref_sigs.pt")
        This will update self.ref_sigs in-memory and then call save_ref_sigs atomically, but only if mutated.
        """
        if not accepted_candidates:
            log_and_print("[DeepCheckManager] No candidate ref_sigs to commit.", log_file=self.log_dir)
            return False

        dst = path or str(DL_DIR / "ref_sigs.pt")
        mutated = False
        for cid, cand in accepted_candidates.items():
            # only write if absent or different
            old = self.ref_sigs.get(cid)
            try:
                cand_cpu = cand.detach().cpu().clone()
            except Exception:
                cand_cpu = cand.clone().cpu()
            if old is None or not torch.equal(old.detach().cpu(), cand_cpu):
                self.ref_sigs[cid] = cand_cpu
                mutated = True

        if mutated:
            self._ref_sigs_dirty = True
            saved = self.save_ref_sigs(dst)
            if not saved:
                log_and_print("[DeepCheckManager] Warning: commit_ref_sigs failed to save to disk.", log_file=self.log_dir)
            return saved
        else:
            log_and_print("[DeepCheckManager] No changes to ref_sigs (commit skipped).", log_file=self.log_dir)
            return True