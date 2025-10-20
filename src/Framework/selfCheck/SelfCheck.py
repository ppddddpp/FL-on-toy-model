from pathlib import Path
import sys
try:
    BASE_DIR = Path(__file__).resolve().parents[3]
except NameError:
    BASE_DIR = Path.cwd().parents[2]
sys.path.append(str(BASE_DIR))
from typing import Dict, Any, Optional
import numpy as np
import torch
import copy
import json
import random

from .s1_light_weight_check.norm_check import NormCheck
from .s1_light_weight_check.cosine_check import CosineCheck
from .s1_light_weight_check.signature_check import SignatureCheck
from .s1_light_weight_check.challenge_check import ChallengeCheck
from .s1_light_weight_check.temporal_check import TemporalCheck
from .s1_suspicion_scoring.triage_manager import TriageManager

from .s2_similarity_scan.similarity_scan import SimilarityScanDetector
from .s3_subset_aggregation.subset_aggregation import SubsetAggregationDetector
from .s4_cluster_detection.cluster_detection import ClusterDetector
from .s5_deep_check.deep_check_eval import DeepCheckManager

BASE_DIR = Path(__file__).resolve().parents[3]
SH_LOG_DIR = BASE_DIR / "scheduler_log"
if not SH_LOG_DIR.exists():
    SH_LOG_DIR.mkdir()

class SelfCheckManager:
    """
    End-to-end FL self-check orchestrator.
    Runs lightweight pre-checks, triage, and extended multi-stage checks.
    """

    def __init__(self,
                    norm_check=None,
                    cos_check=None,
                    sig_check=None,
                    chal_check=None,
                    temp_check=None,
                    triage=None,
                    subset_detector=None,
                    sim_detector=None,
                    cluster_detector=None,
                    deep_check=None,
                    global_model=None,
                    anchor_loader=None,
                    **kwargs
                    ):

        """
        Initialize the SelfCheckManager.

        Parameters
        ----------
        norm_check : Optional[ NormCheck ], default=None
            Lightweight norm check.
        cos_check : Optional[ CosineCheck ], default=None
            Lightweight cosine similarity check.
        sig_check : Optional[ SignatureCheck ], default=None
            Lightweight signature check.
        chal_check : Optional[ ChallengeCheck ], default=None
            Lightweight challenge check.
        temp_check : Optional[ TemporalCheck ], default=None
            Lightweight temporal check.
        triage : Optional[ TriageManager ], default=None
            Suspicion scoring manager.
        subset_detector : Optional[ SubsetAggregationDetector ], default=None
            Subset aggregation detector.
        sim_detector : Optional[ SimilarityScanDetector ], default=None
            Similarity scan detector.
        cluster_detector : Optional[ ClusterDetector ], default=None
            Cluster detection detector.
        deep_check : Optional[ DeepCheckManager ], default=None
            Deep check manager.
        **kwargs
            DeepCheck scheduling parameters.

        Attributes
        ----------
        deepcheck_base_prob : float
            Base probability of scheduling DeepCheck.
        deepcheck_max_prob : float
            Cap on the probability of scheduling DeepCheck.
        min_random_clients : int
            Minimum number of random clients to sample.
        last_anomaly_rate : float
            Last observed anomaly rate.

        Returns
        -------
        None
        """
        # ---- Stage 1: lightweight checks ----
        self.norm_check = norm_check or NormCheck()
        self.cos_check = cos_check or CosineCheck()
        self.sig_check = sig_check or SignatureCheck()
        self.chal_check = chal_check or ChallengeCheck()
        self.temp_check = temp_check or TemporalCheck()
        self.triage = triage or TriageManager()

        # ---- Stage 2–5: advanced detectors ----
        self.subset_detector = subset_detector or SubsetAggregationDetector()
        self.sim_detector = sim_detector or SimilarityScanDetector()
        self.cluster_detector = cluster_detector or ClusterDetector()
        self.deep_check = deep_check or DeepCheckManager(anchor_loader=anchor_loader)

        # ---- DeepCheck scheduling parameters ----
        self.deepcheck_base_prob = 0.1 if kwargs.get("deepcheck_base_prob") is None else kwargs["deepcheck_base_prob"]  # base probability (10%)
        self.deepcheck_max_prob = 0.5  if kwargs.get("deepcheck_max_prob") is None else kwargs["deepcheck_max_prob"]    # cap at 50%
        self.min_random_clients = 1 if kwargs.get("min_random_clients") is None else kwargs["min_random_clients"]
        self.last_anomaly_rate = 0.0 if kwargs.get("last_anomaly_rate") is None else kwargs["last_anomaly_rate"]
        self.calm_threshold = 0.05  # 5% anomalies

        self.global_model = global_model
        self.anchor_loader = anchor_loader

        # ---- Decision / flagging policy (configurable) ----
        # anomaly thresholds (on robust anomaly score)
        self.threshold_low = kwargs.get("threshold_low", 0.2)
        self.threshold_mid = kwargs.get("threshold_mid", 0.4)
        self.threshold_high = kwargs.get("threshold_high", 0.6)

        # trust quantization bins (privacy): allowed outputs returned to server
        # we will quantize trust to these values only (e.g. 0.0, 0.5, 1.0)
        self.trust_bins = kwargs.get("trust_bins", [0.0, 0.5, 1.0])

        # whether to return per-client numeric anomaly scores (False -> do NOT return)
        self.expose_anomaly_scores = kwargs.get("expose_anomaly_scores", False)

        # whether to return only inclusion mask (True -> returns accepted list only)
        self.return_only_mask = kwargs.get("return_only_mask", False)

    def should_skip_deepcheck(self, anomaly_rate: float) -> bool:
        # Return True sometimes when system is calm
        if anomaly_rate < self.calm_threshold and random.random() < 0.3:
            return True
        return False

    def _evaluate_anchor_accs(self, all_updates: dict, global_model: torch.nn.Module) -> list:
        """
        Compute per-client anchor accuracies by applying each client's delta
        to a copy of the global model and evaluating it on the shared anchor loader.

        Parameters
        ----------
        all_updates : Dict[str, torch.Tensor]
            Flattened client update tensors keyed by client ID.
        global_model : torch.nn.Module
            The current global model.
        
        Returns
        -------
        List[float]
            Per-client accuracies evaluated on the shared anchor set.
        """
        if global_model is None and self.global_model is None:
            raise ValueError("[ChallengeCheck] Need global_model for anchor evaluation.")
        if self.anchor_loader is None:
            raise ValueError("[ChallengeCheck] Need anchor_loader for challenge evaluation.")

        anchor_accs = []
        base_model = global_model or self.global_model

        for cid, delta in all_updates.items():
            # clone the base model
            temp_model = copy.deepcopy(base_model)

            # determine device (match global model)
            model_device = next(temp_model.parameters()).device

            # flatten weights
            flat_weights = torch.nn.utils.parameters_to_vector(temp_model.parameters()).to(model_device)

            # flatten & align delta
            delta_flat = delta.flatten().to(model_device)
            if delta_flat.numel() < flat_weights.numel():
                pad = torch.zeros(flat_weights.numel() - delta_flat.numel(), device=model_device)
                delta_flat = torch.cat([delta_flat, pad])
            elif delta_flat.numel() > flat_weights.numel():
                delta_flat = delta_flat[:flat_weights.numel()]

            # apply the delta to model weights
            new_weights = flat_weights + delta_flat

            torch.nn.utils.vector_to_parameters(new_weights, temp_model.parameters())

            # evaluate on anchor loader
            temp_model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for ids, mask, y in self.anchor_loader:
                    ids, mask, y = ids.to(model_device), mask.to(model_device), y.to(model_device)

                    # 🔧 Ensure correct shape for 1D cases
                    if ids.dim() == 1:
                        ids = ids.unsqueeze(0)
                        mask = mask.unsqueeze(0)
                        y = y.unsqueeze(0)

                    logits = temp_model(ids, attention_mask=mask)
                    preds = logits.argmax(dim=1)
                    correct += (preds == y).sum().item()
                    total += y.size(0)
            acc = correct / max(1, total)
            anchor_accs.append(acc)
            print(f"[ChallengeCheck] {cid}: anchor_acc={acc:.4f}")

        return anchor_accs

    def schedule_deepcheck(self, triage_result: Dict[str, Any], escalated_clients: list, round_id: int):
        """
        Adaptive DeepCheck scheduler:
        - Always include escalated/flagged clients
        - Randomly sample some normal clients with adaptive prob p
        - Sometimes skip DeepCheck entirely if system is calm
        """
        flags = triage_result.get("flags", {})
        suspicious = set(escalated_clients or []) | {cid for cid, f in flags.items() if f}
        all_clients = set(flags.keys())
        normal_clients = list(all_clients - suspicious)

        # compute anomaly rate
        anomaly_rate = len(suspicious) / max(1, len(all_clients))
        self.last_anomaly_rate = anomaly_rate

        # adaptive sampling probability (base + scale)
        p = min(self.deepcheck_base_prob + 0.6 * anomaly_rate, self.deepcheck_max_prob)

        # random sample count among normal clients
        random_count = max(self.min_random_clients, int(len(normal_clients) * p))
        random_clients = random.sample(normal_clients, min(random_count, len(normal_clients))) if normal_clients else []

        # optional skip when calm
        if self.should_skip_deepcheck(anomaly_rate):
            print(f"[DeepCheckScheduler] Round {round_id}: system calm (anomaly={anomaly_rate:.2%}) → skipping DeepCheck")
            return []

        selected = sorted(list(suspicious | set(random_clients)))
        print(
            f"[DeepCheckScheduler] Round {round_id}: {len(suspicious)} flagged + {len(random_clients)} random "
            f"-> total {len(selected)} (p={p:.2f}, anomaly={anomaly_rate:.2%})"
        )

        # Save scheduler metadata per round
        log_entry = {
            "round": round_id,
            "anomaly_rate": anomaly_rate,
            "sampling_prob": p,
            "num_flagged": len(suspicious),
            "num_random": len(random_clients),
            "num_selected": len(selected)
        }
        log_path = SH_LOG_DIR / "deepcheck_schedule_log.json"
        log_path.parent.mkdir(exist_ok=True)
        try:
            with open(log_path, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            print(f"[DeepCheckScheduler] Warning: Failed to log — {e}")

        return selected

    def run_round(
        self,
        client_updates: Dict[str, Any],
        round_id: int = 1,
        *,
        global_model: Optional[torch.nn.Module] = None,
        anchor_loader: Optional[Any] = None,
        client_sigs: Optional[Dict[str, torch.Tensor]] = None,
        ref_sigs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, Any]:
        
        # --- Structural consistency check ---
        base_model = global_model or self.global_model
        global_flat = torch.nn.utils.parameters_to_vector(base_model.parameters()).detach().cpu()
        expected_len = global_flat.numel()

        # Prepare unified representations
        all_updates_flat = {}
        all_updates_param_dicts = {}
        client_deltas = {}

        for cid, update in client_updates.items():
            # --- Case 1: dict of parameter tensors ---
            if isinstance(update, dict):
                param_dict = {}
                for k, v in update.items():
                    param_dict[k] = torch.tensor(v, dtype=torch.float32) if not torch.is_tensor(v) else v.detach().cpu().float()

                # flatten in consistent parameter order
                vec_parts = [param_dict.get(name, torch.zeros_like(p)).flatten() for name, p in base_model.named_parameters()]
                flat = torch.cat(vec_parts)

                all_updates_param_dicts[cid] = param_dict
                all_updates_flat[cid] = flat
                client_deltas[cid] = flat.cpu().numpy().reshape(-1)

            # --- Case 2: list of (name, tensor) pairs ---
            elif isinstance(update, (list, tuple)) and len(update) and isinstance(update[0], (list, tuple)):
                param_dict = {}
                for k, v in update:
                    param_dict[k] = torch.tensor(v, dtype=torch.float32) if not torch.is_tensor(v) else v.detach().cpu().float()

                vec_parts = [param_dict.get(name, torch.zeros_like(p)).flatten() for name, p in base_model.named_parameters()]
                flat = torch.cat(vec_parts)

                all_updates_param_dicts[cid] = param_dict
                all_updates_flat[cid] = flat
                client_deltas[cid] = flat.cpu().numpy().reshape(-1)

            # --- Case 3: flattened numeric array / tensor ---
            else:
                t = torch.as_tensor(update, dtype=torch.float32).detach().cpu().flatten()
                if t.numel() < expected_len:
                    pad = torch.zeros(expected_len - t.numel())
                    t = torch.cat([t, pad])
                elif t.numel() > expected_len:
                    t = t[:expected_len]

                all_updates_flat[cid] = t
                client_deltas[cid] = t.cpu().numpy().reshape(-1)

                # reconstruct a structured layer-wise dict (for DeepCheck)
                layer_dict = {}
                offset = 0
                for name, p in base_model.named_parameters():
                    numel = p.numel()
                    layer_dict[name] = t[offset:offset + numel].reshape(p.shape)
                    offset += numel
                all_updates_param_dicts[cid] = layer_dict  # store unflattened version here

        # Report shape mismatches
        for cid, arr in client_deltas.items():
            if arr.size != expected_len:
                print(f"[SelfCheck] STRUCTURAL mismatch: {cid} ({arr.size} vs {expected_len})")

        # ===============================================================
        # Stage 1: Lightweight checks + triage
        # ===============================================================
        all_updates = {}
        for cid, flat_t in all_updates_flat.items():
            # ensure float tensor on cpu (like previous code expected)
            all_updates[cid] = flat_t.clone().detach().float()

        all_norms = [torch.norm(u).item() for u in all_updates.values()]

        base_norm = torch.norm(global_flat).item()
        norm_batch = self.norm_check.compute_batch([n / base_norm for n in all_norms])
        cos_batch = self.cos_check.compute_batch(list(all_updates.values()))
        sig_batch = self.sig_check.compute_from_deltas(list(all_updates.values()))
        anchor_accs = self._evaluate_anchor_accs(all_updates, global_model)
        base_acc = getattr(self.chal_check, "baseline_acc", 0.5)
        chal_batch = self.chal_check.compute_batch(
            [abs(a - base_acc) for a in anchor_accs]
        )

        tem_batch = self.temp_check.compute_batch(all_updates)
        features = {}
        for i, cid in enumerate(all_updates.keys()):
            s_chal_val = chal_batch["s_chal"][i]
            s_temp_val = tem_batch["s_temp"].get(cid, 0.0)
            features[cid] = {
                "norm": float(norm_batch["s_norms"][i]),
                "cos": float(cos_batch["s_cos"][i]),
                "sig": float(sig_batch["s_sig"][i]),
                "chal": float(np.mean(s_chal_val)),
                "temp": float(np.mean(s_temp_val)),
            }

        self.temp_check.prev_round = {cid: u.clone().detach() for cid, u in all_updates.items()}
        print(f"\n[SelfCheck] --- Stage 1: Lightweight feature summary (Round {round_id}) ---")
        for cid, f in features.items():
            print(f"  {cid}: " + " | ".join(f"{k}={v:.4f}" for k,v in f.items()))

        result = self.triage.step(features, round_id)

        # ===============================================================
        # Stage 2–5: Extended checks
        # ===============================================================

        # ---- Stage 1: Subset-Aggregation ----
        flagged_s1, stats_s1 = self.subset_detector.run(client_deltas)

        # ---- Stage 2: Similarity Scan ----
        scope_ids = flagged_s1 if flagged_s1 else list(client_deltas.keys())
        flagged_s2, stats_s2 = self.sim_detector.run(client_deltas, candidate_ids=scope_ids)

        # ---- Stage 3: Cluster Detection ----
        sketches = self.sim_detector._make_sketches(client_deltas, flagged_s2)
        clusters, stats_s3 = self.cluster_detector.run(sketches, flagged_s2, round_id)

        deep_candidates = [m for c in clusters if c.get("action_reco") == "escalate" for m in c["members"]]

        # ---- Stage 4: Deep Check (Randomized Scheduling) ----
        deep_results = {}

        # Combine all prior knowledge for scheduler
        deepcheck_clients = self.schedule_deepcheck(
            triage_result=result,
            escalated_clients=deep_candidates,
            round_id=round_id
        )

        # ---- Stage 4: Deep Check (Randomized Scheduling) ----
        if global_model is None and self.global_model is None:
            raise ValueError("[DeepCheckManager] global_model must be provided for sandbox validation.")

        if deepcheck_clients:
            payload_for_deepcheck = {}

            # Directly reuse structured dicts prepared earlier
            for cid in deepcheck_clients:
                if cid not in all_updates_param_dicts:
                    print(f"[DeepCheck] Warning: missing param_dict for {cid}, skipping.")
                    continue

                # ensure everything is a float tensor
                layer_dict = {}
                for k, v in all_updates_param_dicts[cid].items():
                    layer_dict[k] = torch.as_tensor(v, dtype=torch.float32)
                payload_for_deepcheck[cid] = layer_dict

            # Run DeepCheck directly with layer-wise payload
            deep_results = self.deep_check.run_batch(
                global_model = global_model or self.global_model,
                client_deltas = payload_for_deepcheck,
                candidate_clients = deepcheck_clients,
                client_sigs = client_sigs,
                ref_sigs = ref_sigs,
                anchor_loader = anchor_loader or self.anchor_loader
            )
        else:
            deep_results = {}

        # Combine lightweight + deep results into final per-client decisions
        decisions, trust_scores, public_out = self._decide_clients(
            triage_result=result,
            deep_results=deep_results,
            client_meta=None  # you can pass validation stats here if available
        )

        # Update result dict (internal / local use only)
        result.update({
            "stage1_subset": stats_s1,
            "stage2_similarity": stats_s2,
            "stage3_cluster": stats_s3,
            "stage4_deepcheck": deep_results,
            "escalated_clients": deep_candidates,
            "decisions": decisions,
            "trust_scores": trust_scores,
            "public_out": public_out,
        })

        # Optionally log safe summary (privacy preserved)
        try:
            safe_log = {
                "round": round_id,
                "counts": {
                    "accepted": sum(1 for d in decisions.values() if d == "ACCEPT"),
                    "downweighted": sum(1 for d in decisions.values() if d == "DOWNWEIGHT"),
                    "hold": sum(1 for d in decisions.values() if d == "HOLD"),
                    "rejected": sum(1 for d in decisions.values() if d == "REJECT"),
                },
                "trust_summary": public_out.get("trust_summary", {}),
                # Include high-level anomaly and escalation stats
                "anomaly_rate": float(self.last_anomaly_rate),
                "num_flagged": sum(1 for d in decisions.values() if d == "REJECT"),
                "num_downweighted": sum(1 for d in decisions.values() if d == "DOWNWEIGHT"),
                "num_hold": sum(1 for d in decisions.values() if d == "HOLD"),
                "num_total": len(decisions),
            }

            log_path = SH_LOG_DIR / "selfcheck_public_summary.json"
            with open(log_path, "a") as f:
                f.write(json.dumps(safe_log) + "\n")

        except Exception as e:
            print(f"[SelfCheckManager] Warning: failed to log public summary — {e}")

        print(f"[Round {round_id}] Completed all checks | "
                f"DeepCheck clients={len(deepcheck_clients)} | "
                f"Anomaly rate={self.last_anomaly_rate:.2%} | "
                f"Accepted={public_out['counts'].get('accepted', 0)} | "
                f"Rejected={public_out['counts'].get('rejected', 0)}")

        # Return privacy-preserving output (server-safe)
        return public_out

    def _quantize_trust(self, raw_value: Optional[float]) -> Optional[float]:
        """Quantize trust to closest available bin. None stays None (for HOLD/deep-check)."""
        if raw_value is None:
            return None
        bins = sorted(self.trust_bins)
        # clamp and pick nearest bin
        diffs = [abs(raw_value - b) for b in bins]
        return bins[int(np.argmin(diffs))]

    def _decide_clients(self, triage_result: Dict[str, Any], deep_results: Dict[str, Any], client_meta: Optional[Dict[str, Any]] = None):
        """
        Compose a final decision for each client using:
         - triage_result (must include 'flags' and ideally 'anomaly_scores')
         - deep_results (detailed per-client deep check outcomes; may be empty)
         - client_meta (optional, e.g. reported validation loss)
        Returns:
         - decisions: dict client_id -> "ACCEPT"|"DOWNWEIGHT"|"HOLD"|"REJECT"
         - trust_scores: dict client_id -> float|None (None => HOLD / run deeper)
         - public_out: privacy-preserving structure to return to server
        """
        flags = triage_result.get("flags", {})
        if not flags:
            flags = {cid: False for cid in triage_result.get("features", {}).keys()}
        anomaly_scores = triage_result.get("anomaly_scores", {})  # may be present or empty
        client_meta = client_meta or {}

        decisions = {}
        trust_scores = {}

        # --- base decision from triage/anomaly ---
        for cid in flags.keys():
            a_score = float(anomaly_scores.get(cid, 0.0))
            # escalate if deep_results explicitly mark REJECT or ACCEPT
            deep_info = deep_results.get(cid, {}) if isinstance(deep_results, dict) else {}
            deep_action = deep_info.get("action")  # e.g., "reject","accept","downweight", or None

            if deep_action == "reject":
                dec = "REJECT"
                trust = 0.0
            elif deep_action == "accept":
                dec = "ACCEPT"
                trust = 1.0
            else:
                # default rule-based mapping (tunable)
                if a_score >= self.threshold_high:
                    dec = "REJECT"
                    trust = 0.0
                elif a_score >= self.threshold_mid:
                    dec = "HOLD"    # run deeper check / delay decision
                    trust = None
                elif a_score >= self.threshold_low:
                    dec = "DOWNWEIGHT"
                    trust = 0.5
                else:
                    dec = "ACCEPT"
                    trust = 1.0

            # incorporate simple validation-loss escalation if provided
            meta = client_meta.get(cid, {})
            client_val = meta.get("val_loss")
            global_val = meta.get("global_val_loss")
            if (client_val is not None) and (global_val is not None):
                # if client validation loss is much worse, escalate
                if client_val > global_val + 0.02:  # config tweakable
                    dec = "REJECT"
                    trust = 0.0

            decisions[cid] = dec
            trust_scores[cid] = trust

        # --- make the public (privacy-preserving) output ---
        # Option A: If return_only_mask: return accepted clients list & counts
        if self.return_only_mask:
            accepted = [cid for cid, d in decisions.items() if d == "ACCEPT"]
            downweighted = [cid for cid, d in decisions.items() if d == "DOWNWEIGHT"]
            rejected = [cid for cid, d in decisions.items() if d == "REJECT"]
            hold = [cid for cid, d in decisions.items() if d == "HOLD"]
            public_out = {
                "accepted": accepted,
                "downweighted": downweighted,
                "rejected": rejected,
                "hold": hold,
                "counts": {"accepted": len(accepted), "downweighted": len(downweighted), "rejected": len(rejected), "hold": len(hold)}
            }
        else:
            # Option B: Return quantized trust scores (no raw anomaly values),
            # and optionally aggregate-level summary (mean/std).
            q_trust = {cid: self._quantize_trust(trust_scores[cid]) for cid in trust_scores.keys()}
            trust_vals = [v for v in q_trust.values() if v is not None]
            public_out = {
                "trust_scores_quantized": q_trust,
                "trust_summary": {"mean": float(np.mean(trust_vals)) if trust_vals else None, "std": float(np.std(trust_vals)) if trust_vals else None},
                "counts": {k: sum(1 for v in q_trust.values() if v == k) for k in sorted(self.trust_bins)}
            }

        # optionally include anomaly_scores only if explicitly allowed
        if self.expose_anomaly_scores:
            public_out["anomaly_scores"] = anomaly_scores

        return decisions, trust_scores, public_out
