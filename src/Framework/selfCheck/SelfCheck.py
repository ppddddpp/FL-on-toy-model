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
from Helpers.Helpers import log_and_print

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
                    log_dir= BASE_DIR / "logs" / "run.txt",
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
        self.log_dir = log_dir if log_dir is not None else BASE_DIR / "logs" / "run.txt"

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
            log_and_print(f"[ChallengeCheck] {cid}: anchor_acc={acc:.4f}", log_file=self.log_dir)

        return anchor_accs

    def schedule_deepcheck(
        self,
        triage_result: Dict[str, Any],
        mid_stage_info: Dict[str, Any],
        escalated_clients: list,
        round_id: int,
        survivors: Optional[list] = None,
    ):
        """
        Adaptive DeepCheck scheduler (strict mode):
        - Only considers clients that survived previous stages
        - Always DeepChecks escalated or flagged clients
        - Randomly audits a subset of normal survivors
        - Never skips DeepCheck entirely
        """
        all_clients = set(survivors or [])
        if not all_clients:
            return []

        flags = triage_result.get("flags", {})
        subset_flags = set(mid_stage_info.get("flagged_s1", []))
        similar_flags = set(mid_stage_info.get("flagged_s2", []))
        cluster_flags = set(mid_stage_info.get("cluster_rejects", []))
        all_flags = subset_flags | similar_flags | cluster_flags

        # Combine suspicion sources
        suspicious = (
            set(escalated_clients or [])
            | {cid for cid, f in flags.items() if f and cid in all_clients}
            | (all_flags & all_clients)
        )
        normal_clients = list(all_clients - suspicious)

        # Adaptive random audit among normal survivors
        anomaly_rate = len(suspicious) / max(1, len(all_clients))
        self.last_anomaly_rate = anomaly_rate

        p = min(self.deepcheck_base_prob + 0.6 * anomaly_rate, self.deepcheck_max_prob)
        random_count = max(self.min_random_clients, int(len(normal_clients) * p))
        random_clients = random.sample(normal_clients, min(random_count, len(normal_clients))) if normal_clients else []

        selected = sorted(list(suspicious | set(random_clients)))

        log_and_print(
            f"[DeepCheckScheduler] Round {round_id}: {len(suspicious)} suspicious + {len(random_clients)} random "
            f"-> total={len(selected)} (p={p:.2f}, survivors={len(all_clients)}, anomaly={anomaly_rate:.2%})",
            log_file=self.log_dir
        )

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
                log_and_print(f"[SelfCheck] STRUCTURAL mismatch: {cid} ({arr.size} vs {expected_len})", log_file=self.log_dir)

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
        log_and_print(f"\n[SelfCheck] --- Stage 1: Lightweight feature summary (Round {round_id}) ---", log_file=self.log_dir)
        for cid, f in features.items():
            log_and_print(f"  {cid}: " + " | ".join(f"{k}={v:.4f}" for k,v in f.items()), log_file=self.log_dir)

        result = self.triage.step(features, round_id)

        early_reject = [cid for cid, dec in result.get("decisions", {}).items() if dec == "REJECT"]
        if early_reject:
            log_and_print(f"[SelfCheck] Early rejecting {len(early_reject)} clients: {early_reject}",
                        log_file=self.log_dir)

            remaining_after_early = [cid for cid in all_updates_param_dicts.keys() if cid not in early_reject]
            log_and_print(
                f"[SelfCheck] Remaining clients after Stage 1 (triage): "
                f"{len(remaining_after_early)}/{len(client_updates)} -> {remaining_after_early}",
                log_file=self.log_dir
            )

            # Drop them from all later stage inputs
            client_deltas = {cid: delta for cid, delta in client_deltas.items() if cid not in early_reject}
            all_updates_flat = {cid: flat for cid, flat in all_updates_flat.items() if cid not in early_reject}
            all_updates_param_dicts = {cid: pd for cid, pd in all_updates_param_dicts.items() if cid not in early_reject}

        # ===============================================================
        # Stage 2–4: Extended checks
        # ===============================================================

        # ---- Stage 2: Subset-Aggregation ----
        flagged_s1, stats_s1 = self.subset_detector.run(client_deltas)

        # ---- Stage 3: Similarity Scan ----
        scope_ids = flagged_s1 if flagged_s1 else list(client_deltas.keys())
        flagged_s2, stats_s2 = self.sim_detector.run(client_deltas, candidate_ids=scope_ids)

        # ---- Stage 4: Cluster Detection ----
        sketches = self.sim_detector._make_sketches(client_deltas, flagged_s2)
        clusters, stats_s3 = self.cluster_detector.run(sketches, flagged_s2, round_id)

        deep_candidates = [m for c in clusters if c.get("action_reco") == "escalate" for m in c["members"]]

        flagged_all = set(flagged_s1) | set(flagged_s2)
        cluster_rejects = [m for c in clusters if c.get("action_reco") == "reject" for m in c["members"]]
        mid_reject = list(flagged_all.union(cluster_rejects))

        if mid_reject:
            log_and_print(f"[SelfCheck] Mid-stage rejecting {len(mid_reject)} clients: {mid_reject}",
                        log_file=self.log_dir)

            remaining_after_mid = [cid for cid in all_updates_param_dicts.keys() if cid not in mid_reject]
            log_and_print(
                f"[SelfCheck] Remaining clients after Stage 2 - 4 (subset/similarity/cluster): "
                f"{len(remaining_after_mid)} clients -> {remaining_after_mid}",
                log_file=self.log_dir
            )

            # Drop them from further stages
            client_deltas = {cid: delta for cid, delta in client_deltas.items() if cid not in mid_reject}
            all_updates_param_dicts = {cid: pd for cid, pd in all_updates_param_dicts.items() if cid not in mid_reject}

        mid_stage_info = {
            "flagged_s1": flagged_s1,
            "flagged_s2": flagged_s2,
            "cluster_rejects": cluster_rejects,
        }

        # ===============================================================
        # Stage 5: Deep checks
        # ===============================================================

        # Survivors are clients that passed all prior filters
        survivors = [
            cid for cid in all_updates_param_dicts.keys()
            if cid not in early_reject and cid not in mid_reject
        ]

        # Combine all prior knowledge for scheduler
        deepcheck_clients = self.schedule_deepcheck(
            triage_result=result,
            mid_stage_info=mid_stage_info,
            escalated_clients=deep_candidates,
            round_id=round_id,
            survivors=survivors,
        )

        if global_model is None and self.global_model is None:
            raise ValueError("[DeepCheckManager] global_model must be provided for sandbox validation.")

        deep_results = {}
        if deepcheck_clients:
            payload_for_deepcheck = {}

            # Directly reuse structured dicts prepared earlier
            for cid in deepcheck_clients:
                if cid not in all_updates_param_dicts:
                    log_and_print(f"[DeepCheck] Warning: missing param_dict for {cid}, skipping.", log_file=self.log_dir)
                    continue

                # ensure everything is a float tensor
                layer_dict = {}
                for k, v in all_updates_param_dicts[cid].items():
                    layer_dict[k] = torch.as_tensor(v, dtype=torch.float32)
                payload_for_deepcheck[cid] = layer_dict

            # Run DeepCheck directly with layer-wise payload
            deep_results = self.deep_check.run_batch(
                global_model=global_model or self.global_model,
                client_deltas=payload_for_deepcheck,
                candidate_clients=deepcheck_clients,
                client_sigs=client_sigs,
                ref_sigs=ref_sigs,
                anchor_loader=anchor_loader or self.anchor_loader,
            )

        else:
            # Print to show case if not any passing previous check to reach deepcheck
            log_and_print(f"[DeepCheck] Warning: no clients selected for DeepCheck in Round {round_id}", log_file=self.log_dir)

        # Combine lightweight + deep results into final per-client decisions
        decisions, trust_scores, public_out = self._decide_clients(
            triage_result=result,
            mid_stage_info=mid_stage_info,
            deep_results=deep_results,
            client_meta=None
        )

        # Combine all rejection phases
        all_rejects = set(early_reject) | set(mid_reject)
        final_rejects = {cid for cid, d in decisions.items() if d == "REJECT"}
        total_rejects = list(all_rejects | final_rejects)

        # Include total accepted (those not in rejects)
        accepted_total = [
            cid for cid in decisions.keys()
            if cid not in total_rejects and decisions[cid] == "ACCEPT"
        ]

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
                    "accepted_stage_final": len(accepted_total),
                    "rejected_stage1": len(early_reject),
                    "rejected_stage23": len(mid_reject),
                    "rejected_stage4": len(final_rejects),
                    "rejected_total": len(total_rejects),
                    "downweighted": sum(1 for d in decisions.values() if d == "DOWNWEIGHT"),
                    "hold": sum(1 for d in decisions.values() if d == "HOLD"),
                },
                "trust_summary": public_out.get("trust_summary", {}),
                "anomaly_rate": float(self.last_anomaly_rate),
                "num_flagged": len(total_rejects),
                "num_total": len(decisions) + len(early_reject) + len(mid_reject),
                "reject_clients": list(total_rejects),
                "accepted_clients": accepted_total,
            }

            log_path = SH_LOG_DIR / "selfcheck_public_summary.jsonl"
            with open(log_path, "a") as f:
                f.write(json.dumps(safe_log) + "\n")

        except Exception as e:
            log_and_print(f"[SelfCheckManager] Warning: failed to log public summary — {e}", log_file=self.log_dir)

        log_and_print(f"[Round {round_id}] Completed all checks | "
                f"DeepCheck clients={len(deepcheck_clients)} | "
                f"Anomaly rate={self.last_anomaly_rate:.2%} | "
                f"Accepted={public_out['counts'].get('accepted', 0)} | "
                f"Rejected={public_out['counts'].get('rejected', 0)}",
                log_file=self.log_dir)

        total_clients = len(client_updates)
        log_and_print(
            f"[Round {round_id}] Final accept list ({len(accepted_total)}/{total_clients}): {accepted_total}",
            log_file=self.log_dir
        )
        log_and_print(
            f"[Round {round_id}] Final reject list ({len(total_rejects)}/{total_clients}): {total_rejects}",
            log_file=self.log_dir
        )

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

    def _decide_clients(
        self,
        triage_result: Dict[str, Any],
        mid_stage_info: Optional[Dict[str, Any]] = None,
        deep_results: Optional[Dict[str, Any]] = None,
        client_meta: Optional[Dict[str, Any]] = None,
    ):
        """
        Compose a final decision for each client using:
            - triage_result (must include 'flags' and ideally 'anomaly_scores')
            - mid_stage_info (subset/similarity/cluster flags)
            - deep_results (detailed per-client deep check outcomes; may be empty)
            - client_meta (optional, e.g. reported validation loss)
        Returns:
            - decisions: dict client_id -> "ACCEPT"|"DOWNWEIGHT"|"HOLD"|"REJECT"
            - trust_scores: dict client_id -> float|None (None => HOLD / run deeper)
            - public_out: privacy-preserving structure to return to server
        """

        # === Gather inputs ===
        flags = triage_result.get("flags", {})
        if not flags:
            flags = {cid: False for cid in triage_result.get("features", {}).keys()}

        anomaly_scores = triage_result.get("anomaly_scores", {})
        client_meta = client_meta or {}
        deep_results = deep_results or {}
        mid_stage_info = mid_stage_info or {}

        subset_flags = set(mid_stage_info.get("flagged_s1", []))
        similar_flags = set(mid_stage_info.get("flagged_s2", []))
        cluster_rejects = set(mid_stage_info.get("cluster_rejects", []))

        # --- Start from triage decisions if available ---
        decisions = {}
        if isinstance(triage_result, dict) and "decisions" in triage_result:
            for cid, tri_dec in triage_result["decisions"].items():
                decisions[cid] = tri_dec

        for cid in flags.keys():
            if cid not in decisions:
                decisions[cid] = None

        trust_scores = {}

        # === Normalize deep_results (handle summary or per-client) ===
        deep_accepts = set()
        deep_rejects = set()

        if isinstance(deep_results, dict):
            for cid, info in deep_results.items():
                if not isinstance(info, dict):
                    continue
                action = info.get("action")
                if action == "accept":
                    deep_accepts.add(cid)
                elif action == "reject":
                    deep_rejects.add(cid)

            if "accepted" in deep_results and isinstance(deep_results["accepted"], list):
                deep_accepts.update(deep_results["accepted"])
            if "rejected" in deep_results and isinstance(deep_results["rejected"], list):
                deep_rejects.update(deep_results["rejected"])

        # === Decision logic per client ===
        for cid in flags.keys():
            a_score = float(anomaly_scores.get(cid, 0.0))

            # --- mid-stage influence (soft penalties) ---
            if cid in subset_flags:
                a_score += 0.05
            if cid in similar_flags:
                a_score += 0.05
            if cid in cluster_rejects:
                a_score += 0.10
            a_score = min(a_score, 1.0)

            # --- Deep check override (highest priority) ---
            if cid in deep_rejects:
                dec = "REJECT"
                trust = 0.0
                deep_action = "reject"
            elif cid in deep_accepts:
                dec = "ACCEPT"
                trust = 1.0
                deep_action = "accept"
            else:
                deep_action = None
                # --- Rule-based mapping ---
                if a_score >= self.threshold_high:
                    dec = "REJECT"
                    trust = 0.0
                elif a_score >= self.threshold_mid:
                    dec = "HOLD"
                    trust = None
                elif a_score >= self.threshold_low:
                    dec = "DOWNWEIGHT"
                    trust = 0.5
                else:
                    dec = "ACCEPT"
                    trust = 1.0

            # --- Optional validation-loss escalation ---
            meta = client_meta.get(cid, {})
            client_val = meta.get("val_loss")
            global_val = meta.get("global_val_loss")
            if (client_val is not None) and (global_val is not None):
                if client_val > global_val + 0.02:
                    dec = "REJECT"
                    trust = 0.0

            # --- Penalty for multiple flags ---
            flag_count = sum([
                cid in subset_flags,
                cid in similar_flags,
                cid in cluster_rejects
            ])
            if flag_count >= 2 and dec != "REJECT":
                dec = "DOWNWEIGHT"
                trust = 0.4

            decisions[cid] = dec
            trust_scores[cid] = trust

            log_and_print(
                f"[Decision] {cid}: a_score={a_score:.3f} | "
                f"subset={cid in subset_flags} | similar={cid in similar_flags} | "
                f"cluster={cid in cluster_rejects} | deep_action={deep_action} | "
                f"final={dec} | trust={trust}",
                log_file=self.log_dir,
            )

        # --- Enforce deep check rejections once more (defensive) ---
        for cid in deep_rejects:
            decisions[cid] = "REJECT"
            trust_scores[cid] = 0.0
            log_and_print(f"[Decision][ENFORCE] {cid} forced REJECT by deep_check", log_file=self.log_dir)

        # === Build public output ===
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
                "counts": {
                    "accepted": len(accepted),
                    "downweighted": len(downweighted),
                    "rejected": len(rejected),
                    "hold": len(hold),
                },
            }
        else:
            q_trust = {cid: self._quantize_trust(trust_scores[cid]) for cid in trust_scores.keys()}
            trust_vals = [v for v in q_trust.values() if v is not None]
            public_out = {
                "trust_scores_quantized": q_trust,
                "trust_summary": {
                    "mean": float(np.mean(trust_vals)) if trust_vals else None,
                    "std": float(np.std(trust_vals)) if trust_vals else None,
                },
                "counts": {
                    "accepted": sum(1 for v in q_trust.values() if v == 1.0),
                    "downweighted": sum(1 for v in q_trust.values() if v == 0.5),
                    "rejected": sum(1 for v in q_trust.values() if v == 0.0),
                    "hold": sum(1 for v in q_trust.values() if v is None),
                },
            }

        if self.expose_anomaly_scores:
            public_out["anomaly_scores"] = anomaly_scores

        return decisions, trust_scores, public_out
