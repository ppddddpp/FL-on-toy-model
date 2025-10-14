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
        self.deep_check = deep_check or DeepCheckManager()

        # ---- DeepCheck scheduling parameters ----
        self.deepcheck_base_prob = 0.1 if kwargs.get("deepcheck_base_prob") is None else kwargs["deepcheck_base_prob"]  # base probability (10%)
        self.deepcheck_max_prob = 0.5  if kwargs.get("deepcheck_max_prob") is None else kwargs["deepcheck_max_prob"]    # cap at 50%
        self.min_random_clients = 1 if kwargs.get("min_random_clients") is None else kwargs["min_random_clients"]
        self.last_anomaly_rate = 0.0 if kwargs.get("last_anomaly_rate") is None else kwargs["last_anomaly_rate"]
        self.calm_threshold = 0.05  # 5% anomalies

        self.global_model = global_model
        self.anchor_loader = anchor_loader

    def should_skip_deepcheck(self, anomaly_rate: float) -> bool:
        # Return True sometimes when system is calm
        if anomaly_rate < self.calm_threshold and random.random() < 0.3:
            return True
        return False

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
        fake_client_updates: Dict[str, Any],
        round_id: int = 1,
        *,
        global_model: Optional[torch.nn.Module] = None,
        anchor_loader: Optional[Any] = None,
        client_sigs: Optional[Dict[str, torch.Tensor]] = None,
        ref_sigs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, Any]:
        # ===============================================================
        # Stage 1: Lightweight checks + triage
        # ===============================================================
        all_updates = {
            cid: torch.as_tensor(update, dtype=torch.float32)
            for cid, update in fake_client_updates.items()
        }
        all_norms = [torch.norm(u).item() for u in all_updates.values()]

        norm_batch = self.norm_check.compute_batch(all_norms)
        cos_batch = self.cos_check.compute_batch(list(all_updates.values()))
        sig_batch = self.sig_check.compute_batch(list(all_updates.values()))
        chal_batch = self.chal_check.compute_batch(list(all_updates.values()))
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

        result = self.triage.step(features, round_id)

        # ===============================================================
        # Stage 2–5: Extended checks
        # ===============================================================
        client_deltas = {cid: u.cpu().numpy().reshape(-1) for cid, u in all_updates.items()}

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

        if global_model is None and self.global_model is None:
            raise ValueError("[DeepCheckManager] global_model must be provided for sandbox validation.")

        if deepcheck_clients:
            deep_results = self.deep_check.run_batch(
                global_model = global_model or self.global_model,
                client_deltas=client_deltas,
                candidate_clients=deepcheck_clients,
                client_sigs=client_sigs,
                ref_sigs=ref_sigs,
                anchor_loader = anchor_loader or self.anchor_loader
            )
        else:
            deep_results = {}

        # ---- Collect all outputs ----
        result.update({
            "stage1_subset": stats_s1,
            "stage2_similarity": stats_s2,
            "stage3_cluster": stats_s3,
            "stage4_deepcheck": deep_results,
            "escalated_clients": deep_candidates
        })

        print(f"[Round {round_id}] Completed all checks | DeepCheck clients={len(deepcheck_clients)} | Anomaly rate={self.last_anomaly_rate:.2%}")

        return result
