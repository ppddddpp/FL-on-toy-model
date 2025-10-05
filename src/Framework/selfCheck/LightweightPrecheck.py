from typing import Dict, List, Any
import numpy as np
import torch
from .SelfCheck import SelfCheck, ClientMeta
from utils.tensor_utils import (
    flatten_state_dict_to_tensor,
    topk_weight_indices,
)

class LightweightPrecheck(SelfCheck):
    def __init__(self, cfg: Dict[str,Any]):
        super().__init__(cfg)
        # weights for component composition
        self.w = cfg.get("ANOMALY_WEIGHTS", {"norm":0.2, "cos":0.3, "sig":0.3, "temp":0.2})
        self.sig_dim = int(cfg.get("SIG_DIM", 64))
        self.t_flag = float(cfg.get("T_FLAG", 0.03))
        self.z_max = float(cfg.get("Z_MAX", 3.0))
        # per-client history store (for s_temp)
        self._norm_history: Dict[str, List[float]] = {}
        # lazy projection matrix (created on first use when flat_dim is known)
        self._proj_matrix: np.ndarray = None
        self._proj_seed = int(cfg.get("SIG_SEED", 42))

    def _ensure_proj(self, flat_len: int):
        """Create (or recreate) projection matrix if not present or dimension differs."""
        if self._proj_matrix is None or self._proj_matrix.shape[0] != flat_len:
            rng = np.random.RandomState(self._proj_seed)
            self._proj_matrix = rng.normal(size=(flat_len, self.sig_dim)).astype(np.float32)

    def _compute_signature(self, flat: torch.Tensor) -> np.ndarray:
        """Use the shared projection matrix to compute compressed signature (L2 normalized)."""
        flat_np = flat.detach().cpu().numpy()
        self._ensure_proj(flat_np.size)
        sig = flat_np @ self._proj_matrix    # (SIG_DIM,)
        sig = sig / (np.linalg.norm(sig) + 1e-9)
        return sig

    def collect_meta(self, client, state_dict, num_samples) -> ClientMeta:
        """
        This method runs on the server side using the returned state_dict.
        For better privacy, you can move parts to client side (challenge execution).
        """
        flat = flatten_state_dict_to_tensor(state_dict)
        n = float(torch.norm(flat).item())
        topk = topk_weight_indices(state_dict, k=self.cfg.get("TOPK", 20))
        # try to get per-class counts if client provides them in client.metadata
        per_class = getattr(client, "metadata", {}).get("per_class_counts", {}) if client is not None else {}
        # attempt to get challenge logits: if client exposes run_challenge() we call it (preferred client-side)
        challenge_logits = None
        if hasattr(client, "run_challenge"):
            try:
                # client.run_challenge must execute on client and return logits np.array
                challenge_logits = client.run_challenge(self.cfg.get("anchor_challenge", None))
            except Exception:
                challenge_logits = None

        # compressed signature (random projection, deterministic via seed)
        sig = self._compute_signature(flat)

        meta = ClientMeta(
            client_id = getattr(client, "id", str(id(client))),
            num_samples = int(num_samples),
            norm = n,
            topk_indices = topk,
            per_class_counts = per_class,
            challenge_logits = challenge_logits,
            compressed_signature = sig,
            signature = None
        )
        # update local history
        hist = self._norm_history.setdefault(meta.client_id, [])
        hist.append(n)
        if len(hist) > self.cfg.get("HISTORY_MAX", 5):
            hist.pop(0)
        return meta

    def _compute_s_norm(self, meta: ClientMeta, norm_list: List[float]) -> float:
        if len(norm_list) == 0:
            return 0.0
        mean_n = float(np.mean(norm_list))
        std_n = float(np.std(norm_list)) + 1e-9
        z = (meta.norm - mean_n) / std_n
        return float(min(1.0, abs(z) / self.z_max))

    def _compute_s_cos(self, meta: ClientMeta, mean_flat: torch.Tensor, client_flat: torch.Tensor) -> float:
        # compute cosine between client_flat and mean_flat; map to anomaly in [0,1]
        eps = 1e-9
        denom = (client_flat.norm() * (mean_flat.norm() + eps)) + eps
        if denom == 0:
            cos = 1.0
        else:
            cos = float(torch.dot(client_flat, mean_flat).item() / denom)
        return 1.0 - ((cos + 1.0) / 2.0)

    def _compute_s_sig(self, meta: ClientMeta, all_sigs: np.ndarray) -> float:
        if meta.compressed_signature is None or all_sigs is None or len(all_sigs)==0:
            return 0.0
        mean_sig = all_sigs.mean(axis=0)
        dist = float(np.linalg.norm(meta.compressed_signature - mean_sig))
        med = float(np.median(np.linalg.norm(all_sigs - mean_sig, axis=1))) + 1e-9
        return float(min(1.0, dist / (3.0 * med)))

    def _compute_s_temp(self, meta: ClientMeta) -> float:
        hist = self._norm_history.get(meta.client_id, [])
        if len(hist) < 2:
            return 0.0
        var = float(np.var(hist))
        v_max = float(self.cfg.get("V_TEMP_MAX", 1.0))
        return float(min(1.0, var / v_max))

    def compute_score(self, meta: ClientMeta, global_stats: Dict[str,Any]) -> float:
        """
        global_stats should include:
            - 'all_flat_deltas': List[torch.Tensor] (optional)
            - 'mean_flat_delta': torch.Tensor
            - 'all_signatures': np.ndarray (N x SIG_DIM)
            - 'norms': List[float]
            - 'client_flat_map': Dict[client_id -> flat tensor]
        """
        # build components
        norms = global_stats.get("norms", [])
        s_norm = self._compute_s_norm(meta, norms)

        # need client_flat to compute cos; derive if present in global_stats mapping of client id -> flat
        client_flat = global_stats.get("client_flat_map", {}).get(meta.client_id, None)
        mean_flat = global_stats.get("mean_flat_delta", None)
        s_cos = 0.0
        if client_flat is not None and mean_flat is not None:
            s_cos = self._compute_s_cos(meta, mean_flat, client_flat)

        s_sig = 0.0
        if "all_signatures" in global_stats:
            s_sig = self._compute_s_sig(meta, global_stats["all_signatures"])

        s_temp = self._compute_s_temp(meta)

        # weighted sum
        s = 0.0
        s += self.w.get("norm", 0.0) * s_norm
        s += self.w.get("cos", 0.0)  * s_cos
        s += self.w.get("sig", 0.0)  * s_sig
        s += self.w.get("temp", 0.0) * s_temp

        # clip
        return float(max(0.0, min(1.0, s)))
