from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import time
import numpy as np

@dataclass
class ClientMeta:
    client_id: str
    num_samples: int
    norm: float
    topk_indices: List[int] = field(default_factory=list)
    per_class_counts: Dict[Any,int] = field(default_factory=dict)
    challenge_logits: Optional[np.ndarray] = None
    signature: Optional[str] = None
    signature_ts: float = field(default_factory=time.time)
    compressed_signature: Optional[np.ndarray] = None
    extra: Dict[str,Any] = field(default_factory=dict)

class SelfCheck(ABC):
    def __init__(self, cfg: Dict[str,Any]):
        """
        cfg: dictionary for common configuration (thresholds, dims, etc.)
        """
        self.cfg = cfg

    @abstractmethod
    def collect_meta(self, client, state_dict, num_samples) -> ClientMeta:
        """Collect the lightweight metadata from client/state. Could be run partly client-side."""
        raise NotImplementedError()

    @abstractmethod
    def compute_score(self, meta: ClientMeta, global_stats: Dict[str,Any]) -> float:
        """Return anomaly_score in [0,1] where 1 = highly anomalous"""
        raise NotImplementedError()

    def triage(self, meta: ClientMeta, global_stats: Dict[str,Any]) -> bool:
        """Return True if flagged for deep-check (anomaly_score > threshold)"""
        score = self.compute_score(meta, global_stats)
        return score > float(self.cfg.get("T_FLAG", 0.03))

    def record_ledger_entry(self, ledger_path: str, entry: Dict[str,Any]):
        """Append metadata-only ledger entry (no raw weights)"""
        # lightweight append; in production use atomic write or DB
        import json, os
        if not os.path.exists(ledger_path):
            with open(ledger_path, "w") as f:
                json.dump([], f)
        with open(ledger_path, "r+", encoding="utf-8") as f:
            ledger = json.load(f)
            ledger.append(entry)
            f.seek(0); json.dump(ledger, f, indent=2); f.truncate()