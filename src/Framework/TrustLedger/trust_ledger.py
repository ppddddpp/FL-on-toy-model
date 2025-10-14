import json, hashlib, time, os
from typing import Dict, Any

class TrustLedger:
    def __init__(self, path="ledger.json", lambda_decay=0.3):
        self.path = path
        self.lambda_decay = lambda_decay
        self.reputation = {}  # R_i^(t)
        self.entries = []
        if os.path.exists(path):
            try:
                self.entries = json.load(open(path, "r"))
                for e in self.entries:
                    self.reputation[e["client_id"]] = e.get("R_t", 0.5)
            except Exception:
                self.entries = []

    def hash_update(self, delta_dict: Dict[str, Any]) -> str:
        """Compact SHA256 hash of flattened update."""
        flat = b"".join(v.detach().cpu().numpy().tobytes()
                        for v in delta_dict.values() if hasattr(v, "detach"))
        return hashlib.sha256(flat).hexdigest()[:16]

    def append_entry(self, client_id: str, round_t: int, meta: Dict[str, Any]):
        # reputation update
        old_R = self.reputation.get(client_id, 0.5)
        S_trust = meta.get("S_trust", 0.5)
        new_R = (1 - self.lambda_decay) * old_R + self.lambda_decay * S_trust
        self.reputation[client_id] = new_R

        entry = {
            "round": round_t,
            "client_id": client_id,
            "hash_delta": meta.get("hash_delta"),
            "S_anomaly": meta.get("S_anomaly"),
            "S_deep": meta.get("S_deep"),
            "S_KG": meta.get("S_KG"),
            "S_trust": S_trust,
            "decision": meta.get("decision", "accept"),
            "R_t": new_R,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.entries.append(entry)
        json.dump(self.entries, open(self.path, "w"), indent=2)

    def get_reputation(self, client_id: str) -> float:
        return self.reputation.get(client_id, 0.5)
