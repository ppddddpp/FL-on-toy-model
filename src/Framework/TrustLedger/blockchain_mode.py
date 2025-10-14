import json, hashlib, time, os
from typing import Dict, Any
from .trust_ledger import TrustLedger
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
BC_SAVE_DIR = BASE_DIR / "BC_save"
if not BC_SAVE_DIR.exists():
    BC_SAVE_DIR.mkdir(parents=True)

class BlockchainLedger(TrustLedger):
    """
    Blockchain-style extension of TrustLedger.

    Each entry (block) includes:
        - block_hash: SHA256 over the entry + prev_hash
        - prev_hash: previous block hash

    Guarantees immutability and verifiable audit chains.
    """

    def __init__(self, path="ledger_blockchain.json", lambda_decay=0.3):
        if path is None:
            path = os.path.join(BC_SAVE_DIR, "ledger_blockchain.json")
        super().__init__(path=path, lambda_decay=lambda_decay)

        # Restore the previous hash chain if exists
        if len(self.entries) > 0:
            self.prev_hash = self.entries[-1].get("block_hash", "")
        else:
            self.prev_hash = ""

    def _compute_block_hash(self, entry: Dict[str, Any], prev_hash: str) -> str:
        """Compute SHA-256 hash of serialized entry + previous hash."""
        data = json.dumps(entry, sort_keys=True).encode()
        return hashlib.sha256(data + prev_hash.encode()).hexdigest()

    def append_entry(self, client_id: str, round_t: int, meta: Dict[str, Any]):
        """Append a new ledger entry with blockchain chaining."""
        # Normal TrustLedger logic
        old_R = self.reputation.get(client_id, 0.5)
        S_trust = meta.get("S_trust", 0.5)
        new_R = (1 - self.lambda_decay) * old_R + self.lambda_decay * S_trust
        self.reputation[client_id] = new_R

        # Construct entry
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
            "prev_hash": self.prev_hash
        }

        # Compute new block hash
        block_hash = self._compute_block_hash(entry, self.prev_hash)
        entry["block_hash"] = block_hash
        self.prev_hash = block_hash

        # Append to ledger
        self.entries.append(entry)
        json.dump(self.entries, open(self.path, "w"), indent=2)

    def verify_chain(self) -> bool:
        """Verify that the blockchain hash chain is valid."""
        prev_hash = ""
        for entry in self.entries:
            expected = self._compute_block_hash(
                {k: v for k, v in entry.items() if k not in ("block_hash", "prev_hash")},
                prev_hash
            )
            if expected != entry.get("block_hash"):
                print(f"[ERROR] Ledger tampering detected at round {entry['round']}!")
                return False
            prev_hash = entry["block_hash"]
        print("[OK] Blockchain ledger integrity verified.")
        return True
