import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional
import numpy as np
import secrets
import hashlib

class SignatureEvaluator:
    """
    Computes similarity and consistency scores between client signatures
    and a reference signature — with optional hardening features.
    
    Supports:
        - Salted signature randomization per round
        - Projection seed rotation
        - Differential privacy (DP) noise
        - Rolling normalization for adaptive scoring
    """

    def __init__(
        self,
        loss_type: str = "cosine",
        eps: float = 1e-8,
        use_salt: bool = True,
        dp_sigma: float = 0.0,
        seed_rotation: bool = True,
        lmax_percentile: float = 95.0,
        salt_strength: float = 1e-3,
    ):
        """
        Parameters
        ----------
        loss_type : str
            "cosine" or "l2" for distance metric.
        eps : float
            Small epsilon for stability.
        use_salt : bool
            Whether to apply random per-round salt to signatures.
        dp_sigma : float
            Standard deviation of Gaussian DP noise (0 disables DP).
        seed_rotation : bool
            Whether to rotate projection seed every call.
        lmax_percentile : float
            Percentile for normalization scaling.
        salt_strength : float
            Strength of salt noise (0 disables salt).
        """
        assert loss_type in ["cosine", "l2"], "loss_type must be 'cosine' or 'l2'"
        self.loss_type = loss_type
        self.eps = eps
        self.use_salt = use_salt
        self.dp_sigma = dp_sigma
        self.seed_rotation = seed_rotation
        self.lmax_percentile = lmax_percentile

        self._rolling_L = []
        self.round_seed = secrets.randbits(32)
        self.round_salt = None
        self.round_id = 0
        self.salt_strength = salt_strength

    def set_fixed_seed(self, seed: int):
        """Force a fixed seed for deterministic auditing."""
        self.round_seed = seed
        self.seed_rotation = False
        self._rotate_seed()
        print(f"[SignatureEval] Fixed seed mode enabled (seed={seed})")

    def _rotate_seed(self):
        """Rotate the internal seed per round for projection stability."""
        if self.seed_rotation:
            self._local_torch_rng = torch.Generator().manual_seed(self.round_seed)
            torch.manual_seed(self.round_seed)
            np.random.seed(self.round_seed % (2**32 - 1))
            print(f"[SignatureEval] Rotated seed -> {self.round_seed}")

    def _generate_salt(self, round_id: Optional[int] = None) -> torch.Tensor:
        """
        Create a deterministic salt vector for the given round.
        Combines round_seed + round_id to allow reproducibility.
        """
        if round_id is None:
            round_id = self.round_id

        # combine base seed and round ID
        seed_val = (self.round_seed + round_id * 7919) & 0xFFFFFFFFFFFFFFFF  # 64-bit mix
        rng = np.random.default_rng(seed_val)
        salt = torch.tensor(rng.random(16), dtype=torch.float32)
        return salt

    def _apply_salt_and_noise(self, sig: torch.Tensor) -> torch.Tensor:
        """Apply optional salt and DP noise."""
        sig_mod = sig.clone().float()
        if self.use_salt:
            salt_vec = self._generate_salt().to(sig.device)
            # pad or tile salt to match signature length
            if sig_mod.numel() > len(salt_vec):
                salt_vec = salt_vec.repeat(int(np.ceil(sig_mod.numel() / len(salt_vec))))[: sig_mod.numel()]
            sig_mod = sig_mod + self.salt_strength * salt_vec
        if self.dp_sigma > 0:
            scale = self.dp_sigma / np.sqrt(sig_mod.numel())
            sig_mod = sig_mod + torch.randn_like(sig_mod) * scale

        return sig_mod

    def _update_Lmax(self, L_val: float):
        self._rolling_L.append(float(L_val))
        if len(self._rolling_L) > 512:
            self._rolling_L.pop(0)

    def _compute_Lmax(self) -> float:
        if not self._rolling_L:
            return 1.0
        return max(1e-6, float(np.percentile(self._rolling_L, self.lmax_percentile)))

    def make_signature_from_delta(self, delta: dict, dim: int = 256, device="cpu") -> torch.Tensor:
        parts = []
        for k in sorted(delta.keys()):
            v = delta[k]
            if isinstance(v, torch.Tensor):
                parts.append(v.detach().cpu().flatten())
            else:
                parts.append(torch.as_tensor(v).cpu().flatten())
        if len(parts) == 0:
            return torch.zeros(dim, dtype=torch.float32, device=device)
        flat = torch.cat(parts).numpy().astype(np.float32)
        h = hashlib.sha256(flat.tobytes()).digest()
        seed = int.from_bytes(h[:4], "little")
        rng = np.random.default_rng(seed)
        proj = rng.standard_normal((flat.shape[0], dim)).astype(np.float32)
        sig = flat @ proj
        sig = sig / (np.linalg.norm(sig) + 1e-12)
        return torch.from_numpy(sig).to(device)

    def encode(self, delta: dict, dim: int = 256, device: str = "cpu") -> torch.Tensor:
        """
        Unified API: produce a signature from a delta dict.
        Keeps callers independent of implementation details (deterministic vs learned).
        """
        return self.make_signature_from_delta(delta, dim=dim, device=device)

    def compute(
        self,
        client_sig: torch.Tensor,
        reference_sig: torch.Tensor,
        round_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Compute hardened signature similarity between client and reference.

        Parameters
        ----------
        client_sig : torch.Tensor
            Client signature vector.
        reference_sig : torch.Tensor
            Trusted reference signature vector.
        round_id : int, optional
            Used to rotate seeds and salts.
        """
        if round_id is not None:
            if round_id != self.round_id:
                self._rotate_seed()
                self.round_salt = self._generate_salt(round_id)
                self.round_id = round_id

        # Apply salt + DP noise
        client_sig = self._apply_salt_and_noise(client_sig.flatten())
        reference_sig = self._apply_salt_and_noise(reference_sig.flatten())

        # Compute similarity
        if self.loss_type == "cosine":
            sim = F.cosine_similarity(
                client_sig.unsqueeze(0), reference_sig.unsqueeze(0), dim=1
            ).item()
            L_sig = 1.0 - sim
        else:
            diff = client_sig - reference_sig
            L_sig = torch.norm(diff, p=2).item()
            sim = 1.0 / (1.0 + L_sig)

        self._update_Lmax(L_sig)
        L_max = self._compute_Lmax()
        S_sig = 1.0 - min(max(L_sig / (L_max + self.eps), 0.0), 1.0)
        
        # Log reproducible signature hash (for ledger or audit)
        hash_input = (client_sig + reference_sig).detach().cpu().numpy().tobytes()
        sig_hash = hashlib.sha256(hash_input).hexdigest()[:16]

        return {
            "round_id": self.round_id,
            "L_sig": float(L_sig),
            "S_sig": float(S_sig),
            "similarity": float(sim),
            "sig_hash": sig_hash,
            "salt_used": self.use_salt,
            "dp_sigma": self.dp_sigma,
            "seed": self.round_seed,
        }

# Example usage
if __name__ == "__main__":
    a = torch.randn(128)
    b = a + 0.05 * torch.randn(128)
    c = a + 0.5 * torch.randn(128)

    se = SignatureEvaluator(use_salt=True, dp_sigma=0.01)
    print("\n--- Round 1 ---")
    print("Benign:", se.compute(a, b, round_id=1))
    print("Malicious:", se.compute(a, c, round_id=1))

    print("\n--- Round 2 ---")
    print("Benign:", se.compute(a, b, round_id=2))
