import torch
import numpy as np
import hashlib
from typing import Dict, List

def flatten_state_dict_to_tensor(state_dict: Dict[str, torch.Tensor], device: str = 'cpu') -> torch.Tensor:
    """Flatten a model's state_dict into one long tensor."""
    parts = []
    for v in state_dict.values():
        parts.append(v.view(-1).float().to(device))
    return torch.cat(parts)

def topk_weight_indices(state_dict: Dict[str, torch.Tensor], k: int = 20) -> List[int]:
    """Return indices of top-k largest (by magnitude) weights."""
    flat = flatten_state_dict_to_tensor(state_dict)
    vals = flat.abs()
    if k >= flat.numel():
        return list(range(flat.numel()))
    return torch.topk(vals, k).indices.cpu().tolist()

def compact_hash_of_tensor(t: torch.Tensor) -> str:
    """Compute a compact SHA-256 hash of tensor bytes (used for ledger traceability)."""
    b = t.detach().cpu().numpy().tobytes()
    return hashlib.sha256(b).hexdigest()

def random_projection_signature(flat: torch.Tensor, dim: int = 64, seed: int = 42) -> np.ndarray:
    """Generate a deterministic compressed signature via random projection."""
    rng = np.random.RandomState(seed)
    W = rng.normal(size=(flat.numel(), dim)).astype(np.float32)
    sig = flat.detach().cpu().numpy() @ W
    sig = sig / (np.linalg.norm(sig) + 1e-9)
    return sig
