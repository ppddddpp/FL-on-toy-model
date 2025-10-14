from typing import Dict


def update_reputation(old_R: float, S_trust: float, lambda_decay: float = 0.3) -> float:
    """
    Exponential smoothing update for reputation R_i^(t):
        R_i^(t) = (1 - lambda) * R_i^(t-1) + lambda * S_trust(C_i)
    """
    return (1 - lambda_decay) * old_R + lambda_decay * S_trust


def fuse_trust_score(scores: Dict[str, float], weights: Dict[str, float] = None) -> float:
    """
    Fuse multiple verification scores into a single S_trust.

    Inputs:
        scores: e.g., {"S_anomaly": 0.8, "S_deep": 0.9, "S_KG": 0.95}
        weights: optional custom weighting dict (defaults uniform)

    Returns:
        Weighted average trust score in [0, 1].
    """
    if not scores:
        return 0.5

    if weights is None:
        weights = {k: 1.0 for k in scores}

    total_weight = sum(weights.values())
    if total_weight <= 0:
        total_weight = 1.0

    weighted_sum = sum(scores[k] * weights.get(k, 1.0) for k in scores)
    return float(weighted_sum / total_weight)


def classify_client(S_trust: float, threshold: float = 0.4) -> str:
    """
    Decision rule based on S_trust threshold.
    Returns one of {"accept", "down-weight", "reject"}.
    """
    if S_trust >= 0.7:
        return "accept"
    elif S_trust >= threshold:
        return "down-weight"
    else:
        return "reject"
