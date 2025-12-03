import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import entropy

class MaliciousContributionsOnGradientProving:
    """
    Prover for verifying malicious contribution generators on gradient-level attacks.

    Detects + proves poisoning via:
        - sign changes
        - L2 / Linf norm changes
        - cosine similarity drift
        - noise magnitude
        - per-layer pattern injection
        - gradient distribution shift
        - PCA outlier scores
        - gradient sparsity shift
        - KL divergence between gradient histograms
        - top-k amplification patterns
    """

    def __init__(self, benign_update, malicious_update):
        self.benign = benign_update
        self.malicious = malicious_update

    def l2_norms(self):
        def _norm(d):
            return np.sqrt(sum(np.sum(v**2) for v in d.values()))
        return {
            "benign_norm": _norm(self.benign),
            "malicious_norm": _norm(self.malicious)
        }

    def cosine_similarity(self):
        b = np.concatenate([v.flatten() for v in self.benign.values()])
        m = np.concatenate([v.flatten() for v in self.malicious.values()])
        dot = np.dot(b, m)
        denom = np.linalg.norm(b) * np.linalg.norm(m)
        if denom == 0:
            return 0.0
        return dot / denom

    def sign_flip_rate(self):
        b = np.concatenate([v.flatten() for v in self.benign.values()])
        m = np.concatenate([v.flatten() for v in self.malicious.values()])
        flips = np.sum(np.sign(b) != np.sign(m))
        return flips / len(b)

    def noise_magnitude(self):
        diff = np.concatenate([
            (self.malicious[k] - self.benign[k]).flatten()
            for k in self.benign
        ])
        return np.linalg.norm(diff)

    def layer_wise_diff(self):
        return {k: self.malicious[k] - v for k, v in self.benign.items()}

    def linf_norm_change(self):
        """Max absolute difference (∞-norm)."""
        diff = np.concatenate([
            (self.malicious[k] - self.benign[k]).flatten()
            for k in self.benign
        ])
        return float(np.max(np.abs(diff)))

    def gradient_distribution_kl(self, bins=100):
        """KL divergence between histogram distributions."""
        b = np.concatenate([v.flatten() for v in self.benign.values()])
        m = np.concatenate([v.flatten() for v in self.malicious.values()])

        hist_b, _ = np.histogram(b, bins=bins, density=True)
        hist_m, _ = np.histogram(m, bins=bins, density=True)

        hist_b += 1e-12
        hist_m += 1e-12

        kl = entropy(hist_m, hist_b)
        return float(kl)

    def sparsity_shift(self):
        """Change in % of zero-gradient components."""
        b = np.concatenate([v.flatten() for v in self.benign.values()])
        m = np.concatenate([v.flatten() for v in self.malicious.values()])

        p0_b = np.mean(np.isclose(b, 0))
        p0_m = np.mean(np.isclose(m, 0))

        return {
            "zero_ratio_benign": float(p0_b),
            "zero_ratio_malicious": float(p0_m),
            "shift": float(p0_m - p0_b)
        }

    def pca_outlier_score(self):
        """Project to PCA and measure deviation of malicious from benign."""
        b = np.concatenate([v.flatten() for v in self.benign.values()])[:, None]
        m = np.concatenate([v.flatten() for v in self.malicious.values()])[:, None]

        X = np.hstack([b, m]).T  # shape 2 x N
        pca = PCA(n_components=1)
        pca.fit(X)

        benign_score = float(pca.transform([b.flatten()])[0])
        malicious_score = float(pca.transform([m.flatten()])[0])

        return {
            "benign_pca_score": benign_score,
            "malicious_pca_score": malicious_score,
            "absolute_deviation": abs(malicious_score - benign_score)
        }

    def topk_amplification(self, k=50):
        """Detect if top-k components are abnormally amplified."""
        b = np.concatenate([v.flatten() for v in self.benign.values()])
        m = np.concatenate([v.flatten() for v in self.malicious.values()])

        idx = np.argsort(-np.abs(b))[:k]

        ratio = np.abs(m[idx]) / (np.abs(b[idx]) + 1e-12)
        return {
            "top_k_ratio_mean": float(np.mean(ratio)),
            "top_k_ratio_max": float(np.max(ratio)),
            "indices": idx[:10].tolist()
        }

    def layer_pattern_score(self):
        """Detect constant-pattern injection across layers."""
        scores = {}
        for k in self.benign.keys():
            shift = self.malicious[k] - self.benign[k]
            if np.mean(np.abs(shift)) == 0:
                scores[k] = 0
                continue
            scores[k] = float(np.std(shift) / (np.mean(np.abs(shift)) + 1e-12))
        return scores

    def detect_attack(self):
        norm_info = self.l2_norms()
        cos = self.cosine_similarity()
        sign_rate = self.sign_flip_rate()
        diff_norm = self.noise_magnitude()

        attack_types = []

        if cos < -0.9:
            attack_types.append("Sign Flip")

        ratio = norm_info["malicious_norm"] / (norm_info["benign_norm"] + 1e-12)
        if ratio > 5:
            attack_types.append("Scaling Attack")

        if norm_info["malicious_norm"] < 1e-9:
            attack_types.append("Zero Gradient Attack")

        if diff_norm > 0 and 0.1 < cos < 0.95:
            attack_types.append("Random Noise Attack")

        for k in self.benign.keys():
            if not np.allclose(self.benign[k], 0):
                shift = self.malicious[k] - self.benign[k]
                if np.std(shift) < 0.1 * np.mean(np.abs(shift) + 1e-12):
                    attack_types.append("Backdoor or Collusion")
                    break

        if 0.95 < norm_info["malicious_norm"] < 1.0:
            attack_types.append("Norm-Clip Evasion")

        if not attack_types:
            return ["No detectable attack"]

        return list(set(attack_types))

    def summary(self):
        return {
            "L2_norms": self.l2_norms(),
            "Linf_norm_change": self.linf_norm_change(),
            "cosine_similarity": self.cosine_similarity(),
            "sign_flip_rate": self.sign_flip_rate(),
            "noise_magnitude": self.noise_magnitude(),
            "KL_divergence": self.gradient_distribution_kl(),
            "sparsity_shift": self.sparsity_shift(),
            "PCA_outlier": self.pca_outlier_score(),
            "TopK_amplification": self.topk_amplification(),
            "Layer_pattern_score": self.layer_pattern_score(),
            "detected_attack_type": self.detect_attack()
        }
