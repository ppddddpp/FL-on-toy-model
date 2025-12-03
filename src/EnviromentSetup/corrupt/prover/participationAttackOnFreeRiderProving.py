import os
import json
import hashlib
import numpy as np
import pandas as pd
from collections import Counter
from typing import Dict, Any, Optional, Tuple, List


class MaliciousContributionsOnFreeRiderProving:
    """
    Prover for Free-Rider style attacks (gradient-level and data-level).
    Works with:
        - gradient updates (dict of numpy arrays)
        - datasets (pandas.DataFrame or python list)
        - metadata dicts (e.g., {'num_samples': N})

    Produces:
        - report dict (returned by run_all)
        - writes JSON/CSV outputs to output_dir/probe_name/
    """

    def __init__(
        self,
        probe_name: str = "freerider_test",
        output_dir: str = "mc_proof_outputs"
    ):
        self.probe_name = probe_name
        self.output_dir = os.path.join(output_dir, probe_name)
        os.makedirs(self.output_dir, exist_ok=True)

    def _save_path(self, name: str) -> str:
        return os.path.join(self.output_dir, name)
    
    def _flatten_update(self, update: Dict[str, np.ndarray]) -> np.ndarray:
        """Concatenate all param arrays into a single 1D vector."""
        if update is None:
            return np.array([])
        arrs = []
        for k, v in update.items():
            a = np.asarray(v).flatten()
            if a.size:
                arrs.append(a)
        if not arrs:
            return np.array([])
        return np.concatenate(arrs)

    def _row_hash(self, row: pd.Series) -> str:
        """Stable hash for a dataframe row (used for duplicate detection)."""
        # convert row to bytes string deterministically
        s = "|".join([str(x) for x in row.tolist()]).encode("utf-8")
        return hashlib.md5(s).hexdigest()

    def _list_sample_hash(self, sample) -> str:
        """Hash for non-DataFrame sample (e.g., tuple/list)."""
        s = str(sample).encode("utf-8")
        return hashlib.md5(s).hexdigest()

    def audit_gradient_basic(self, benign_update: Dict[str, np.ndarray], malicious_update: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Basic checks for gradient-level free-rider behavior.
        - norms (L2, Linf)
        - ratio of norms
        - near-zero detection
        - cosine similarity to benign
        - top-k energy contribution
        """
        out = {}
        b = self._flatten_update(benign_update)
        m = self._flatten_update(malicious_update)

        out["len_b"] = int(b.size)
        out["len_m"] = int(m.size)

        # norms
        l2_b = float(np.linalg.norm(b)) if b.size else 0.0
        l2_m = float(np.linalg.norm(m)) if m.size else 0.0
        linf_diff = float(np.max(np.abs(m - b))) if b.size and m.size else float(np.max(np.abs(m))) if m.size else 0.0
        out["l2_norm_benign"] = l2_b
        out["l2_norm_malicious"] = l2_m
        out["linf_max_abs_diff"] = linf_diff
        out["norm_ratio_m_to_b"] = float(l2_m / (l2_b + 1e-12))

        # near zero
        out["is_zero_like_malicious"] = bool(l2_m < 1e-9)

        # tiny scaled (free-rider weak)
        out["is_tiny_scaled"] = bool(l2_m < max(1e-6, 1e-3 * (l2_b + 1e-12)))

        # cosine similarity
        if b.size and m.size and np.linalg.norm(b) > 0 and np.linalg.norm(m) > 0:
            cos = float(np.dot(b, m) / (np.linalg.norm(b) * np.linalg.norm(m)))
        else:
            cos = 0.0
        out["cosine_similarity_with_benign"] = cos

        # top-k energy fraction (how much of malicious energy lies in top-k indices of benign)
        try:
            k = min(1000, b.size) if b.size else 0
            if k > 0:
                top_idx = np.argsort(-np.abs(b))[:k]
                energy_top_b = np.sum(np.abs(b[top_idx]))
                energy_top_m = np.sum(np.abs(m[top_idx]))
                out["energy_topk_fraction_benign"] = float(energy_top_b / (np.sum(np.abs(b)) + 1e-12))
                out["energy_topk_fraction_malicious_on_b_topk"] = float(energy_top_m / (np.sum(np.abs(m)) + 1e-12))
            else:
                out["energy_topk_fraction_benign"] = 0.0
                out["energy_topk_fraction_malicious_on_b_topk"] = 0.0
        except Exception:
            out["energy_topk_fraction_benign"] = 0.0
            out["energy_topk_fraction_malicious_on_b_topk"] = 0.0

        return out

    def detect_cached_gradient(self, history_hashes: List[str], malicious_update: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Detect whether the malicious update matches a previously seen update (cached free rider).
        history_hashes: list of hex hashes of previously seen flattened updates (strings)
        """
        out = {"cached_match": False, "matched_index": None}
        m = self._flatten_update(malicious_update)
        if m.size == 0 or not history_hashes:
            return out
        # create hash of current malicious update
        cur_hash = hashlib.md5(m.tobytes()).hexdigest()
        if cur_hash in history_hashes:
            out["cached_match"] = True
            out["matched_index"] = history_hashes.index(cur_hash)
        return out

    def gradient_vs_claimed_samples(self, malicious_update: Dict[str, np.ndarray], metadata: Dict[str, Any], expected_scale_per_sample: float = 1.0) -> Dict[str, Any]:
        """
        Rough check: if client claims many samples but gradient norm is tiny,
        likely a free-rider (they lied about data size).
        expected_scale_per_sample: expected L2 norm contribution per sample (heuristic)
        """
        out = {}
        l2_m = float(np.linalg.norm(self._flatten_update(malicious_update)))
        claimed = int(metadata.get("num_samples", 0))
        out["claimed_num_samples"] = claimed
        out["l2_norm_malicious"] = l2_m
        # expected total norm ~ expected_scale_per_sample * sqrt(claimed) (very rough)
        expected_norm = expected_scale_per_sample * np.sqrt(max(1, claimed))
        out["expected_norm_estimate"] = float(expected_norm)
        out["norm_vs_expected_ratio"] = float(l2_m / (expected_norm + 1e-12))
        out["is_under_contributing"] = bool(l2_m < 0.01 * (expected_norm + 1e-12))
        return out

    def audit_dataset_basic(self, dataset, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Basic dataset-level checks:
            - actual_num_samples vs metadata['num_samples']
            - duplicate ratio (exact)
            - unique rows count
            - sample hashes distribution
            - if numerical columns exist: per-column std/mean
        """
        out = {}
        md = metadata.copy() if metadata else {}
        # determine length
        if dataset is None:
            out["actual_num_samples"] = 0
            out["claimed_num_samples"] = md.get("num_samples", None)
            return out

        # Support DataFrame, list, numpy array
        if hasattr(dataset, "shape") and len(getattr(dataset, "shape")) >= 1:
            try:
                actual_len = int(dataset.shape[0])
            except Exception:
                actual_len = len(dataset)
        else:
            actual_len = len(dataset)

        out["actual_num_samples"] = int(actual_len)
        out["claimed_num_samples"] = int(md.get("num_samples", actual_len))

        # If DataFrame: duplicate detection via row hashing
        try:
            if isinstance(dataset, pd.DataFrame):
                # compute hash per row
                row_hashes = dataset.apply(self._row_hash, axis=1).tolist()
                counts = Counter(row_hashes)
                max_repeat = max(counts.values()) if counts else 0
                unique_rows = len(counts)
                out["duplicate_ratio"] = float(1.0 - unique_rows / (actual_len + 1e-12))
                out["max_repeat"] = int(max_repeat)
                out["unique_rows"] = int(unique_rows)

                # save top repeated rows (sample)
                top = counts.most_common(10)
                out["top_repeated_hashes"] = top
                # optionally save a CSV of repeated rows for inspection
                repeated_info = [{"hash": h, "count": c} for h, c in top]
                try:
                    pd.DataFrame(repeated_info).to_csv(self._save_path("dataset_repeated_hashes.csv"), index=False)
                except Exception:
                    pass

                # basic numeric stats
                numeric_cols = dataset.select_dtypes(include=[np.number]).columns.tolist()
                col_stats = {}
                for c in numeric_cols:
                    col_stats[c] = {
                        "mean": float(dataset[c].mean()),
                        "std": float(dataset[c].std()),
                        "min": float(dataset[c].min()),
                        "max": float(dataset[c].max())
                    }
                out["numeric_column_stats"] = col_stats

            else:
                # list-like
                # compute sample hashes
                hashes = [self._list_sample_hash(s) for s in dataset]
                counts = Counter(hashes)
                unique = len(counts)
                out["duplicate_ratio"] = float(1.0 - unique / (actual_len + 1e-12))
                out["max_repeat"] = int(max(counts.values()) if counts else 0)
                out["unique_rows"] = int(unique)
                # save repeated samples summary
                top = counts.most_common(10)
                try:
                    pd.DataFrame([{"hash": h, "count": c} for h, c in top]).to_csv(self._save_path("dataset_repeated_hashes.csv"), index=False)
                except Exception:
                    pass

        except Exception as e:
            out["dataset_audit_error"] = str(e)

        return out

    def detect_tiny_or_duplicate(self, dataset, tiny_fraction_threshold: float = 0.1, duplicate_factor_threshold: int = 10) -> Dict[str, Any]:
        """
        Heuristics for:
            - tiny dataset: len(dataset) << expected per-client size
            - duplicate factor: small unique_rows but large actual_num_samples
        """
        out = {}
        if dataset is None:
            out["is_tiny"] = True
            out["tiny_fraction"] = 0.0
            return out

        # length
        if hasattr(dataset, "shape") and len(getattr(dataset, "shape")) >= 1:
            n = int(dataset.shape[0])
        else:
            n = len(dataset)

        out["actual_len"] = n

        # unique rows (reuse audit_dataset_basic's result but recompute lightweight)
        try:
            if isinstance(dataset, pd.DataFrame):
                unique_rows = dataset.drop_duplicates().shape[0]
            else:
                # list-like
                unique_rows = len({self._list_sample_hash(s) for s in dataset})
        except Exception:
            unique_rows = n

        out["unique_rows"] = int(unique_rows)
        out["tiny_fraction"] = float(unique_rows / (n + 1e-12))

        out["is_tiny"] = bool((n > 0) and (unique_rows / (n + 1e-12) < tiny_fraction_threshold))
        out["duplicate_factor"] = int(n // max(1, unique_rows))
        out["is_duplicate_heavily"] = bool(out["duplicate_factor"] >= duplicate_factor_threshold)

        return out

    def detect_random_noise_data(self, dataset, random_noise_dim_threshold: Optional[int] = None) -> Dict[str, Any]:
        """
        Heuristic to detect random/noise dataset:
        - measure per-column entropy for categorical/text columns
        - measure numerical column std: pure noise often has very high variance
        - compute sample-level string-length distribution (for text data)
        """
        out = {}
        if dataset is None or (isinstance(dataset, (list, tuple)) and len(dataset) == 0):
            out["is_noise_like"] = True
            return out

        if isinstance(dataset, pd.DataFrame):
            # categorical / text columns
            cols = dataset.columns.tolist()
            cat_stats = {}
            for c in cols:
                try:
                    vals = dataset[c].astype(str)
                    uniq = vals.nunique(dropna=True)
                    top_freq = vals.value_counts(dropna=True).iloc[0] if uniq > 0 else 0
                    cat_stats[c] = {"unique": int(uniq), "top_freq": int(top_freq)}
                except Exception:
                    cat_stats[c] = {"unique": None, "top_freq": None}
            out["column_uniqueness"] = cat_stats

            # numeric columns: overall std
            try:
                numerics = dataset.select_dtypes(include=[np.number])
                if not numerics.empty:
                    out["numeric_overall_std_mean"] = float(numerics.std().mean())
                else:
                    out["numeric_overall_std_mean"] = None
            except Exception:
                out["numeric_overall_std_mean"] = None

            # simple heuristic
            # if many columns have very high unique values and top_freq == 1 -> likely random/noisy
            noisy_cols = sum(1 for v in cat_stats.values() if v["unique"] is not None and v["top_freq"] <= 1)
            out["noisy_textual_cols"] = int(noisy_cols)
            out["is_noise_like"] = bool(noisy_cols >= max(1, len(cols) // 2))

        else:
            # list-like: compute string uniqueness heuristics
            try:
                stringified = [str(x) for x in dataset]
                uniq = len(set(stringified))
                out["unique_ratio"] = float(uniq / len(stringified))
                out["is_noise_like"] = out["unique_ratio"] > 0.95 and len(stringified) > 50
            except Exception:
                out["is_noise_like"] = False

        return out

    def prove_gradient_free_rider(self,
                                    benign_update: Dict[str, np.ndarray],
                                    malicious_update: Dict[str, np.ndarray],
                                    metadata: Optional[Dict[str, Any]] = None,
                                    history_hashes: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run gradient-level checks and save outputs.
        """
        md = metadata.copy() if metadata else {}
        report = {}
        report["basic"] = self.audit_gradient_basic(benign_update, malicious_update)
        report["cached_check"] = self.detect_cached_gradient(history_hashes or [], malicious_update)
        report["claim_vs_norm"] = self.gradient_vs_claimed_samples(malicious_update, md)
        # Save small CSV for quick inspection
        try:
            pd.DataFrame([report["basic"]]).to_csv(self._save_path("gradient_basic.csv"), index=False)
            pd.DataFrame([report["claim_vs_norm"]]).to_csv(self._save_path("gradient_claim_vs_norm.csv"), index=False)
        except Exception:
            pass
        return report

    def prove_data_free_rider(self,
                                dataset,
                                metadata: Optional[Dict[str, Any]] = None,
                                tiny_fraction_threshold: float = 0.1,
                                duplicate_factor_threshold: int = 10) -> Dict[str, Any]:
        """
        Run data-level checks and save outputs.
        """
        md = metadata.copy() if metadata else {}
        report = {}
        report["basic_audit"] = self.audit_dataset_basic(dataset, md)
        report["tiny_or_duplicate"] = self.detect_tiny_or_duplicate(dataset, tiny_fraction_threshold, duplicate_factor_threshold)
        report["noise_check"] = self.detect_random_noise_data(dataset)

        # Save dataset sample (if DataFrame) for manual inspection
        try:
            if isinstance(dataset, pd.DataFrame):
                sample_path = self._save_path("dataset_sample_head.csv")
                dataset.head(50).to_csv(sample_path, index=False)
        except Exception:
            pass

        # Save report pieces
        try:
            pd.DataFrame([report["basic_audit"]]).to_csv(self._save_path("data_basic_audit.csv"), index=False)
            pd.DataFrame([report["tiny_or_duplicate"]]).to_csv(self._save_path("data_tiny_duplicate.csv"), index=False)
            pd.DataFrame([report["noise_check"]]).to_csv(self._save_path("data_noise_check.csv"), index=False)
        except Exception:
            pass

        return report

    def run_all(self,
                benign_update: Optional[Dict[str, np.ndarray]] = None,
                malicious_update: Optional[Dict[str, np.ndarray]] = None,
                dataset = None,
                metadata: Optional[Dict[str, Any]] = None,
                history_hashes: Optional[List[str]] = None,
                tiny_fraction_threshold: float = 0.1,
                duplicate_factor_threshold: int = 10
                ) -> Dict[str, Any]:
        """
        Run all relevant checks depending on inputs provided.
        Writes a JSON report and CSV summaries into the safe output directory.
        """
        report = {"probe_name": self.probe_name}
        try:
            if benign_update is not None and malicious_update is not None:
                report["gradient_proof"] = self.prove_gradient_free_rider(
                    benign_update, malicious_update, metadata=metadata, history_hashes=history_hashes
                )
            else:
                report["gradient_proof"] = None
        except Exception as e:
            report["gradient_proof_error"] = str(e)

        try:
            if dataset is not None:
                report["data_proof"] = self.prove_data_free_rider(
                    dataset, metadata=metadata,
                    tiny_fraction_threshold=tiny_fraction_threshold,
                    duplicate_factor_threshold=duplicate_factor_threshold
                )
            else:
                report["data_proof"] = None
        except Exception as e:
            report["data_proof_error"] = str(e)

        # Save full JSON
        try:
            with open(self._save_path(f"{self.probe_name}_full_report.json"), "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

        # Save short CSV summary
        try:
            summary = {}
            if report.get("gradient_proof"):
                g = report["gradient_proof"]["basic"]
                summary.update({
                    "l2_norm_benign": g.get("l2_norm_benign"),
                    "l2_norm_malicious": g.get("l2_norm_malicious"),
                    "cosine_with_benign": g.get("cosine_similarity_with_benign")
                })
            if report.get("data_proof"):
                d = report["data_proof"]["basic_audit"]
                summary.update({
                    "data_actual_num": d.get("actual_num_samples"),
                    "data_claimed_num": d.get("claimed_num_samples"),
                    "data_duplicate_ratio": d.get("duplicate_ratio")
                })
            pd.DataFrame([summary]).to_csv(self._save_path(f"{self.probe_name}_summary.csv"), index=False)
        except Exception:
            pass

        return report
