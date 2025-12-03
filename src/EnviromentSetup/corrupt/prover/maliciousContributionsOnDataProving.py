import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import json
import os
from typing import Optional, Dict, Any, List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class MaliciousContributionsGeneratorOnDataProving:
    """
    Prover / Evaluator for data-level malicious contribution generator.

    Capabilities:
        - audit_summary() : quick diff summary (text changes, label changes, new rows)
        - embedding_distance(...) : semantic distance metrics (sentence-transformer or TF-IDF)
        - distribution_shift(...) : label/text-length distribution checks
        - ner_anomaly(...) : named-entity anomaly detection (spaCy or heuristic)
        - run_all(...) : run the entire pipeline
    """

    def __init__(
        self,
        original_df: pd.DataFrame,
        corrupted_df: pd.DataFrame,
        text_col: str = "Information",
        label_col: str = "Group",
        probe_name: str = "mc_data_proof",
        output_dir: str = "mc_proof_outputs"
    ):
        # Store data
        self.orig = original_df.reset_index(drop=True).copy()
        self.corr = corrupted_df.reset_index(drop=True).copy()
        self.text_col = text_col
        self.label_col = label_col
        self.probe_name = probe_name

        # Create isolated output directory
        self.output_dir = os.path.join(output_dir, probe_name)
        os.makedirs(self.output_dir, exist_ok=True)

    def _save(self, filename: str) -> str:
        """Generate full output path inside the safe directory."""
        return os.path.join(self.output_dir, filename)

    def audit_summary(self) -> Dict[str, Any]:
        report = {}

        # Label changes
        if self.label_col in self.orig.columns:
            label_change_count = (
                self.orig[self.label_col].fillna("N/A") !=
                self.corr[self.label_col].fillna("N/A")
            ).sum()

            report["label_changes"] = int(label_change_count)
            report["label_freq_before"] = dict(Counter(self.orig[self.label_col].fillna("N/A")))
            report["label_freq_after"] = dict(Counter(self.corr[self.label_col].fillna("N/A")))

        # Text changes
        text_change_count = (
            self.orig[self.text_col].fillna("") !=
            self.corr[self.text_col].fillna("")
        ).sum()

        report["text_changes"] = int(text_change_count)

        # New rows
        report["new_rows_added"] = int(len(self.corr) - len(self.orig))

        # Changed indices (sample)
        changed_indices = []
        min_len = min(len(self.orig), len(self.corr))

        for i in range(min_len):
            text_changed = (str(self.orig.at[i, self.text_col]) != str(self.corr.at[i, self.text_col]))
            label_changed = (
                self.label_col in self.orig.columns and
                str(self.orig.at[i, self.label_col]) != str(self.corr.at[i, self.label_col])
            )
            if text_changed or label_changed:
                changed_indices.append(i)
                if len(changed_indices) >= 200:
                    break

        report["changed_indices_sample"] = changed_indices
        return report
    
    def embedding_distance(
        self,
        method: str = "auto",
        model_name: Optional[str] = None,
        batch_size: int = 64,
        top_k: int = 10,
        distance_threshold: float = 0.4
    ) -> Dict[str, Any]:

        texts_orig = self.orig[self.text_col].fillna("").astype(str).tolist()
        texts_corr = self.corr[self.text_col].fillna("").astype(str).tolist()
        n = min(len(texts_orig), len(texts_corr))
        texts_orig = texts_orig[:n]
        texts_corr = texts_corr[:n]

        embeddings_orig = None
        embeddings_corr = None
        used_method = None

        # Try sentence-transformers
        if method in ("auto", "sentence_transformer"):
            try:
                from sentence_transformers import SentenceTransformer

                used_method = "sentence_transformer"
                model_name = model_name or "all-MiniLM-L6-v2"
                model = SentenceTransformer(model_name)

                embeddings_orig = model.encode(texts_orig, batch_size=batch_size, show_progress_bar=False)
                embeddings_corr = model.encode(texts_corr, batch_size=batch_size, show_progress_bar=False)

            except Exception:
                embeddings_orig = None
                embeddings_corr = None
                if method == "sentence_transformer":
                    raise

        # TF-IDF fallback
        if embeddings_orig is None and method in ("auto", "tfidf"):
            used_method = "tfidf"
            vect = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
            combined = texts_orig + texts_corr
            X = vect.fit_transform(combined)

            Xo = X[:n]
            Xc = X[n:n + n]

            sims = cosine_similarity(Xo, Xc, dense_output=False)
            sims_diag = np.array([sims[i, i] for i in range(n)])
            distances = 1.0 - sims_diag

            out_file = self._save(f"{self.probe_name}_tfidf_dist_hist.png")
            self._plot_histogram(distances, "TF-IDF embedding distance", out_file)

            return {
                "method": "tfidf",
                "n_samples": n,
                "distances": distances.tolist(),
                "mean_distance": float(np.mean(distances)),
                "median_distance": float(np.median(distances)),
                "max_distance": float(np.max(distances)),
                "pct_over_threshold": float(100 * (distances > distance_threshold).sum() / n)
            }

        # If we have ST embeddings
        if embeddings_orig is not None:
            import sklearn.preprocessing as skp

            eo = skp.normalize(embeddings_orig)
            ec = skp.normalize(embeddings_corr)
            sims = np.sum(eo * ec, axis=1)
            distances = 1.0 - sims

            out_file = self._save(f"{self.probe_name}_embedding_dist_hist.png")
            self._plot_histogram(distances, f"Embedding distance ({model_name})", out_file)

            # Top K changes
            idx_sorted = np.argsort(-distances)[:top_k]

            top_changes = [{
                "index": int(idx),
                "orig": texts_orig[idx][:300],
                "corr": texts_corr[idx][:300],
                "distance": float(distances[idx]),
            } for idx in idx_sorted]

            return {
                "method": used_method,
                "model": model_name,
                "n_samples": n,
                "distances": distances.tolist(),
                "mean_distance": float(np.mean(distances)),
                "median_distance": float(np.median(distances)),
                "max_distance": float(np.max(distances)),
                "pct_over_threshold": float(100 * (distances > distance_threshold).sum() / n),
                "top_k_changes": top_changes
            }

        raise RuntimeError("Embedding distance failed. No backend available.")

    def _plot_histogram(self, values, title, out_path, bins=50):
        try:
            plt.figure(figsize=(6, 3.5))
            plt.title(title)
            plt.hist(values, bins=bins)
            plt.xlabel("Distance")
            plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig(out_path)
            plt.close()
        except:
            pass

    def distribution_shift(self, label_chi2: bool = True) -> Dict[str, Any]:
        report = {}

        # Label distributions
        if self.label_col in self.orig.columns:
            before = Counter(self.orig[self.label_col].astype(str).fillna("N/A"))
            after = Counter(self.corr[self.label_col].astype(str).fillna("N/A"))

            labels = sorted(set(before.keys()).union(after.keys()))
            obs = np.array([before.get(l, 0) for l in labels], float)
            obs2 = np.array([after.get(l, 0) for l in labels], float)

            report["label_freq_before"] = dict(zip(labels, obs.tolist()))
            report["label_freq_after"] = dict(zip(labels, obs2.tolist()))

            try:
                from scipy.stats import chi2_contingency

                table = np.vstack([obs, obs2])
                chi2, p, _, _ = chi2_contingency(table)
                report["label_chi2_stat"] = float(chi2)
                report["label_chi2_pvalue"] = float(p)

            except:
                report["label_js_divergence"] = float(self._js(obs, obs2))

        # Text length distribution
        lengths_before = self.orig[self.text_col].astype(str).apply(len).values
        lengths_after = self.corr[self.text_col].astype(str).apply(len).values

        report["len_mean_before"] = float(lengths_before.mean())
        report["len_mean_after"] = float(lengths_after.mean())
        report["len_std_before"] = float(lengths_before.std())
        report["len_std_after"] = float(lengths_after.std())

        # KS test or fallback
        try:
            from scipy.stats import ks_2samp
            stat, p = ks_2samp(lengths_before, lengths_after)
            report["length_ks_stat"] = float(stat)
            report["length_ks_pvalue"] = float(p)
        except:
            bins = np.histogram_bin_edges(np.concatenate([lengths_before, lengths_after]), bins="auto")
            hb, _ = np.histogram(lengths_before, bins=bins)
            ha, _ = np.histogram(lengths_after, bins=bins)
            report["length_hist_js"] = float(self._js(hb, ha))

        # Save length distribution plot
        try:
            plt.figure(figsize=(7, 3.5))
            plt.title("Text Length Distribution")
            plt.hist(lengths_before, bins=50, alpha=0.6, label="Before")
            plt.hist(lengths_after, bins=50, alpha=0.6, label="After")
            plt.legend()
            plt.tight_layout()
            plt.savefig(self._save(f"{self.probe_name}_length_hist.png"))
            plt.close()
        except:
            pass

        return report

    def _js(self, a, b, eps=1e-10):
        a = np.array(a, float) + eps
        b = np.array(b, float) + eps
        a /= a.sum()
        b /= b.sum()
        m = 0.5 * (a + b)
        return 0.5 * (
            np.sum(a * np.log(a / m)) +
            np.sum(b * np.log(b / m))
        )

    def ner_anomaly(self, model_name="en_core_web_sm", top_k=20, anomaly_threshold_ratio=5.0):
        report = {}

        # Try spaCy
        try:
            import spacy
            try:
                nlp = spacy.load(model_name)
            except:
                nlp = spacy.blank("en")
            use_spacy = True
        except:
            use_spacy = False
            nlp = None

        def spacy_extract(texts):
            ent_text = Counter()
            ent_type = defaultdict(Counter)
            for t in texts:
                doc = nlp(t)
                for e in doc.ents:
                    ent_text[e.text] += 1
                    ent_type[e.label_][e.text] += 1
            return ent_text, ent_type

        def heuristic_extract(texts):
            ent_text = Counter()
            ent_type = defaultdict(Counter)
            ood = ["hanoi", "vietnam", "pho", "banh mi", "mekong", "stock", "cancer", "black hole"]
            for t in texts:
                tokens = t.split()
                for tok in tokens:
                    tok_clean = tok.strip(".,;:()[]\"'")
                    if len(tok_clean) > 2 and tok_clean[0].isupper():
                        ent_text[tok_clean] += 1
                        ent_type["PROPER_NOUN"][tok_clean] += 1
                for kw in ood:
                    if kw in t.lower():
                        ent_text[kw] += 1
                        ent_type["KEYWORD"][kw] += 1
            return ent_text, ent_type

        texts_before = self.orig[self.text_col].astype(str).tolist()
        texts_after = self.corr[self.text_col].astype(str).tolist()

        if use_spacy:
            ent_before, type_before = spacy_extract(texts_before)
            ent_after, type_after = spacy_extract(texts_after)
        else:
            ent_before, type_before = heuristic_extract(texts_before)
            ent_after, type_after = heuristic_extract(texts_after)

        report["num_unique_entities_before"] = len(ent_before)
        report["num_unique_entities_after"] = len(ent_after)

        # Entities introduced
        introduced = [(e, int(c)) for e, c in ent_after.items() if e not in ent_before]

        # Large frequency jumps
        large_inc = []
        for e, cnt_after in ent_after.items():
            cnt_before = ent_before.get(e, 0)
            ratio = float("inf") if cnt_before == 0 else (cnt_after + 1) / (cnt_before + 1)
            if ratio >= anomaly_threshold_ratio:
                large_inc.append({
                    "entity": e,
                    "before": int(cnt_before),
                    "after": int(cnt_after),
                    "ratio": ratio
                })

        report["introduced_entities_sample"] = introduced[:top_k]
        report["large_increases_sample"] = large_inc[:top_k]

        # Save comparison CSV
        try:
            df1 = pd.DataFrame(ent_before.most_common(), columns=["entity", "count_before"])
            df2 = pd.DataFrame(ent_after.most_common(), columns=["entity", "count_after"])
            merged = pd.merge(df1, df2, on="entity", how="outer").fillna(0)
            merged.to_csv(self._save(f"{self.probe_name}_entities_comparison.csv"), index=False)
        except:
            pass

        return report
    
    def run_all(self, do_embedding=True, embedding_method="auto",
                do_distribution=True, do_ner=True):

        report = {}
        report["audit_summary"] = self.audit_summary()

        if do_embedding:
            try:
                report["embedding_distance"] = self.embedding_distance(method=embedding_method)
            except Exception as e:
                report["embedding_distance_error"] = str(e)

        if do_distribution:
            try:
                report["distribution_shift"] = self.distribution_shift()
            except Exception as e:
                report["distribution_shift_error"] = str(e)

        if do_ner:
            try:
                report["ner_anomaly"] = self.ner_anomaly()
            except Exception as e:
                report["ner_anomaly_error"] = str(e)

        # Save JSON report
        try:
            with open(self._save(f"{self.probe_name}_full_report.json"), "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
        except:
            pass

        return report
    
    def export_report_csv(self, report: Dict[str, Any], filename: str = None):
        filename = filename or self._save(f"{self.probe_name}_summary.csv")

        summary = {
            "text_changes": report["audit_summary"].get("text_changes", 0),
            "label_changes": report["audit_summary"].get("label_changes", 0),
            "new_rows_added": report["audit_summary"].get("new_rows_added", 0)
        }

        emb = report.get("embedding_distance", {})
        if emb:
            summary.update({
                "embed_method": emb.get("method", None),
                "embed_mean_distance": emb.get("mean_distance", None),
                "embed_median_distance": emb.get("median_distance", None),
            })

        df = pd.DataFrame([summary])
        df.to_csv(filename, index=False)
        return filename
