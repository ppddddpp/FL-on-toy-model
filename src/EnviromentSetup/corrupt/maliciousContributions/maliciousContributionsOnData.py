import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from cleanlab.filter import find_label_issues
from scipy.stats import fisher_exact, beta as beta_fn
from statsmodels.stats.multitest import multipletests
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_predict

class MaliciousContributionsGeneratorOnData:
    """
        Class to analyze and identify malicious contributions in a federated learning setting using Cleanlab and statistical tests on the data.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe containing information about clients and their contributions.
        text_col : str, optional
            Name of the column containing the text data about clients.
            Defaults to 'Information'.
        label_col : str, optional
            Name of the column containing the label data about clients.
            Defaults to 'Group'.
        client_col : str, optional
            Name of the column containing the client data about clients.
            Defaults to 'Group'.
        seed : int, optional
            Random seed for reproducibility.
            Defaults to 2709.
        suspect_fraction : float, optional
            Fraction of clients to label as suspect.
            Defaults to 0.10.

        Attributes
        ----------
        df : pd.DataFrame
            Input dataframe containing information about clients and their contributions.
        text_col : str
            Name of the column containing the text data about clients.
        label_col : str
            Name of the column containing the label data about clients.
        client_col : str
            Name of the column containing the client data about clients.
        embedding_choice : str
            Choice of embedding method (bert or tfidf).
        cleanlab_threshold : float
            Fraction of clients to label as suspect.
        seed : int
            Random seed for reproducibility.
        results_fisher : pd.Series
            Fisher exact test results.
        results_post : pd.DataFrame
            Post-hoc analysis results.
        behavior_df : pd.DataFrame
            Behavior dataset for further analysis.
        """
    def __init__(self, df, text_col='Information', label_col='Group',
                    client_col='Group', 
                    seed=2709,suspect_fraction=0.10):
        """
        Initialize MaliciousContributionsGenerator instance.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe containing information about clients and their contributions.
        text_col : str, optional
            Name of the column containing the text data about clients.
            Defaults to 'Information'.
        label_col : str, optional
            Name of the column containing the label data about clients.
            Defaults to 'Group'.
        client_col : str, optional
            Name of the column containing the client data about clients.
            Defaults to 'Group'.
        seed : int, optional
            Random seed for reproducibility.
            Defaults to 2709.
        suspect_fraction : float, optional
            Fraction of clients to label as suspect.
            Defaults to 0.10.

        Attributes
        ----------
        df : pd.DataFrame
            Input dataframe containing information about clients and their contributions.
        text_col : str
            Name of the column containing the text data about clients.
        label_col : str
            Name of the column containing the label data about clients.
        client_col : str
            Name of the column containing the client data about clients.
        embedding_choice : str
            Choice of embedding method (bert or tfidf).
        cleanlab_threshold : float
            Fraction of clients to label as suspect.
        seed : int
            Random seed for reproducibility.
        results_fisher : pd.Series
            Fisher exact test results.
        results_post : pd.DataFrame
            Post-hoc analysis results.
        behavior_df : pd.DataFrame
            Behavior dataset for further analysis.
        """
        
        self.df = df.copy()
        self.text_col = text_col
        self.label_col = label_col
        self.client_col = client_col
        self.df['suspect'] = False

        self.embedding_choice = 'bert'
        self.cleanlab_threshold = suspect_fraction
        self.seed = seed

        self.results_fisher = None
        self.results_post = None
        self.behavior_df = None

    def embed_features(self, method='tfidf'):
        self.embedding_choice = method
        if method == 'tfidf':
            vectorizer = TfidfVectorizer(max_features=500)
            X = vectorizer.fit_transform(self.df[self.text_col]).toarray()

        elif method == 'bert':
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu', cache_folder='./.cache/sentence_transformers')
            X = model.encode(self.df[self.text_col], convert_to_numpy=True)
        else:
            raise ValueError("Unknown embedding method")

        return X

    def run_cleanlab(self, X, classifier=None):
        """
        Run the Cleanlab algorithm to detect suspect clients.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of clients.
        classifier : sklearn.base.BaseEstimator, optional
            Classifier to use for predicting probabilities.

        Returns
        -------
        None

        Notes
        -----
        The Cleanlab algorithm is used to detect suspect clients
        based on their feature matrix. The algorithm works
        by training a classifier on the feature matrix and
        then predicting the probabilities of each client being
        a suspect. The clients with the highest predicted probabilities
        are then labeled as suspect.

        The number of clients to label as suspect is determined by
        the `cleanlab_threshold` parameter, which is the fraction
        of clients to label as suspect. The default value is 0.10.

        The algorithm returns a boolean array indicating which
        clients are suspect. The array is stored in the `df`
        attribute of the class.
        """
        if classifier is None:
            classifier = RandomForestClassifier(
                n_estimators=300,
                random_state=self.seed
            )
        
        y = self.df[self.label_col].values
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        pred_probs = cross_val_predict(
            classifier, X, y_enc,
            cv=5, method='predict_proba'
        )

        # Find suspect indices using Cleanlab
        suspect_indices_ranked = find_label_issues(
            labels=y_enc,
            pred_probs=pred_probs,
            return_indices_ranked_by='self_confidence'
        )
        k = int(len(y_enc) * self.cleanlab_threshold)
        suspect_indices = suspect_indices_ranked[:max(k,1)]
        self.df['suspect'] = False
        self.df.loc[suspect_indices, 'suspect'] = True

    def fisher_bh(self, alpha=0.05):
        """
        Computes the Fisher's exact test for each client to determine if it submitted a higher than expected proportion of suspect updates.

        Parameters
        ----------
        alpha : float, optional
            Significance level for the BH correction. Defaults to 0.05.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the results of the Fisher's exact test and the BH correction.
        """
        df = self.df
        N = len(df)
        X_total = df['suspect'].sum()

        rows = []

        for client, g in df.groupby(self.client_col):
            n_i = len(g)
            x_i = int(g['suspect'].sum())

            # Handle edge case
            if N == n_i:
                p_raw = 1.0
                odds = np.nan
            else:
                table = np.array([
                    [x_i, n_i - x_i],
                    [int(X_total - x_i), int((N - n_i) - (X_total - x_i))]
                ])

                # Ensure no negative values
                table = np.clip(table, 0, None)

                odds, p_raw = fisher_exact(table, alternative='greater')

            rows.append({
                'client': client,
                'n_i': n_i,
                'x_i': x_i,
                'suspect_rate': x_i / n_i,
                'p_raw': p_raw
            })

        out = pd.DataFrame(rows)

        # BH correction
        rej, p_bh, _, _ = multipletests(
            out['p_raw'], alpha=alpha, method='fdr_bh'
        )
        out['p_BH'] = p_bh
        out['reject_BH'] = rej

        self.results_fisher = out
        return out

    def empirical_bayes(self):
        """
        Computes the empirical Bayes estimate for each client to determine if it submitted a higher than expected proportion of suspect updates.

        The function first computes the proportion of suspect updates p_i for each client, then computes the mean and variance of p_i over all clients.

        Next, it sets the conjugate prior parameters for the Beta distribution, and computes the posterior mean and variance for each client.

        Finally, it computes the P(p_i > tau) for each client, where tau is the maximum proportion of suspect updates that we consider "normal".

        Returns
        -------
        pd.DataFrame
            DataFrame containing the results of the empirical Bayes estimate and the probability of being malicious.
        """
        grouped = self.df.groupby(self.client_col)['suspect'].agg(
            ['sum', 'count']
        )
        grouped.columns = ['x_i', 'n_i']

        p_i = grouped['x_i'] / grouped['n_i']
        p_bar = p_i.mean()
        var_p = p_i.var(ddof=0)

        # Conjugate prior parameters
        k = 10
        alpha0 = p_bar * k
        beta0 = (1 - p_bar) * k

        rows = []

        for client, row in grouped.iterrows():
            x_i, n_i = row['x_i'], row['n_i']

            post_a = x_i + alpha0
            post_b = (n_i - x_i) + beta0
            post_mean = post_a / (post_a + post_b)

            rows.append({
                'client': client,
                'n_i': n_i,
                'x_i': x_i,
                'post_a': post_a,
                'post_b': post_b,
                'post_mean': post_mean
            })

        out = pd.DataFrame(rows).set_index('client')

        # Compute P(p_i > tau)
        tau = min(0.999, p_bar + 3 * np.sqrt(max(var_p, 1e-12)))

        out['P_mal_gt_tau'] = 1 - beta_fn.cdf(tau, out['post_a'], out['post_b'])

        self.results_post = out
        return out

    def hierarchical_beta_binomial(self):
        """
        Replace simple EB with hierarchical Beta-Binomial.
        Estimates global alpha0, beta0 from all clients (method-of-moments),
        then computes per-client posterior with shrinkage.
        Returns a DataFrame like empirical_bayes() but with stronger shrinkage.
        """
        grouped = self.df.groupby(self.client_col)['suspect'].agg(['sum','count']).rename(columns={'sum':'x_i','count':'n_i'})
        # global mean and variance of proportions
        p_i = grouped['x_i'] / grouped['n_i']
        p_bar = p_i.mean()
        var_p = p_i.var(ddof=0)

        # Method-of-moments hierarchical prior
        common = max(p_bar*(1-p_bar)/max(var_p,1e-12) - 1.0, 1e-3)
        alpha0 = max(p_bar * common, 1e-3)
        beta0  = max((1 - p_bar) * common, 1e-3)

        rows = []
        for client, row in grouped.iterrows():
            x_i, n_i = row['x_i'], row['n_i']
            post_a = x_i + alpha0
            post_b = (n_i - x_i) + beta0
            post_mean = post_a / (post_a + post_b)
            rows.append({
                'client': client,
                'n_i': n_i,
                'x_i': x_i,
                'post_a': post_a,
                'post_b': post_b,
                'post_mean': post_mean
            })
        out = pd.DataFrame(rows).set_index('client')
        tau = min(0.999, p_bar + 3*np.sqrt(var_p))
        out['P_mal_gt_tau'] = 1 - beta_fn.cdf(tau, out['post_a'], out['post_b'])
        self.results_post = out
        return out

    def storey_qvalue(self, lamb=0.5):
        """
        Compute Storey q-values for multiple testing.
        Must run fisher_bh() first.
        lamb : float
            Threshold for estimating pi0 (proportion of nulls), typical 0.5.
        Returns:
            DataFrame with additional column 'q_storey'
        """
        if self.results_fisher is None:
            raise ValueError("Run fisher_bh() first")
        pvals = self.results_fisher['p_raw'].values
        m = len(pvals)
        # pi0 estimate
        pi0 = min(1.0, np.sum(pvals > lamb)/(m*(1-lamb)))
        p_ordered = np.argsort(pvals)
        p_sorted = pvals[p_ordered]
        qvals = pi0 * m * p_sorted / np.arange(1, m+1)
        qvals = np.minimum.accumulate(qvals[::-1])[::-1]  # monotone
        out = self.results_fisher.copy()
        out['q_storey'] = qvals
        self.results_fisher = out
        return out

    def bootstrap_ablation(self, train_full_fn, train_repaired_fn, eval_fn, R=30, B=1000):
        """
        Computes effect of removing suspect clients on a model metric with bootstrap CI.

        Parameters
        ----------
        train_full_fn : function(seed) -> model
            Function to train model on full data.
        train_repaired_fn : function(seed) -> model
            Function to train model on data with suspects removed/repaired.
        eval_fn : function(model) -> float
            Function to compute metric (accuracy/F1/etc.) on held-out data.
        R : int
            Number of independent restarts (seeds).
        B : int
            Number of bootstrap resamples for CI.

        Returns
        -------
        dict : {
            'mean_delta': float,
            'ci95': (lower, upper),
            'deltas': np.array,
            'M_full': np.array,
            'M_repaired': np.array
        }
        """
        M_full = []
        M_repaired = []
        deltas = []
        seeds = [self.seed + i for i in range(R)]

        for seed in seeds:
            model_full = train_full_fn(seed)
            m_full = eval_fn(model_full)
            model_rep = train_repaired_fn(seed)
            m_rep = eval_fn(model_rep)

            M_full.append(m_full)
            M_repaired.append(m_rep)
            deltas.append(m_rep - m_full)

        M_full = np.array(M_full)
        M_repaired = np.array(M_repaired)
        deltas = np.array(deltas)

        # bootstrap CI
        rng = np.random.default_rng(0)
        boot_means = []
        for _ in range(B):
            idx = rng.integers(0, R, size=R)
            boot_means.append(deltas[idx].mean())
        lower = np.percentile(boot_means, 2.5)
        upper = np.percentile(boot_means, 97.5)
        mean_delta = deltas.mean()

        return {'mean_delta': mean_delta, 'ci95': (lower, upper), 'deltas': deltas, 'M_full': M_full, 'M_repaired': M_repaired}


    def get_behavior_dataset(self):
        """
        Return a pandas DataFrame containing the behavior features of all clients.

        The DataFrame will have the following columns:
            - client: client ID
            - n_i: number of contributions
            - x_i: number of suspect contributions
            - suspect_rate: proportion of suspect contributions
            - p_raw: Fisher's exact p-value
            - p_BH: BH-adjusted p-value
            - reject_BH: BH rejection boolean
            - q_storey: Storey q-value
            - post_mean: posterior mean of the client's contribution probability
            - P_mal_gt_tau: probability that the client's contribution probability is greater than tau

        Raises:
            ValueError: if fisher_bh() or hierarchical_beta_binomial() has not been run first
        """
        if self.results_fisher is None or self.results_post is None:
            raise ValueError("Must run fisher_bh() and hierarchical_beta_binomial() first")

        # Ensure Storey q-values exist
        if 'q_storey' not in self.results_fisher.columns:
            self.storey_qvalue()

        # Join all results
        behavior_df = self.results_fisher.set_index('client').join(
            self.results_post[['post_mean', 'P_mal_gt_tau']]
        ).reset_index()

        self.behavior_df = behavior_df
        return behavior_df

    def sensitivity_analysis(self, embedding_methods=['tfidf', 'bert'],
                                seeds=[2709, 279, 42]):
        """
        Perform sensitivity analysis on the empirical Bayes estimate.

        Parameters
        ----------
        embedding_methods : list of str
            List of embedding methods to use for the sensitivity analysis.
            Defaults to ['tfidf', 'bert'].
        seeds : list of int
            List of seeds to use for the sensitivity analysis.
            Defaults to [2709, 279, 42].

        Returns
        -------
        dict
            A dictionary containing the results of the sensitivity analysis.
            The keys are in the format '{embedding_method}_seed{seed_value}'.
            The values are the DataFrames containing the results of the empirical Bayes estimate.
        """
        results = {}

        for emb in embedding_methods:
            for s in seeds:
                analyzer = MaliciousContributionsGeneratorOnData(
                    self.df,
                    text_col=self.text_col,
                    label_col=self.label_col,
                    client_col=self.client_col,
                    suspect_fraction=self.cleanlab_threshold
                )
                analyzer.seed = s

                X = analyzer.embed_features(method=emb)
                analyzer.run_cleanlab(X)
                analyzer.fisher_bh()
                analyzer.empirical_bayes()
                results[f"{emb}_seed{s}"] = analyzer.get_behavior_dataset()

        return results

    def plot_suspect_rates(self, metric='suspect_rate', top_n=None):
        """
        Plot a client-level metric for inspection.

        Parameters
        ----------
        metric : str
            Column in behavior_df to plot. Options: 'suspect_rate', 'post_mean', 'P_mal_gt_tau', 'q_storey'
            Defaults to 'suspect_rate'.
        top_n : int, optional
            If set, only plot the top N clients sorted by the selected metric.

        Raises
        ------
        ValueError
            If behavior_df has not been created or metric not found.
        """
        import matplotlib.pyplot as plt

        if self.behavior_df is None:
            raise ValueError("Run get_behavior_dataset() first.")

        if metric not in self.behavior_df.columns:
            raise ValueError(f"Metric '{metric}' not found in behavior dataset.")

        df_plot = self.behavior_df.sort_values(metric, ascending=False)
        if top_n is not None:
            df_plot = df_plot.head(top_n)

        plt.figure(figsize=(10, 5))
        plt.bar(df_plot['client'], df_plot[metric], color='salmon')
        plt.xticks(rotation=45)
        plt.ylabel(metric)
        plt.title(f'{metric} per Client')
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    # Exmaple run
    df = pd.read_csv("animal.csv") 

    analyzer = MaliciousContributionsGeneratorOnData(df, text_col='Information', label_col='Group', client_col='Group')
    X = analyzer.embed_features(method='tfidf')
    
    # Run Cleanlab to mark suspects
    analyzer.run_cleanlab(X)

    # Fisher exact test + BH
    fisher_results = analyzer.fisher_bh(alpha=0.05)

    # Compute Storey q-values
    analyzer.storey_qvalue(lamb=0.5)

    # Hierarchical Beta-Binomial posterior
    posterior_results = analyzer.hierarchical_beta_binomial()

    # Get behavior dataset
    behavior_df = analyzer.get_behavior_dataset()
    print(behavior_df)

    # Optional: bootstrap ablation
    # Must define train_full_fn, train_repaired_fn, eval_fn for model before use
    # ablation_results = analyzer.bootstrap_ablation(train_full_fn, train_repaired_fn, eval_fn)

    # Optional: sensitivity analysis over embeddings and seeds
    # sensitivity_results = analyzer.sensitivity_analysis(embedding_methods=['tfidf','bert'], seeds=[42, 2709])

    # Optional: plotting
    # Plot top 10 clients by suspect rate
    #analyzer.plot_suspect_rates(metric='suspect_rate', top_n=10)

    # Plot posterior probability of being malicious
    #analyzer.plot_suspect_rates(metric='P_mal_gt_tau', top_n=10)

    # Plot Storey q-values
    #analyzer.plot_suspect_rates(metric='q_storey', top_n=10)

