import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import LabelEncoder

class MaliciousContributionsGeneratorOnData:
    """
    True malicious data generator for data poisoning attacks.
    """

    def __init__(self, df, text_col='Information', label_col='Group',
                    client_col='Group', seed=2709, suspect_fraction=0.10):

        self.df = df.copy()
        self.text_col = text_col
        self.label_col = label_col
        self.client_col = client_col
        self.seed = seed
        self.suspect_fraction = suspect_fraction

        random.seed(seed)
        np.random.seed(seed)

        # internal utilities
        self.le = LabelEncoder()

        # avoid crash when df is empty or label column missing
        if len(self.df) > 0 and label_col in self.df.columns:
            self.df[label_col] = self.le.fit_transform(self.df[label_col])
            self.classes = list(self.le.classes_)
        else:
            self.classes = []

    def _choose_indices(self):
        k = max(1, int(len(self.df) * self.suspect_fraction))
        return np.random.choice(self.df.index, size=k, replace=False)

    def _random_other_label(self, current):
        choices = [c for c in range(len(self.classes)) if c != current]
        return random.choice(choices)

    def random_label_flip(self):
        idx = self._choose_indices()
        for i in idx:
            current = self.df.at[i, self.label_col]
            self.df.at[i, self.label_col] = self._random_other_label(current)

        return self.df
    
    def _generate_random_text(self, length=10):
        """
        Generate nonsensical random text as free-rider noise.
        """
        vocab = "abcdefghijklmnopqrstuvwxyz"
        words = []

        for _ in range(length):
            wlen = np.random.randint(3, 8)
            word = "".join(random.choice(vocab) for _ in range(wlen))
            words.append(word)

        return " ".join(words)

    def random_text_noise(self, min_len=6, max_len=15):
        """
        Replace some clients' text with random garbage sequences.
        Similar to free-rider or zero-contribution attacks.
        """
        idx = self._choose_indices()
        for i in idx:
            L = np.random.randint(min_len, max_len)
            self.df.at[i, self.text_col] = self._generate_random_text(L)
        return self.df

    def targeted_label_flip(self, src_label, tgt_label):
        src_encoded = self.le.transform([src_label])[0]
        tgt_encoded = self.le.transform([tgt_label])[0]

        src_indices = self.df[self.df[self.label_col] == src_encoded].index
        k = max(1, int(len(src_indices) * self.suspect_fraction))
        chosen = np.random.choice(src_indices, size=k, replace=False)

        for i in chosen:
            self.df.at[i, self.label_col] = tgt_encoded
        return self.df

    def add_backdoor_trigger(self, trigger="cfX42"):
        idx = self._choose_indices()
        for i in idx:
            self.df.at[i, self.text_col] = trigger + " " + self.df.at[i, self.text_col]
        return self.df

    def semantic_noise(self, noise_tokens=None):
        noise_tokens = ["@@", "###", "!!", "..."] if noise_tokens is None else noise_tokens
        idx = self._choose_indices()

        for i in idx:
            text = self.df.at[i, self.text_col]
            words = text.split()
            if len(words) >= 3:
                pos = random.randint(0, len(words) - 1)
                words.insert(pos, random.choice(noise_tokens))
            self.df.at[i, self.text_col] = " ".join(words)
        return self.df

    def duplicate_flood(self, factor=5):
        idx = self._choose_indices()
        duplicated = self.df.loc[idx].copy()

        out = pd.concat([self.df] + [duplicated for _ in range(factor)],
                        ignore_index=True)

        self.df = out
        return self.df

    def ood_injection(self, sentences=None):
        if sentences is None:
            sentences = [
                "Hanoi is capital of Vietnam with rich history.",
                "Vietnamese cuisine includes pho and banh mi.",
                "The Mekong Delta is a vital region in southern Vietnam.",
            ]
        idx = self._choose_indices()
        for i in idx:
            self.df.at[i, self.text_col] = random.choice(sentences)
        return self.df

    def get_corrupted_dataset(self):
        return self.df
