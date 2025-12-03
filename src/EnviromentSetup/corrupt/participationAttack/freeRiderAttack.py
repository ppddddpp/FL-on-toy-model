import numpy as np
import copy
import pandas as pd

class FreeRiderAttack:
    """
    Gradient-side free-rider behavior.
    Works AFTER MC-Grad and should never pollute MC-Grad history.
    """

    def __init__(self, mode="zero", fake_data_size=None, noise_scale=0.001, seed=123):
        self.mode = mode
        self.fake_data_size = fake_data_size
        self.noise_scale = noise_scale
        self.rng = np.random.default_rng(seed)
        self.cached_gradient = None

    def apply(self, benign_update, client_metadata=None):
        """
        Parameters
        ----------
        benign_update: dict of numpy arrays (ALREADY ATTACKED by MC-Grad)
        client_metadata: dict with num_samples

        Returns malicious_update, updated_metadata
        """

        update = {k: v.copy() for k, v in benign_update.items()}
        metadata = (client_metadata or {}).copy()

        # override dataset size (lie)
        if self.fake_data_size is not None:
            metadata["num_samples"] = self.fake_data_size

        # modes
        if self.mode == "zero":
            for k in update:
                update[k] = np.zeros_like(update[k])

        elif self.mode == "weak":
            for k in update:
                update[k] = update[k] * self.noise_scale

        elif self.mode == "cached":
            if self.cached_gradient is None:
                # store clean copy for next time
                self.cached_gradient = {k: v.copy() for k, v in update.items()}
            else:
                # reuse previous gradient
                update = {k: v.copy() for k, v in self.cached_gradient.items()}

        elif self.mode == "no_data":
            for k in update:
                update[k] = self.rng.normal(
                    loc=0.0,
                    scale=self.noise_scale,
                    size=update[k].shape
                )

        else:
            raise ValueError(f"Unknown FreeRiderAttack mode: {self.mode}")

        return update, metadata

class FreeRiderDataAttack:
    """
    Simulates data-level free rider behavior in Federated Learning.
    These clients:
        - train on no data
        - or use extremely tiny or duplicated data
        - or lie about their dataset size
    """

    def __init__(
        self,
        mode="empty",
        tiny_fraction=0.05,
        duplicate_factor=50,
        fake_data_size=None,
        random_noise_dim=None,
        seed=123
    ):
        """
        Parameters
        ----------
        mode:
            - 'empty'      → no training data
            - 'tiny'       → use tiny fraction of data
            - 'duplicate'  → reuse a few samples many times
            - 'random'     → random noise samples (low-quality data)
        
        tiny_fraction:
            Fraction of the real dataset to keep in 'tiny' mode.
        
        duplicate_factor:
            Number of times to repeat the tiny subset in 'duplicate' mode.

        fake_data_size:
            If not None, override metadata['num_samples'] to lie about data size.

        random_noise_dim:
            Dimensionality of synthetic random input for 'random' mode.

        """
        self.mode = mode
        self.tiny_fraction = tiny_fraction
        self.duplicate_factor = duplicate_factor
        self.fake_data_size = fake_data_size
        self.random_noise_dim = random_noise_dim
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def apply(self, dataset, metadata=None):
        """
        Parameters
        ----------
        dataset : pandas.DataFrame or list of samples
            The original clean dataset assigned to the client.
        
        metadata : dict
            e.g., {"num_samples": len(dataset)}

        Returns
        -------
        malicious_dataset
        updated_metadata
        """
        if metadata is None:
            metadata = {}

        # Default metadata setup
        original_len = len(dataset)
        metadata = metadata.copy()

        if self.mode == "empty":
            # Client has no local data
            malicious_data = dataset.iloc[0:0] if hasattr(dataset, "iloc") else []

        elif self.mode == "tiny":
            # Keep only a tiny subset of the data
            k = max(1, int(original_len * self.tiny_fraction))
            malicious_data = dataset.sample(k, replace=False) if hasattr(dataset, "sample") else dataset[:k]

        elif self.mode == "duplicate":
            # Pick a small subset and duplicate it heavily
            k = max(1, int(original_len * self.tiny_fraction))
            small_subset = dataset.sample(k) if hasattr(dataset, "sample") else dataset[:k]

            if hasattr(dataset, "iloc"):
                malicious_data = pd.concat([small_subset] * self.duplicate_factor, ignore_index=True)
            else:
                malicious_data = small_subset * self.duplicate_factor

        elif self.mode == "random":
            # Replace dataset with random noise (bad-quality samples)
            if self.random_noise_dim is None:
                raise ValueError("random_noise_dim must be set for random mode.")

            malicious_data = self._generate_random_dataset(original_len)

        else:
            raise ValueError(f"Unknown FreeRiderDataAttack mode: {self.mode}")

        if self.fake_data_size is not None:
            metadata["num_samples"] = self.fake_data_size
        else:
            metadata["num_samples"] = len(malicious_data)

        return malicious_data, metadata
    
    def _generate_random_dataset(self, dataset):
        """
        Generate a random (non-informative) dataset matching the input dataset.
        
        Parameters
        ----------
        dataset : pandas.DataFrame or list/np.array
            The original dataset to mimic shape.
        
        Returns
        -------
        malicious_dataset : pandas.DataFrame or np.array
            Random dataset of the same shape as input.
        """
        length = len(dataset)

        if isinstance(dataset, pd.DataFrame):
            data = {}
            for col in dataset.columns:
                if pd.api.types.is_numeric_dtype(dataset[col]):
                    data[col] = self.rng.normal(0, 1, size=length)
                else:
                    data[col] = [f"rand_{self.rng.integers(1e9)}" for _ in range(length)]
            return pd.DataFrame(data)

        else:
            dim = self.random_noise_dim or 32  # default dim if list/array
            return self.rng.normal(0, 1, size=(length, dim))

