import numpy as np
import copy

class SybilAmplificationAttack:
    """
    Simulates Sybil amplification attacks in FL.
    Multiple fake identities coordinate to amplify their gradient direction.
    """

    def __init__(self, amplification_factor=5.0, shared_vector=None, 
                    fake_data_size=500, collusion=True):
        """
        amplification_factor : float
            Strength of boosting malicious gradient direction.
        shared_vector : dict
            The collusion gradient vector shared across Sybils.
        fake_data_size : int
            Each Sybil claims a large dataset to maximize aggregator weight.
        collusion : bool
            Whether Sybil identities coordinate (identical updates).
        """
        self.amplification_factor = amplification_factor
        self.shared_vector = shared_vector  # dict of same shape as update
        self.fake_data_size = fake_data_size
        self.collusion = collusion

    def apply(self, benign_update, client_metadata=None):
        """
        Applies Sybil attack to a single client's update.
        """
        update = copy.deepcopy(benign_update)
        metadata = client_metadata.copy() if client_metadata else {}

        # Fake data contribution to influence FedAvg/FedProx weighting
        metadata['num_samples'] = self.fake_data_size

        if self.collusion:
            # Replace benign gradient with shared malicious vector
            if self.shared_vector is None:
                raise ValueError("shared_vector must be provided for sybil collusion attack")
            for k in update:
                update[k] = self.shared_vector[k] * self.amplification_factor
        else:
            # Non-collusive amplification: scale the benign update
            for k in update:
                update[k] = update[k] * self.amplification_factor

        return update, metadata
