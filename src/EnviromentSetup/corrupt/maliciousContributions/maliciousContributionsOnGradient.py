import numpy as np
import copy

class MaliciousContributionsGeneratorOnGradient:
    """ 
    Class to analyze and identify malicious contributions in a federated learning setting on gradient updates.

    Parameters
        ----------
        attack_type : str
            Type of gradient attack to perform.
            Options: ['label_flip', 'sign_flip', 'scaling', 'backdoor', 
                        'random_noise', 'zero_grad', 'small_boost', 
                        'collusion', 'norm_clip_evasion', 'adaptive_multi_round']
        scale_factor : float
            Scaling factor for magnitude-based attacks.
        seed : int
            Random seed for reproducibility.
    """

    def __init__(self, attack_type='label_flip', scale_factor=10.0, seed=2709):
        """
        Parameters
        ----------
        attack_type : str
            Type of gradient attack to perform.
            Options: ['label_flip', 'sign_flip', 'scaling', 'backdoor', 
                        'random_noise', 'zero_grad', 'small_boost', 
                        'collusion', 'norm_clip_evasion', 'adaptive_multi_round']
        scale_factor : float
            Scaling factor for magnitude-based attacks.
        seed : int
            Random seed for reproducibility.
        """
        self.attack_type = attack_type
        self.scale_factor = scale_factor
        self.rng = np.random.default_rng(seed)

    def generate(self, benign_update, **kwargs):
        """
        Generate a malicious version of the given benign update.
        
        Parameters
        ----------
        benign_update : dict
            Dictionary of model parameter gradients (e.g., {'layer1': np.array(...), ...})
        
        Returns
        -------
        malicious_update : dict
            Maliciously modified update.
        """
        attack_fn = getattr(self, f"_attack_{self.attack_type}", None)
        if attack_fn is None:
            raise ValueError(f"Unknown attack_type: {self.attack_type}")
        return attack_fn(benign_update, **kwargs)

    def _attack_label_flip(self, update, **kwargs):
        """
        Reverses the gradient direction for a targeted label.

        Parameters
        ----------
        update : dict
            Dictionary of model parameter gradients (e.g., {'layer1': np.array(...), ...})

        Returns
        -------
        malicious : dict
            Maliciously modified update.
        """

        # Reverse gradient direction for targeted label
        malicious = copy.deepcopy(update)
        for k, v in malicious.items():
            malicious[k] = -v
        return malicious

    def _attack_sign_flip(self, update, **kwargs):

        """
        Flip sign of all gradients (with scaling)

        Parameters
        ----------
        update : dict
            Dictionary of model parameter gradients (e.g., {'layer1': np.array(...), ...})

        Returns
        -------
        malicious : dict
            Maliciously modified update.
        """

        # Flip sign of all gradients (with scaling)
        malicious = copy.deepcopy(update)
        for k, v in malicious.items():
            malicious[k] = -self.scale_factor * v
        return malicious

    def _attack_scaling(self, update, **kwargs):
        """
        Magnitude attack: scale the gradient to exaggerate impact.
        Scales all gradients by a factor of self.scale_factor.
        """

        # Magnitude attack: scale the gradient to exaggerate impact
        malicious = copy.deepcopy(update)
        for k, v in malicious.items():
            malicious[k] = v * self.scale_factor
        return malicious

    def _attack_backdoor(self, update, trigger_grad=None, **kwargs):
        """
        Backdoor attack: inject a trigger gradient into the update.
        If trigger_grad is not provided, will use a randomized trigger_grad.
        
        Parameters
        ----------
        update : dict
            Dictionary of model parameter gradients (e.g., {'layer1': np.array(...), ...})
        trigger_grad : dict, optional
            Dictionary of trigger gradients (e.g., {'layer1': np.array(...), ...})
            If not provided, will use a randomized trigger_grad.
        
        Returns
        -------
        malicious : dict
            Maliciously modified update.
        """

        if trigger_grad is None:
            # Placeholder for trigger_grad if not provided
            trigger_grad = {k: self.scale_factor * self.rng.normal(size=v.shape) for k, v in update.items()}
            print("Warning: Using randomized trigger_grad for backdoor.")
            
        malicious = copy.deepcopy(update)
        for k, v in malicious.items():
            malicious[k] += trigger_grad.get(k, np.zeros_like(v))
        return malicious

    def _attack_random_noise(self, update, noise_scale=1.0, **kwargs):
        """
        Add random noise to gradient update to simulate noisy contributions.
        
        Parameters
        ----------
        update (dict): Gradient update to be modified.
        noise_scale (float, optional): Standard deviation of the random noise to add. Defaults to 1.0.
        
        Returns
        -------
        dict: The modified gradient update with added random noise.
        """

        # Add random noise to gradient
        malicious = copy.deepcopy(update)
        for k, v in malicious.items():
            malicious[k] += self.rng.normal(0, noise_scale, size=v.shape)
        return malicious

    def _attack_zero_grad(self, update, **kwargs):
        """
        Zero out gradients (hide contribution)
        
        Parameters
        ----------
        update : dict
            Dictionary of model parameter gradients (e.g., {'layer1': np.array(...), ...})
        
        Returns
        -------
        malicious : dict
            Maliciously modified update.
        """
        
        # Zero out gradients (hide contribution)
        malicious = copy.deepcopy(update)
        for k in malicious.keys():
            malicious[k] = np.zeros_like(malicious[k])
        return malicious

    def _attack_small_boost(self, update, boost_factor=1.5, **kwargs):
        """
        Slightly increase gradient to bypass norm-based defenses.
        Args:
            update (Dict[str, np.ndarray]): Gradient update to be modified.
            boost_factor (float): Multiplier for gradient update.
        Returns:
            malicious (Dict[str, np.ndarray]): Malicious gradient update.
        """

        # Slightly increase gradient to bypass norm-based defenses
        malicious = copy.deepcopy(update)
        for k, v in malicious.items():
            malicious[k] = v * boost_factor
        return malicious

    def _attack_collusion(self, update, collusion_vector=None, **kwargs):
        """
        Add a crafted collusion vector to gradient update.
        `collusion_vector` should be same shape as update layer.
        If not provided, will use a randomized collusion vector.
        """
        if collusion_vector is None:
            # Placeholder for collusion_vector if not provided
            collusion_vector = {k: self.scale_factor * self.rng.normal(size=v.shape) for k, v in update.items()}
            print("Warning: Using randomized collusion_vector.")
            
        malicious = copy.deepcopy(update)
        for k, v in malicious.items():
            malicious[k] += collusion_vector.get(k, np.zeros_like(v))
        return malicious

    def _attack_norm_clip_evasion(self, update, clip_threshold=1.0, **kwargs):
        """
        Clip the L2 norm of the gradient update to be just below a given threshold.
        This is useful for simulating evasion attacks against norm-based defenses.
        
        Parameters:
        update (dict): Gradient update to be modified.
        clip_threshold (float, optional): Threshold for clipping. Defaults to 1.0.
        
        Returns:
        dict: The modified gradient update with L2 norm clipped to be just below the threshold.
        """

        malicious = copy.deepcopy(update)
        # Calculate L2 norm of the flattened update vector
        total_norm = np.sqrt(sum(np.sum(v**2) for v in malicious.values()))
        # Scale to be just below the threshold
        # We use 0.999 * clip_threshold to ensure it's strictly below the threshold
        scale = (0.999 * clip_threshold) / max(1e-9, total_norm) 
        
        for k, v in malicious.items():
            malicious[k] = v * scale
        return malicious

    def _attack_adaptive_multi_round(self, update, prev_updates=None, adaptation_factor=0.5, **kwargs):
        """
        Adaptively modify the gradient by adding a scaled version of the previous
        updates' mean. This is useful for simulating multi-round attacks where the
        adversary has knowledge of the previous rounds' updates.

        Args:
            update (dict[str, np.ndarray]): The current update to modify.
            prev_updates (list[dict[str, np.ndarray]], optional): A list of previous
                updates. Defaults to None.
            adaptation_factor (float, optional): The scaling factor to apply to the
                previous mean. Defaults to 0.5.

        Returns:
            dict[str, np.ndarray]: The modified update.
        """

        malicious = copy.deepcopy(update)
        if prev_updates is not None and len(prev_updates) > 0:
            for k, v in malicious.items():
                # Calculate the mean of the previous updates for this layer
                prev_mean = np.mean([u[k] for u in prev_updates if k in u], axis=0)
                # Apply adaptation: add a scaled version of the previous mean
                malicious[k] = v + adaptation_factor * prev_mean
        elif prev_updates is None or len(prev_updates) == 0:
            print("Adaptive attack requires 'prev_updates'; running as benign for this round.")
        return malicious

# Example usage
if __name__ == '__main__':
    # Simulate a simple benign update
    benign_update = {
        'layer1': np.array([0.1, -0.2]),
        'layer2': np.array([[0.05, -0.05], [0.02, 0.03]])
    }
    
    print("--- Benign Update ---")
    print(benign_update)
    print("---------------------\n")
    
    # Example 1: Sign Flip Attack
    mc_grad_sign_flip = MaliciousContributionsGeneratorOnGradient(attack_type='sign_flip', scale_factor=5.0)
    malicious_update_sf = mc_grad_sign_flip.generate(benign_update)
    print("--- Sign Flip Attack (scale=5.0) ---")
    print(malicious_update_sf)
    
    # Example 2: Norm Clip Evasion Attack
    clip_threshold_val = 0.5
    mc_grad_clip_evasion = MaliciousContributionsGeneratorOnGradient(attack_type='norm_clip_evasion', scale_factor=1.0)
    malicious_update_nce = mc_grad_clip_evasion.generate(benign_update, clip_threshold=clip_threshold_val)
    print("\n--- Norm Clip Evasion Attack (target norm < {}) ---".format(clip_threshold_val))
    print(malicious_update_nce)

    # Verify the norm for the evasion attack (should be just under 0.5)
    total_norm = np.sqrt(sum(np.sum(v**2) for v in malicious_update_nce.values()))
    print(f"Malicious Update Norm: {total_norm:.6f}")