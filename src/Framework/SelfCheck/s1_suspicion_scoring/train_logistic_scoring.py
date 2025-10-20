
import os
import numpy as np
import torch
from .logistic_model import LogisticScoring
from typing import Optional
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[4]

def generate_clients(n_good=50, n_bad=50, seed=42):
    """
    Generate synthetic clients for training a LogisticScoring model.

    Generates two sets of clients with n_good and n_bad samples each, respectively.
    The generated clients are drawn from two normal distributions with different means and standard
    deviations. The clients are then shuffled and returned as a single array.

    Args:
        n_good (int, optional): The number of benign clients to generate. Defaults to 50.
        n_bad (int, optional): The number of malicious clients to generate. Defaults to 50.
        seed (int, optional): The seed for the random number generator. Defaults to 42.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The generated clients and their corresponding labels.
    """
    np.random.seed(seed)
    X_good = np.clip(np.random.normal(loc=[0.1, 0.9, 0.1, 0.2, 0.1],
                                        scale=0.05, size=(n_good, 5)), 0, 1)
    X_bad  = np.clip(np.random.normal(loc=[0.9, 0.1, 0.8, 0.8, 0.5],
                                        scale=0.1, size=(n_bad, 5)), 0, 1)
    X = np.vstack([X_good, X_bad])
    y = np.array([0]*n_good + [1]*n_bad, dtype=np.float32)
    idx = np.random.permutation(len(X))
    return X[idx], y[idx]

def train_model(model: Optional[LogisticScoring]=None,
                X: Optional[np.ndarray]=None, y: Optional[np.ndarray]=None, 
                epochs: int=10, batch_size: int=32,
                save_dir: str = "./models",
                test_clients: Optional[np.ndarray]=None):
    """
    Train a LogisticScoring model on given or generated benign/malicious clients.

    Args:
        model (LogisticScoring, optional): The model to be trained. Defaults to None.
        X (np.ndarray, optional): The feature vectors for the clients. Defaults to None.
        y (np.ndarray, optional): The labels for the clients. Defaults to None.
        epochs (int, optional): The number of epochs to train. Defaults to 10.
        batch_size (int, optional): The batch size for training. Defaults to 32.
        save_dir (str, optional): The directory to save the trained model. Defaults to "./models".

    Returns:
        LogisticScoring: The trained model.
        List[float]: The list of losses for each epoch.
    """
    save_dir = BASE_DIR / "models" if save_dir is None else save_dir
    os.makedirs(save_dir, exist_ok=True)  # ensure folder exists
    model = LogisticScoring(T_flag=0.2, lr=0.05) if model is None else model

    # Prepare dataset
    if X is None and y is None:
        print("[WARN] Dataset not provided. Generating clients...")
    X, y = generate_clients() if X is None and y is None else (X, y)
    n_samples = len(X)
    losses = []

    for epoch in range(epochs):
        perm = np.random.permutation(n_samples)
        epoch_loss = 0.0
        for i in range(0, n_samples, batch_size):
            batch_idx = perm[i:i+batch_size]
            xb = X[batch_idx]
            yb = y[batch_idx]
            loss = model.train_step(xb, yb)
            epoch_loss += loss * len(xb)
        epoch_loss /= n_samples
        losses.append(epoch_loss)
        print(f"Epoch {epoch+1:03d} | Loss = {epoch_loss:.4f}")

    # Test evaluation
    test_clients_local = {
        "benign_1": {"norm": 0.12, "cos": 0.95, "sig": 0.10, "chal": 0.20, "temp": 0.12},
        "benign_2": {"norm": 0.15, "cos": 0.85, "sig": 0.12, "chal": 0.18, "temp": 0.11},
        "anomaly_1": {"norm": 0.55, "cos": 0.40, "sig": 0.35, "chal": 0.40, "temp": 0.25},
        "malicious_1": {"norm": 0.95, "cos": 0.05, "sig": 0.85, "chal": 0.80, "temp": 0.60},
        "malicious_2": {"norm": 0.75, "cos": 0.15, "sig": 0.70, "chal": 0.65, "temp": 0.50},
    }
    test_clients = test_clients_local if test_clients is None else test_clients
    results = model.compute_batch(test_clients)
    model.report(results)

    # Save model with round timestamp or epoch info
    model_path = os.path.join(save_dir, "logistic_trained.pt")
    torch.save(model.state_dict(), model_path)
    print(f"\nSaved trained LogisticScoring to '{model_path}'")

    return model, losses


# Optional: run directly
if __name__ == "__main__":
    train_model(epochs=40, batch_size=16, save_dir=f"{BASE_DIR}/checkpoints/logistic_scoring")
