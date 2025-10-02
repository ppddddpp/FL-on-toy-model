import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os

class Client:
    def __init__(self, client_id, model_fn, dataset, device="cpu"):
        """
        Args:
            client_id (str): identifier for the client (e.g., "africa_40")
            model_fn (callable): function that returns a new model instance
            dataset (Dataset): local dataset for this client
            device (str): "cpu" or "cuda"
        """
        self.client_id = client_id
        self.model_fn = model_fn
        self.dataset = dataset
        self.device = device

    def local_train(self, global_weights, epochs=1, batch_size=16, lr=1e-3,
                    save_path=None, log_interval=10):
        """
        Perform local training starting from global weights.

        Returns:
            dict: updated model weights (state_dict)
            int: number of samples used in training
        """
        # Initialize model
        model = self.model_fn().to(self.device)
        model.load_state_dict(global_weights)

        # Setup dataloader, optimizer, loss
        loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        # Train locally
        model.train()
        for epoch in range(epochs):
            for batch_idx, (ids, mask, y) in enumerate(loader):
                ids, mask, y = ids.to(self.device), mask.to(self.device), y.to(self.device)

                optimizer.zero_grad()
                logits = model(ids, attention_mask=mask)
                loss = criterion(logits, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

                if (batch_idx + 1) % log_interval == 0:
                    print(f"[{self.client_id}] Epoch {epoch+1} "
                          f"Batch {batch_idx+1}/{len(loader)} "
                          f"Loss: {loss.item():.4f}")

        # Optionally save checkpoint
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"[{self.client_id}] Saved model to {save_path}")

        # Return updated weights + number of samples
        return {k: v.cpu() for k, v in model.state_dict().items()}, len(self.dataset)

    def evaluate(self, weights=None, batch_size=16):
        """
        Evaluate accuracy on the client's dataset.

        Returns:
            float: accuracy (0-1)
        """
        model = self.model_fn().to(self.device)
        if weights is not None:
            model.load_state_dict(weights)

        loader = DataLoader(self.dataset, batch_size=batch_size)
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for ids, mask, y in loader:
                ids, mask, y = ids.to(self.device), mask.to(self.device), y.to(self.device)
                logits = model(ids, attention_mask=mask)
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        acc = correct / total if total > 0 else 0.0
        print(f"[{self.client_id}] Evaluation Accuracy: {acc:.4f}")
        return float(acc)