import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

class FinetuneBaseModel:
    def __init__(self, model: nn.Module, dataset, val_dataset=None, device="cpu", lr=1e-4, batch_size=8):
        self.model = model.to(device)
        self.dataset = dataset
        self.val_dataset = val_dataset
        self.loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size) if val_dataset else None
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.device = device

    def finetune(self, local_epochs=1):
        self.model.train()
        for epoch in range(local_epochs):
            total_loss = 0.0
            for batch in self.loader:
                if len(batch) == 2:
                    input_ids, labels = [x.to(self.device) for x in batch]
                    logits = self.model(input_ids)
                else:
                    input_ids, attention_mask, labels = [x.to(self.device) for x in batch]
                    logits = self.model(input_ids, attention_mask)

                loss = self.criterion(logits, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(self.loader)
            print(f"[FinetuneBaseModel] Epoch {epoch+1}/{local_epochs} - Loss: {avg_loss:.4f}")
        return self.model.state_dict()

    def evaluate(self, split="val"):
        """
        Evaluate the model on validation or training set.
        """
        loader = self.val_loader if split == "val" else self.loader
        if loader is None:
            print(f"[FinetuneBaseModel] No {split} dataset available.")
            return None

        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in loader:
                if len(batch) == 2:
                    input_ids, labels = [x.to(self.device) for x in batch]
                    logits = self.model(input_ids)
                else:
                    input_ids, attention_mask, labels = [x.to(self.device) for x in batch]
                    logits = self.model(input_ids, attention_mask)

                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        acc = correct / total
        print(f"[FinetuneBaseModel] {split} accuracy: {acc:.4f}")
        return acc

    def load_weights(self, state_dict):
        self.model.load_state_dict(state_dict)

    def get_weights(self):
        return self.model.state_dict()
