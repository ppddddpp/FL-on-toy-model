import os
import torch
from pathlib import Path

from Trainer.train_base import BaseTrainer 
from Trainer.finetune_base import FinetuneBaseModel

class Server:
    def __init__(self, model_cls, config, clients, checkpoint_dir="checkpoints/base_model", device="cpu"):
        self.model_cls = model_cls
        self.config = config
        self.clients = clients
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # try to load base model
        self.global_model = self._load_or_train_base()

    def _load_or_train_base(self):
        latest_ckpt = self._get_latest_checkpoint()
        model = self.model_cls(**self.config["model"])
        if latest_ckpt is None:
            print("[Server] No base model found, training new base model...")
            trainer = BaseTrainer(model, self.config, device=self.device)
            trainer.train()  # will save checkpoints automatically
            latest_ckpt = self._get_latest_checkpoint()

        print(f"[Server] Loading base model from {latest_ckpt}")
        model.load_state_dict(torch.load(latest_ckpt, map_location=self.device))
        return model

    def _get_latest_checkpoint(self):
        ckpts = list(self.checkpoint_dir.glob("epoch*.pt"))
        if not ckpts:
            return None
        return max(ckpts, key=os.path.getctime)  # latest

    def fedavg(self, client_updates):
        """Federated averaging of model weights."""
        new_state = {}
        total_samples = sum(num_samples for _, num_samples in client_updates)

        for k in client_updates[0][0].keys():
            new_state[k] = sum(
                weights[k] * (num_samples / total_samples)
                for weights, num_samples in client_updates
            )
        return new_state

    def run(self, num_rounds=5, local_epochs=3):
        global_weights = self.global_model.state_dict()

        for rnd in range(num_rounds):
            print(f"\n[Server] Round {rnd+1}/{num_rounds}")
            client_updates = []

            for cid, client_data in enumerate(self.clients):
                client = FinetuneBaseModel(
                    model=self.model_cls(**self.config["model"]),
                    dataset=client_data["train"],
                    val_dataset=client_data.get("val"),
                    device=self.device,
                )
                client.load_weights(global_weights)
                new_weights = client.finetune(local_epochs=local_epochs)
                acc = client.evaluate(split="val")
                print(f"[Client {cid}] Local Val Acc: {acc:.4f}")
                client_updates.append((new_weights, len(client_data["train"])))

            # FedAvg aggregation
            global_weights = self.fedavg(client_updates)
            self.global_model.load_state_dict(global_weights)

            # save checkpoint each round
            ckpt_path = self.checkpoint_dir / f"round{rnd+1}.pt"
            torch.save(global_weights, ckpt_path)
            print(f"[Server] Saved global model -> {ckpt_path}")
