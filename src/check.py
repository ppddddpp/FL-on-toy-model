from pathlib import Path
from Helpers import Config
from DataHandler import DatasetBuilder
from Trainer.train_base import BaseTrainer
import torch

BASE_DIR = Path(__file__).resolve().parent.parent

def main():
    # 1. Load config
    cfg = Config.load(BASE_DIR / "config" / "config.yaml")

    # 2. Build dataset
    dataset_path = BASE_DIR / "data" / "medical_data" / "base" / "base_model.csv"
    train_ds, val_ds, test_ds, vocab, label2id = DatasetBuilder.build_dataset(
        path=dataset_path,
        max_len=cfg.max_seq_len
    )

    # 3. Create trainer
    trainer = BaseTrainer(
        train_dataset=train_ds,
        val_dataset=val_ds,
        test_dataset=test_ds,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        cfg=cfg,
        use_wandb=False
    )

    # 4. Train
    trainer.train(epochs=cfg.epochs)

    # 5. Evaluate on test set
    test_acc = trainer.evaluate(split="test")
    print(f"Final Test Accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    main()
