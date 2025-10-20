import csv
from collections import Counter
from sklearn.model_selection import train_test_split
from .dataloader import ToyTextDataset
from pathlib import Path
import re

def normalize_label(lbl: str) -> str:
    if not isinstance(lbl, str):
        return lbl
    lbl = re.sub(r'[\u200b\u200c\u200d\uFEFF]', '', lbl)  # remove zero-width
    lbl = lbl.strip().lower()
    lbl = re.sub(r'\s+', ' ', lbl)  # collapse multiple spaces
    return lbl

class DatasetBuilder:
    @staticmethod
    def build_vocab(texts, min_freq=1, specials=["[PAD]", "[UNK]"]):
        counter = Counter()
        for text in texts:
            tokens = text.lower().split()
            counter.update(tokens)

        vocab = {}
        idx = 0

        # add specials first
        for sp in specials:
            vocab[sp] = idx
            idx += 1

        # add rest
        for tok, freq in counter.items():
            if freq >= min_freq:
                vocab[tok] = idx
                idx += 1

        return vocab

    @staticmethod
    def load_csv(path, text_col=None, label_col=None):
        if text_col is None or label_col is None:
            raise ValueError("text_col and label_col must be specified")
        texts, labels = [], []
        with open(path, newline="", encoding="utf-8-sig") as f:  # utf-8-sig handles BOM
            reader = csv.DictReader(f)
            # normalize headers: strip spaces
            reader.fieldnames = [h.strip() for h in reader.fieldnames]
            for row in reader:
                texts.append(row[text_col.strip()].strip())
                labels.append(normalize_label(row[label_col.strip()]))

        return texts, labels

    @staticmethod
    def encode_labels(labels):
        unique = sorted(set(labels))
        label2id = {lbl: i for i, lbl in enumerate(unique)}
        y = [label2id[lbl] for lbl in labels]
        return y, label2id

    @staticmethod
    def build_dataset(path, max_len=32, val_ratio=0.1, test_ratio=0.1,
                        vocab=None, label2id=None, text_col=None, label_col=None, config=None):
        # load
        if text_col is None or label_col is None:
            raise ValueError("text_col and label_col must be specified")
        texts, labels = DatasetBuilder.load_csv(path, text_col=text_col, label_col=label_col)

        # reuse or build label2id
        if label2id is None:
            # build fresh mapping
            y, label2id = DatasetBuilder.encode_labels(labels)
        else:
            # extend or validate label2id depending on config
            from Helpers.configLoader import Config
            cfg = config or Config.load(Path(__file__).resolve().parents[2] / "config" / "config.yaml")

            new_labels = [lbl for lbl in labels if lbl not in label2id]

            if new_labels:
                if getattr(cfg, "allow_dynamic_label_expansion", False):
                    for lbl in new_labels:
                        print(f"[DatasetBuilder] New label discovered: {lbl}")
                        label2id[lbl] = len(label2id)
                else:
                    raise ValueError(
                        f"[DatasetBuilder] Found new labels {new_labels} "
                        f"in dataset '{path}', but dynamic label expansion is disabled."
                    )

            # Ensure contiguous IDs (fixes gap like [0..7,9])
            ids = sorted(label2id.values())
            if ids != list(range(len(label2id))):
                print("[DatasetBuilder] Reindexing label2id to be contiguous 0..N-1")
                label2id = {lbl: i for i, lbl in enumerate(sorted(label2id.keys()))}

            # Encode labels after normalization
            y = [label2id[lbl] for lbl in labels]

        # reuse or build vocab
        if vocab is None:
            vocab = DatasetBuilder.build_vocab(texts)

        # split
        X_train, X_temp, y_train, y_temp = train_test_split(texts, y, test_size=val_ratio+test_ratio, random_state=42)
        val_size = int(len(X_temp) * val_ratio / (val_ratio + test_ratio))
        X_val, X_test, y_val, y_test = X_temp[:val_size], X_temp[val_size:], y_temp[:val_size], y_temp[val_size:]

        # wrap in ToyTextDataset
        train_ds = ToyTextDataset(X_train, y_train, vocab, max_len=max_len, num_classes=len(label2id))
        val_ds = ToyTextDataset(X_val, y_val, vocab, max_len=max_len, num_classes=len(label2id))
        test_ds = ToyTextDataset(X_test, y_test, vocab, max_len=max_len, num_classes=len(label2id))

        return train_ds, val_ds, test_ds, vocab, label2id
