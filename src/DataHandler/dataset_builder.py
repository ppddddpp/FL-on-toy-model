import csv
from collections import Counter
from sklearn.model_selection import train_test_split
from .dataloader import ToyTextDataset

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
    def load_csv(path, text_col="Disease_Information", label_col="Body_System"):
        texts, labels = [], []
        with open(path, newline="", encoding="utf-8-sig") as f:  # utf-8-sig handles BOM
            reader = csv.DictReader(f)
            # normalize headers: strip spaces
            reader.fieldnames = [h.strip() for h in reader.fieldnames]
            for row in reader:
                texts.append(row[text_col.strip()])
                labels.append(row[label_col.strip()])
        return texts, labels

    @staticmethod
    def encode_labels(labels):
        unique = sorted(set(labels))
        label2id = {lbl: i for i, lbl in enumerate(unique)}
        y = [label2id[lbl] for lbl in labels]
        return y, label2id

    @staticmethod
    def build_dataset(path, max_len=32, val_ratio=0.1, test_ratio=0.1,
                        vocab=None, label2id=None):
        # load
        texts, labels = DatasetBuilder.load_csv(path)

        # reuse or build label2id
        if label2id is None:
            y, label2id = DatasetBuilder.encode_labels(labels)
        else:
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
