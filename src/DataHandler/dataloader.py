import torch
from torch.utils.data import Dataset

class ToyTextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=32, num_classes=None):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

        self._vocab_size = len(vocab)
        self._num_classes = num_classes

        # special tokens
        self.pad_id = vocab.get("[PAD]", 0)
        self.unk_id = vocab.get("[UNK]", 1)

    def __len__(self):
        return len(self.texts)

    def encode(self, text):
        tokens = text.lower().split()
        ids = [self.vocab.get(tok, self.unk_id) for tok in tokens]

        # truncate
        ids = ids[:self.max_len]

        # pad
        pad_len = self.max_len - len(ids)
        ids = ids + [self.pad_id] * pad_len

        # attention mask
        mask = [1] * (len(ids) - pad_len) + [0] * pad_len

        return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.long)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        ids, mask = self.encode(text)
        return ids, mask, torch.tensor(label, dtype=torch.long)

    @property
    def vocab_size(self):
        return self._vocab_size

    @property
    def num_classes(self):
        return self._num_classes

