import torch
import torch.nn as nn

class ToyBERTClassifier(nn.Module):
    def __init__(self,
                    vocab_size,
                    num_classes,
                    d_model=128,
                    nhead=4,
                    num_layers=2,
                    dim_ff=256,
                    max_len=128,
                    dropout=0.1):
        super().__init__()
        self.d_model = d_model

        # Embeddings
        self.token_embeddings = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.position_embeddings = nn.Embedding(max_len, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))  # [CLS]

        # (optional: segment embeddings if needed)
        self.layernorm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Transformer Encoder (BERT-style blocks)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),          # BERT uses tanh in pooler
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x, attention_mask=None, return_hidden: bool = False):
        """
        x: (B, T) token IDs
        attention_mask: (B, T) with 1 for real tokens, 0 for padding
        return_hidden: if True, also return the CLS embedding
        returns: 
            logits (B, num_classes)
            cls_vec (B, D) if return_hidden=True
        """
        B, T = x.size()
        pos_ids = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        tok_emb = self.token_embeddings(x)  # (B, T, D)
        pos_emb = self.position_embeddings(pos_ids)
        emb = tok_emb + pos_emb

        # prepend CLS
        cls = self.cls_token.expand(B, -1, -1)
        emb = torch.cat([cls, emb], dim=1)

        emb = self.layernorm(emb)
        emb = self.dropout(emb)

        # --- handle attention mask ---
        if attention_mask is not None:
            cls_mask = torch.ones((B, 1), device=x.device, dtype=attention_mask.dtype)
            attn_mask = torch.cat([cls_mask, attention_mask], dim=1)  # (B, T+1)
            key_padding_mask = (attn_mask == 0)  # (B, T+1)
        else:
            key_padding_mask = None

        # transformer encoder
        out = self.encoder(emb, src_key_padding_mask=key_padding_mask)

        # CLS vector
        cls_vec = out[:, 0, :]

        # logits
        logits = self.classifier(cls_vec)

        if return_hidden:
            return logits, cls_vec
        return logits

