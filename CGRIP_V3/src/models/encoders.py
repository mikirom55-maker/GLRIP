import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import BertModel, BertTokenizer


class PhysicalEncoder(nn.Module):
    def __init__(self, input_dim: int = 10, embedding_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.mlp(x), p=2, dim=-1)


class SocialEncoder(nn.Module):
    def __init__(self, embedding_dim: int = 256, finetune_last_n: int = 2):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        for param in self.bert.parameters():
            param.requires_grad = False
        for layer in self.bert.encoder.layer[-finetune_last_n:]:
            for param in layer.parameters():
                param.requires_grad = True
        for param in self.bert.pooler.parameters():
            param.requires_grad = True
        self.proj = nn.Sequential(
            nn.Linear(768, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        emb = self.proj(pooled)
        return F.normalize(emb, p=2, dim=-1)


class DualEncoder(nn.Module):
    def __init__(self, physical_input_dim: int = 10, embedding_dim: int = 256,
                 dropout: float = 0.1, finetune_last_n: int = 2):
        super().__init__()
        self.physical_encoder = PhysicalEncoder(physical_input_dim, embedding_dim, dropout)
        self.social_encoder = SocialEncoder(embedding_dim, finetune_last_n)
        self.embedding_dim = embedding_dim

    def forward(self, phys_feat, input_ids, attention_mask):
        p_emb = self.physical_encoder(phys_feat)
        s_emb = self.social_encoder(input_ids, attention_mask)
        return p_emb, s_emb

    def encode_physical(self, x):
        return self.physical_encoder(x)

    def encode_social(self, input_ids, attention_mask):
        return self.social_encoder(input_ids, attention_mask)


def extract_physical_features(sat_df) -> torch.Tensor:
    rows = []
    for _, r in sat_df.iterrows():
        ts = r["timestamp"]
        hour = ts.hour + ts.minute / 60.0
        doy = ts.timetuple().tm_yday
        rows.append([
            r["Lat"] / 90.0,
            r["Lon"] / 180.0,
            r["Area"] / 10.0,
            r["FRP"] / 500.0,
            r["Level"] / 5.0,
            r["Reliability"] / 5.0,
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
            np.sin(2 * np.pi * doy / 365),
            np.cos(2 * np.pi * doy / 365),
        ])
    return torch.tensor(rows, dtype=torch.float32)


_tokenizer_cache = {}


def get_bert_tokenizer():
    if "tok" not in _tokenizer_cache:
        _tokenizer_cache["tok"] = BertTokenizer.from_pretrained("bert-base-uncased")
    return _tokenizer_cache["tok"]


def tokenize_social_texts(texts, max_length: int = 64):
    tok = get_bert_tokenizer()
    enc = tok(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return enc["input_ids"], enc["attention_mask"]
