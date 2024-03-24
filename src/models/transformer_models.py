import math
import os

import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoderLayer

class PositionalEncoding(nn.Module):
    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        max_len: int = 5000
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class ClasificationTransformerModel(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        nhead: int,
        d_hid: int,
        maxlen: int,
        dropout: float = 0.5
    ):
        super().__init__()
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout, maxlen)
        self.transformer_encoder_layer = TransformerEncoderLayer(embedding_dim, nhead, d_hid, dropout, batch_first = True)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.d_model = embedding_dim
        self.maxlen = maxlen
        self.linear0 = nn.Linear(embedding_dim, 256)
        self.linear1 = nn.Linear(256, 128)
        self.linear2 = nn.Linear(128, 1)
        self.activation = nn.ReLU()
        
    def forward(self, src: Tensor, mask: Tensor = None) -> Tensor:
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        encoding = self.transformer_encoder_layer(src, src_key_padding_mask = ~mask)
        cls_token = encoding[:, 0, :].view(src.shape[0], -1)
        output = self.activation(self.linear0(cls_token))
        output = self.activation(self.linear1(output))
        output = self.linear2(output)
        return output
    
def build_transfomer(config):
    config.pop('type', None)
    config.pop('out_classes', None)
    return ClasificationTransformerModel(**config)
