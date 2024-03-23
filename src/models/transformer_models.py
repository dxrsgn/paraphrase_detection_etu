import math
import os

import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer

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
        self.transformer_encoder_layer = TransformerEncoderLayer(embedding_dim, nhead, d_hid, dropout)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.d_model = embedding_dim
        self.maxlen = maxlen
        self.linear0 = nn.Linear(embedding_dim, 64)
        self.linear1 = nn.Linear(64, 1)
        self.linear2 = nn.Linear(self.maxlen, 32)
        self.linear3 = nn.Linear(32, 1)
        self.activation = nn.ReLU()
        
    def forward(self, src: Tensor) -> Tensor:
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder_layer(src)
        output = self.activation(self.linear0(output))
        output = self.activation(self.linear1(output))
        output = output.view(-1, self.maxlen)
        output = self.activation(self.linear2(output))
        output = self.linear3(output)
        return output
    
def build_transfomer(config):
    return ClasificationTransformerModel(**config)