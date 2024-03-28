import math
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
    """Simple classfication transformer

    It consist of one encoder layer and feedforward network for classification
    by first [CLS] token

    Attributes:
        pos_encoder: Module for calculating pos embeddings
        transformer_encoder_layer: Encoder layer
        embedding: Embedding layer
        d_model: Embedding dimensions
        linears: Feedforward classification model
    """
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        nhead: int,
        d_hid: int,
        maxlen: int,
        dropout: float,
        linears: list[nn.Module],
    ):
        super().__init__()
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout, maxlen)
        self.transformer_encoder_layer = TransformerEncoderLayer(
            embedding_dim,
            nhead,
            d_hid,
            dropout,
            batch_first = True
        )
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.d_model = embedding_dim
        self.linears = nn.Sequential(*linears)

    def forward(self, src: Tensor, mask: Tensor = None) -> Tensor:
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        encoding = self.transformer_encoder_layer(src, src_key_padding_mask = ~mask)
        cls_token = encoding[:, 0, :].view(src.shape[0], -1)
        output = self.linears(cls_token)
        return output


def build_transfomer(config: dict):
    """Build transformer model based on config

    Args:
        config (dict): Model config

    Returns:
        ClasificationTransformerModel: Builded model
    """
    transformer_params = config["transformer"]
    linear_params = config["linear"]
    # Build feed forward classificator
    linears = []
    in_neurons = transformer_params["embedding_dim"]
    for i in range(linear_params["num_layers"]):
        out_neurons = linear_params["layers"][i]
        layer = [
            nn.Linear(in_features=in_neurons, out_features=out_neurons)
        ]
        if linear_params["include_bn"]:
            layer.append(nn.BatchNorm1d(out_neurons))
        layer.append(nn.ReLU())
        if linear_params["dropout"] is not False:
            layer.append(nn.Dropout(linear_params["dropout"]))
        in_neurons = out_neurons
        linears.extend(layer)
    final_layer_out = config["out_classes"] if config["out_classes"] > 2 else 1
    linears.append(nn.Linear(out_neurons, final_layer_out))
    return ClasificationTransformerModel(**transformer_params, linears = linears)
