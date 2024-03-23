import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_features, interim_features, out_features, include_bn) -> None:
        super().__init__()
        self.include_bn = include_bn
        self.lin0 = nn.Linear(in_features, interim_features)
        if self.include_bn:
            self.bn0 = nn.BatchNorm1d(in_features)
        self.lin1 = nn.Linear(interim_features, out_features)
        if self.include_bn:
            self.bn1 = nn.BatchNorm1d(in_features)
        self.lin_projection = nn.Linear(in_features, out_features)
        self.act = nn.ReLU()
    
    def forward(self, x):
        out = self.lin0(x)
        out = self.act(out)
        if self.include_bn:
            out = self.bn0(out)
        out = self.lin1(out)
        out = self.act(out)
        if self.include_bn:
            out = self.bn1(out)
        # Project x to out_features dim
        projection = self.lin_projection(x)
        return self.act(out + projection)
        
class LSTMModel(nn.Module):
    def __init__(self, embedding, lstm, linear) -> None:
        super().__init__()
        self.embedding = embedding
        self.lstm = lstm
        self.linear = nn.Sequential(*linear)
    
    def forward(self, x):
        x=x.squeeze(1)
        embedded = self.embedding(x)
        _, (h_n, _) = self.lstm(embedded)
        hidden = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim = 1)
        out = self.linear(hidden)
        return out
        
def build_SimpleBiLSTM(config: dict):
    lstm_params = config["lstm"]
    linear_params = config["linear"]
    embedding = nn.Embedding(config["vocab_size"], config["embedding_dim"])
    lstm = nn.LSTM(
        input_size=config["embedding_dim"],
        hidden_size=lstm_params["hidden_size"],
        num_layers=lstm_params["num_layers"],
        batch_first=True,
        dropout=lstm_params["dropout"],
        bidirectional=True
    )
    linears = []
    # Multypling by 2 for forward and backward direction
    in_neurons = lstm_params["hidden_size"] * 2
    for i in range(linear_params["num_layers"]):
        out_neurons = linear_params["layers"][i]
        layer = [
            nn.Linear(in_features=in_neurons, out_features=out_neurons),
            nn.ReLU()
        ]
        if linear_params["include_bn"]:
            layer.append(nn.BatchNorm1d(out_neurons))
        if linear_params["dropout"] != False:
            layer.append(nn.Dropout(linear_params["dropout"]))
        in_neurons = out_neurons
        linears.extend(layer)
    final_layer_out = config["out_classes"] if config["out_classes"] > 2 else 1
    linears.append(nn.Linear(out_neurons, final_layer_out))
    return LSTMModel(embedding, lstm, linears)