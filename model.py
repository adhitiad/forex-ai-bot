import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.get_buffer("pe")[: x.size(1), :]
        return x


class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, d_model=128, output_dim=3):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.decoder = nn.Sequential(
            nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, output_dim)
        )

    def forward(self, x):
        x = self.pos_encoder(self.embedding(x))
        x = self.transformer(x)
        return F.softmax(self.decoder(x[:, -1, :]), dim=1)
