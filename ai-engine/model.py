import math

import torch
import torch.nn as nn


class TimeSeriesTransformer(nn.Module):
    """
    Transformer model for time series forecasting.
    """

    def __init__(self, input_dim, d_model=128, nhead=4, num_layers=3, output_dim=3):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, d_model), nn.LayerNorm(d_model), nn.GELU()
        )
        self.pos_encoder = nn.Parameter(torch.zeros(1, 500, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(d_model, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """
        Forward pass for the model.
        """
        x = self.embedding(x) + self.pos_encoder[:, : x.size(1), :]
        x = self.transformer(x)
        return self.softmax(self.decoder(x[:, -1, :]))


def init_weights(m):
    """
    Initialize weights for the model.
    """
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
