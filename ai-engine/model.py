import logging

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AI-Engine")


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
        return self.decoder(x[:, -1, :])


def init_weights(m):
    """
    Initialize weights for the model.
    """
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)


def train_model(model, features, labels, epochs=15, batch_size=32, lr=0.001):
    """
    Train the TimeSeriesTransformer model.

    Args:
        model (TimeSeriesTransformer): The model to train.
        features (np.ndarray): Input features of shape (N, seq_len, input_dim).
        labels (np.ndarray): Target labels of shape (N,).
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        lr (float): Learning rate.

    Returns:
        TimeSeriesTransformer: The trained model.
    """
    # Convert to tensors
    features = torch.tensor(features, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)

    dataset = TensorDataset(features, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.apply(init_weights)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_features, batch_labels in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")

    return model
