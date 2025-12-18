import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # Membuat matriks posisi (tetap sama seperti sebelumnya)
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

        # Menggunakan batch_first=True agar dimensi input (Batch, Seq, Feature)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, batch_first=True, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)

        self.decoder = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.1),  # Tambahkan Dropout untuk mencegah Overfitting
            nn.Linear(64, output_dim),
        )

    def forward(self, x):
        # 1. Embedding & Positional Encoding
        x = self.pos_encoder(self.embedding(x))

        # 2. Transformer Pass
        x = self.transformer(x)

        # 3. ARCHITECTURE FIX: Global Average Pooling
        # Sebelumnya: x[:, -1, :] (Hanya melihat candle terakhir)
        # Sekarang: x.mean(dim=1) (Melihat rata-rata konteks dari 30 candle)
        x = x.mean(dim=1)

        # 4. MATH FIX: Hapus Softmax
        # Kembalikan Logits murni untuk kestabilan training
        return self.decoder(x)


class LSTMModel(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, num_layers=2, output_dim=3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x shape: (Batch, Seq, Feature)
        out, _ = self.lstm(x)
        # Ambil output dari langkah terakhir saja
        out = out[:, -1, :]
        return self.fc(out)
