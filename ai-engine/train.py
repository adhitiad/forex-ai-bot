import asyncio
import os
import time

# Import Integrasi
import config
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from features import fetcher, process_features
from model import TimeSeriesTransformer, init_weights
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset


class TradingDataset(Dataset):
    """Dataset for Trading Data"""

    def __init__(self, x, y):
        self.X = torch.FloatTensor(x)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# --- TRAINING LOGIC ---
async def run_training():
    """Run the training process."""
    print(f"🌟 Starting Training for {config.SYMBOL_YF}...")

    # 1. Fetch Data XAUUSD
    raw_df = await fetcher.fetch_market_data(
        symbol=config.SYMBOL_YF, period=config.TRAIN_PERIOD, interval=config.TIMEFRAME
    )

    if raw_df.empty:
        print("❌ Error: Gagal mengambil data ", config.SYMBOL_YF)
        return

    # 2. Process Features
    df, feature_matrix = process_features(raw_df)

    # 3. Labeling
    # Logic: Label BUY jika harga naik, SELL jika turun
    labels = []
    prices = df["close"].values
    lookahead = config.PREDICT_WINDOW

    for i in range(len(prices) - lookahead):
        curr = prices[i]
        fut = prices[i + lookahead]
        change = (fut - curr) / curr

        # Ambang batas volatilitas Gold (lebih kecil dari crypto)
        threshold = 0.002  # 0.2% move

        if change > threshold:
            labels.append(0)  # BUY
        elif change < -threshold:
            labels.append(1)  # SELL
        else:
            labels.append(2)  # HOLD

    # 4. Sliding Window
    X, y = [], []
    for i in range(config.SEQ_LEN, len(labels)):
        X.append(feature_matrix[i - config.SEQ_LEN : i])
        y.append(labels[i])

    X = np.array(X)
    y = np.array(y)

    # 5. Setup & Train Model
    # Hitung bobot agar model tidak bias ke HOLD
    class_weights = compute_class_weight("balanced", classes=np.unique(y), y=y)
    weights_tensor = torch.FloatTensor(class_weights)

    model = TimeSeriesTransformer(input_dim=X.shape[2], d_model=128)
    model.apply(init_weights)

    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = optim.AdamW(model.parameters(), lr=0.0001)

    # Simple Training Loop (Tanpa validasi kompleks demi ringkas)
    dataset = TradingDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    model.train()
    print("🚀 Training Neural Network...")

    start_time = time.perf_counter()
    for epoch in range(35):  # 35 Epochs
        total_loss = 0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            out = model(batch_X)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(
            f"   Epoch {epoch+1} Loss: {total_loss/len(loader):.5f} time {time.perf_counter() - start_time:.2f}s"
        )

    # 6. Save
    os.makedirs(os.path.dirname(config.MODEL_FILE), exist_ok=True)
    torch.save(model.state_dict(), config.MODEL_FILE)
    print(f"✅ Model saved to {config.MODEL_FILE}")


if __name__ == "__main__":
    asyncio.run(run_training())
