import asyncio
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset

from config import settings
from features import fetcher, processor
from model import TimeSeriesTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ForexTrainer")


async def train():
    logger.info(f"🚀 Training Forex Model for {settings.ACTIVE_SYMBOL}...")

    # 1. Fetch Data Forex Jangka Panjang (2 Tahun)
    fetcher.update_config("FOREX", settings.ACTIVE_SYMBOL)
    raw = await fetcher.fetch_market_data(period="2y", limit=None)  # Limit None = Max

    if raw.empty:
        logger.error("❌ Failed to fetch Forex data via Yahoo Finance.")
        return

    # Download data simpan ke dataset pada folder "data"
    processor.save_to_dataset(df, "data")

    logger.info(f"📊 Downloaded {len(raw)} candles.")

    # 2. Process & Scale
    df, scaled = processor.process(raw, is_training=True)

    # 3. Labeling Khusus Forex
    X, y = [], []
    prediction_window = 4

    for i in range(len(scaled) - settings.SEQ_LEN - prediction_window):
        X.append(scaled[i : i + settings.SEQ_LEN])

        current_close = df.iloc[i + settings.SEQ_LEN]["close"]
        future_close = df.iloc[i + settings.SEQ_LEN + prediction_window]["close"]

        diff = (future_close - current_close) / current_close

        # Threshold Forex lebih kecil daripada Crypto
        # 0.001 = 0.1% (sekitar 10 pips). Jika gerak > 10 pips, anggap tren.
        if diff > 0.001:
            y.append(1)  # BUY
        elif diff < -0.001:
            y.append(2)  # SELL
        else:
            y.append(0)  # HOLD

    X = np.array(X)
    y = np.array(y)

    # 4. Class Balancing (Agar AI tidak bias ke "HOLD")
    classes = np.unique(y)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    class_weights = torch.FloatTensor(weights)
    logger.info(f"⚖️ Class Weights: {weights} (Hold, Buy, Sell)")

    # Convert to Tensor
    dataset = TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))
    # Batch size diperbesar agar training lebih stabil
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    # 5. Model Setup
    model = TimeSeriesTransformer(input_dim=4, output_dim=3)
    opt = optim.Adam(model.parameters(), lr=0.0001)  # Learning rate diperkecil
    # Masukkan weight ke loss function
    crit = nn.CrossEntropyLoss(weight=class_weights)

    model.train()
    logger.info(f"🔥 Start Training on {len(X)} sequences")

    # Epoch Loop
    epochs = 20  # Tingkatkan epoch
    for e in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        for bx, by in loader:
            opt.zero_grad()
            outputs = model(bx)  # Output: Logits
            loss = crit(outputs, by)
            loss.backward()
            opt.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += by.size(0)
            correct += (predicted == by).sum().item()

        avg_loss = total_loss / len(loader)
        acc = 100 * correct / total
        print(f"Epoch {e+1}/{epochs} | Loss: {avg_loss:.4f} | Acc: {acc:.2f}%")

    # 6. Save Model
    torch.save(model.state_dict(), settings.MODEL_FILE)
    logger.info("✅ Model Saved Successfully")


if __name__ == "__main__":
    asyncio.run(train())
