import asyncio
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.utils.class_weight import compute_class_weight

from config import settings
from features import fetcher, processor, DataFetcher, FeatureEngineer
from yfinance_fetcher import yfinance_fetcher
from model import TimeSeriesTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Trainer")


async def train():
    symbol = (
        settings.YFINANCE_SYMBOL if settings.USE_YFINANCE else settings.ACTIVE_SYMBOL
    )
    logger.info(f"🚀 Training Model for {symbol}...")
    logger.info(
        f"📊 Data Source: {'Yahoo Finance' if settings.USE_YFINANCE else 'CCXT Exchange'}"
    )

    # 1. Download Data (Setahun terakhir)
    if settings.USE_YFINANCE:
        # Gunakan Yahoo Finance (tidak diblokir)
        df = await yfinance_fetcher.fetch_market_data(
            symbol=settings.YFINANCE_SYMBOL, days=365, interval="1h"
        )
    else:
        # Gunakan CCXT Exchange
        df = await fetcher.fetch_market_data(days=365)

    if df.empty:
        logger.error("❌ No data fetched. Check internet connection or symbol.")
        return

    # 2. Process & Scale
    df, scaled = processor.process(df, is_training=True)

    if len(scaled) < 100:
        logger.error("❌ Not enough data points for training.")
        return

    # 3. Labeling (Menentukan Target Buy/Sell)
    X, y = [], []
    prediction_window = 4  # Prediksi 4 candle ke depan

    # Threshold Profit (Sesuaikan dengan volatilitas Crypto)
    # Jika harga naik > 1.5% dalam 4 jam -> BUY
    THRESHOLD = 0.015

    for i in range(len(scaled) - settings.SEQ_LEN - prediction_window):
        # Input: Sequence candle terakhir
        X.append(scaled[i : i + settings.SEQ_LEN])

        # Target Calculation
        current_close = df.iloc[i + settings.SEQ_LEN]["close"]
        future_close = df.iloc[i + settings.SEQ_LEN + prediction_window]["close"]

        diff = (future_close - current_close) / current_close

        if diff > THRESHOLD:
            y.append(1)  # BUY
        elif diff < -THRESHOLD:
            y.append(2)  # SELL
        else:
            y.append(0)  # HOLD

    X = np.array(X)
    y = np.array(y)

    logger.info(f"📊 Dataset size: {len(X)} samples")
    unique, counts = np.unique(y, return_counts=True)
    logger.info(f"⚖️ Class Distribution: {dict(zip(unique, counts))}")

    # 4. Convert to PyTorch Tensor
    # Class Weighting agar model tidak bias ke HOLD (0)
    class_weights = compute_class_weight("balanced", classes=np.unique(y), y=y)
    weights_tensor = torch.FloatTensor(class_weights)

    dataset = TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 5. Model Setup
    model = TimeSeriesTransformer(input_dim=4, output_dim=3)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    epochs = 20

    logger.info("🔥 Starting Training Loop...")

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

        logger.info(
            f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(loader):.4f} | Acc: {100*correct/total:.2f}%"
        )

    # 6. Save Model
    torch.save(model.state_dict(), settings.MODEL_FILE)
    logger.info(f"✅ Model saved to {settings.MODEL_FILE}")


if __name__ == "__main__":
    asyncio.run(train())
