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
logger = logging.getLogger("Trainer")


async def train():
    logger.info(f"🚀 Training for {settings.YFINANCE_SYMBOL}...")
    df = await fetcher.fetch_market_data(days=730)  # 2 Tahun data

    if df.empty:
        return
    df, scaled = processor.process(df, is_training=True)

    X, y = [], []
    prediction_window = 4
    THRESHOLD = 0.0015  # 0.15% (Sekitar 15-20 Pips EURUSD)

    for i in range(len(scaled) - settings.SEQ_LEN - prediction_window):
        X.append(scaled[i : i + settings.SEQ_LEN])
        curr = df.iloc[i + settings.SEQ_LEN]["close"]
        fut = df.iloc[i + settings.SEQ_LEN + prediction_window]["close"]
        diff = (fut - curr) / curr

        if diff > THRESHOLD:
            y.append(1)  # BUY
        elif diff < -THRESHOLD:
            y.append(2)  # SELL
        else:
            y.append(0)  # HOLD

    X, y = np.array(X), np.array(y)

    class_weights = compute_class_weight("balanced", classes=np.unique(y), y=y)
    full_weights = np.ones(3)
    for cls, w in zip(np.unique(y), class_weights):
        full_weights[cls] = w

    dataset = TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = TimeSeriesTransformer(input_dim=4, output_dim=3)
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(full_weights))
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(20):
        total_loss = 0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            out = model(batch_X)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            accuracy = (out.argmax(dim=1) == batch_y).float().mean().item()
        logger.info(
            f"Epoch {epoch+1}/20 | Loss: {total_loss/len(loader):.4f} | Accuracy: {accuracy*100:.2f}%"
        )

    torch.save(model.state_dict(), settings.MODEL_FILE)
    logger.info(f"✅ Saved to {settings.MODEL_FILE}")


if __name__ == "__main__":
    asyncio.run(train())
