import asyncio

import config
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from features import fetcher, processor
from model import TimeSeriesTransformer
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset


async def train():
    print(f"🚀 Starting Training {config.settings.SYMBOL_YF}...")
    raw = await fetcher.fetch_market_data(
        config.settings.SYMBOL_YF, period="2y", interval=config.settings.TIMEFRAME
    )
    df, feats = processor.process(raw, is_training=True)

    # Labeling (ATR based)
    labels, prices, atrs = [], df["close"].values, df["ATRr_14"].values
    lookahead = config.settings.PREDICT_WINDOW

    for i in range(len(prices) - lookahead):
        change = prices[i + lookahead] - prices[i]
        threshold = atrs[i] * 1.0
        if change > threshold:
            labels.append(0)  # BUY
        elif change < -threshold:
            labels.append(1)  # SELL
        else:
            labels.append(2)  # HOLD

    X, y = [], []
    seq_len = config.settings.SEQ_LEN
    for i in range(seq_len, len(labels)):
        X.append(feats[i - seq_len : i])
        y.append(labels[i])

    X, y = np.array(X), np.array(y)
    weights = compute_class_weight("balanced", classes=np.unique(y), y=y)

    model = TimeSeriesTransformer(input_dim=4)
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(weights))
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    loader = DataLoader(
        TensorDataset(torch.FloatTensor(X), torch.LongTensor(y)),
        batch_size=32,
        shuffle=True,
    )
    model.train()

    for epoch in range(20):
        total_loss = 0
        for bx, by in loader:
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} Loss: {total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), config.settings.MODEL_FILE)
    print("✅ Model Saved.")


if __name__ == "__main__":
    asyncio.run(train())
