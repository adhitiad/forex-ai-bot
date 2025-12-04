import asyncio
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from config import settings
from features import fetcher, processor
from model import TimeSeriesTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Trainer")


async def train():
    logger.info("🚀 Training...")
    fetcher.update_config("FOREX", settings.SYMBOL_FOREX)
    raw = await fetcher.fetch_market_data(period="1y", limit=2000)
    df, scaled = processor.process(raw, is_training=True)

    X, y = [], []
    for i in range(len(scaled) - settings.SEQ_LEN - 4):
        X.append(scaled[i : i + settings.SEQ_LEN])
        diff = (
            df.iloc[i + settings.SEQ_LEN + 4]["close"]
            - df.iloc[i + settings.SEQ_LEN]["close"]
        ) / df.iloc[i + settings.SEQ_LEN]["close"]
        y.append(1 if diff > 0.002 else 2 if diff < -0.002 else 0)

    dataset = TensorDataset(
        torch.FloatTensor(np.array(X)), torch.LongTensor(np.array(y))
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = TimeSeriesTransformer(input_dim=4)
    opt = optim.Adam(model.parameters(), lr=0.001)
    crit = nn.CrossEntropyLoss()
    logger.info("🚀 Training Started")

    model.train()

    for e in range(5):
        for bx, by in loader:
            opt.zero_grad()
            loss = crit(model(bx), by)
            loss.backward()
            opt.step()
        print(f"Epoch {e+1} Done")

    torch.save(model.state_dict(), settings.MODEL_FILE)
    logger.info("✅ Model Saved")


if __name__ == "__main__":
    asyncio.run(train())
