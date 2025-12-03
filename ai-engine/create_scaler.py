import asyncio
import os

from config import settings
from features import fetcher, processor


async def create_scaler():
    print("Creating scaler...")
    raw = await fetcher.fetch_market_data(
        settings.SYMBOL_YF, period="1mo", interval="1h"
    )
    print(f"Raw data shape: {raw.shape}")
    df, feats = processor.process(raw, is_training=True)
    print("Scaler created successfully!")
    print(f"Processed data shape: {df.shape}")
    print(f"Features shape: {feats.shape}")
    print(f"Scaler path: {processor.path}")
    print(f"Scaler file exists: {os.path.exists(processor.path)}")


if __name__ == "__main__":
    asyncio.run(create_scaler())
