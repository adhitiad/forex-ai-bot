import asyncio
import json
import logging

import numpy as np
import redis.asyncio as redis
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset

# Import modul internal
from cloud_manager import cloud_manager
from config import settings
from features import fetcher, processor
from model import TimeSeriesTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Trainer")


async def train():
    logger.info(f"🚀 Training for {settings.ACTIVE_SYMBOLS[0]}...")

    # 1. Ambil Data (Misal 2 Tahun)
    df = await fetcher.fetch_market_data(days=730)
    if df.empty:
        logger.error("❌ Data kosong. Training dibatalkan.")
        return

    # 2. Preprocessing (Fit scaler hanya pada data awal, idealnya split dulu baru scale)
    # Untuk simplifikasi, kita gunakan logic processor yang sudah diperbaiki (Cek step sebelumnya)
    df, scaled = processor.process(df, is_training=True)

    X, y = [], []
    prediction_window = settings.PREDICTION_WINDOW  # Default 15 candle ke depan
    THRESHOLD = settings.VOLATILITY_THRESHOLD  # Default 0.0025

    # 3. Labeling
    for i in range(len(scaled) - settings.SEQ_LEN - prediction_window):
        # Input: Sequence candle terakhir
        X.append(scaled[i : i + settings.SEQ_LEN])

        # Target: Harga masa depan
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

    # 4. SPLIT TRAIN/VAL (SEKUENSIAL - WAJIB!)
    # Jangan di-shuffle sebelum split!
    split_idx = int(len(X) * 0.8)  # 80% Training, 20% Validasi

    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    logger.info(f"Data Split: Train={len(X_train)} | Val={len(X_val)}")

    # Hitung bobot kelas hanya dari data training
    class_weights = compute_class_weight(
        "balanced", classes=np.unique(y_train), y=y_train
    )
    full_weights = torch.tensor(class_weights, dtype=torch.float)

    # Dataset & Loader
    # Shuffle=True HANYA BOLEH di Training set setelah di-split
    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)),
        batch_size=settings.BATCH_SIZE,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val)),
        batch_size=settings.BATCH_SIZE,
        shuffle=False,
    )

    model = TimeSeriesTransformer(input_dim=4, output_dim=3)
    criterion = nn.CrossEntropyLoss(weight=full_weights)
    optimizer = optim.Adam(model.parameters(), lr=settings.LEARNING_RATE)

    # 5. Training Loop dengan Evaluasi
    best_acc = 0.0

    for epoch in range(20):
        model.train()
        total_loss = 0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            out = model(batch_X)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Evaluasi (Validation)
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for val_X, val_y in val_loader:
                out = model(val_X)
                preds = torch.argmax(out, dim=1)
                correct += (preds == val_y).sum().item()
                total += val_y.size(0)

        val_acc = correct / total
        logger.info(
            f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f} | Val Acc: {val_acc*100:.2f}%"
        )

        # Save Best Model Only
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), settings.MODEL_FILE)
            logger.info("✅ New Best Model Saved!")

    # 6. Upload & Notify
    await asyncio.to_thread(cloud_manager.upload_model)

    # Notify System
    try:
        r = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            password=settings.REDIS_PASSWORD,
        )
        await r.publish(
            settings.CHANNEL_SYSTEM, json.dumps({"event": "TRAINING_COMPLETED"})
        )
        await r.close()
    except:
        pass


if __name__ == "__main__":
    asyncio.run(train())
