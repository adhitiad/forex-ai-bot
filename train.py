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

# Import internal
from cloud_manager import cloud_manager
from config import settings
from features import fetcher, processor
from model import TimeSeriesTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Trainer")


async def train():
    logger.info(f"🚀 Training for {settings.ACTIVE_SYMBOLS[0]}...")

    # 1. Ambil Data
    df = await fetcher.fetch_market_data(days=730)
    if df.empty:
        logger.error("❌ Data kosong.")
        return

    # 2. Add Indicators (BELUM DI-SCALE)
    df = processor.add_indicators(df)
    if df.empty:
        return

    # 3. Split Data (Training 80%, Val 20%) - TIME SERIES SPLIT (Tanpa Shuffle)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    val_df = df.iloc[split_idx:].copy()

    logger.info(f"Data Split: Train Rows={len(train_df)} | Val Rows={len(val_df)}")

    # 4. Fit Scaler ONLY on Training Data
    processor.fit_scaler(train_df)

    # 5. Transform Data
    train_scaled = processor.transform(train_df)
    val_scaled = processor.transform(val_df)

    # 6. Fungsi Helper membuat Sequence
    def create_sequences(data_scaled, original_close, seq_len, pred_window, threshold):
        X, y = [], []
        # original_close harus align dengan data_scaled
        # Kita butuh index reset agar iloc mudah
        closes = original_close.values

        for i in range(len(data_scaled) - seq_len - pred_window):
            # Input
            X.append(data_scaled[i : i + seq_len])

            # Target (Melihat masa depan dari sequence terakhir)
            curr = closes[i + seq_len]
            fut = closes[i + seq_len + pred_window]
            diff = (fut - curr) / curr

            if diff > threshold:
                y.append(1)  # BUY
            elif diff < -threshold:
                y.append(2)  # SELL
            else:
                y.append(0)  # HOLD

        return np.array(X), np.array(y)

    # Buat Sequence untuk Train dan Val terpisah
    logger.info("⏳ Creating Sequences...")
    X_train, y_train = create_sequences(
        train_scaled,
        train_df["close"],
        settings.SEQ_LEN,
        settings.PREDICTION_WINDOW,
        settings.VOLATILITY_THRESHOLD,
    )

    X_val, y_val = create_sequences(
        val_scaled,
        val_df["close"],
        settings.SEQ_LEN,
        settings.PREDICTION_WINDOW,
        settings.VOLATILITY_THRESHOLD,
    )

    if len(X_train) == 0 or len(X_val) == 0:
        logger.error("❌ Sequence generation failed (not enough data).")
        return

    # 7. Class Weights & DataLoader
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    # Pastikan weights mencakup 3 kelas (0,1,2)
    full_weights = torch.ones(3)
    for c, w in zip(classes, weights):
        full_weights[c] = w

    logger.info(f"Class Weights: {full_weights}")

    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)),
        batch_size=settings.BATCH_SIZE,
        shuffle=True,  # Shuffle OK disini karena sudah bentuk window
    )
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val)),
        batch_size=settings.BATCH_SIZE,
        shuffle=False,
    )

    # 8. Model Setup
    model = TimeSeriesTransformer(input_dim=4, output_dim=3)
    criterion = nn.CrossEntropyLoss(weight=full_weights)
    optimizer = optim.Adam(model.parameters(), lr=settings.LEARNING_RATE)

    # 9. Training Loop
    best_acc = 0.0
    patience = 5
    no_improve = 0

    for epoch in range(30):  # Naikkan epoch
        model.train()
        total_loss = 0

        for bx, by in train_loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for vx, vy in val_loader:
                out = model(vx)
                pred = torch.argmax(out, dim=1)
                correct += (pred == vy).sum().item()
                total += vy.size(0)

        val_acc = correct / total
        logger.info(
            f"Ep {epoch+1} | Loss: {total_loss/len(train_loader):.4f} | Val Acc: {val_acc*100:.2f}%"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), settings.MODEL_FILE)
            no_improve = 0
            logger.info("✅ Best Model Saved")
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info("Early stopping triggered.")
                break

    # 10. Upload & Notify
    await asyncio.to_thread(cloud_manager.upload_model)

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
