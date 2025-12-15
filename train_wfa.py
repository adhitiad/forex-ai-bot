import asyncio
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset

# Import modul internal
from config import settings
from features import fetcher, processor
from model import TimeSeriesTransformer

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("WFA")

# --- KONFIGURASI WFA ---
TRAIN_MONTHS = 12  # Periode Training (misal: 1 tahun ke belakang)
TEST_MONTHS = 1  # Periode Testing (misal: 1 bulan ke depan)
STEP_MONTHS = 1  # Geser jendela setiap 1 bulan
EPOCHS_PER_FOLD = 15  # Epoch per fold (lebih sedikit karena retraining terus)


async def run_walk_forward():
    logger.info(
        f"🚀 Memulai Walk-Forward Analysis ({TRAIN_MONTHS} bln Train -> {TEST_MONTHS} bln Test)..."
    )

    # 1. Ambil Data History Panjang (Misal 4 Tahun)
    df = await fetcher.fetch_market_data(days=1460)
    if df.empty:
        return

    # 2. Preprocessing Global
    # Kita scale seluruh data dulu, ATAU idealnya scale per fold (tapi untuk simplifikasi kita scale global)
    df, scaled = processor.process(df, is_training=True)

    # 3. Sequencing Data
    X, y = [], []
    prediction_window = 4
    THRESHOLD = 0.0015

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

    # Konversi Index ke Tanggal untuk pelaporan
    dates = df.index[settings.SEQ_LEN : len(scaled) - prediction_window]

    # --- LOGIKA SLIDING WINDOW ---
    # Hitung jumlah sampel per bulan (asumsi 1 bulan = 22 hari trading * 24 jam = ~500 candle H1)
    # Ini estimasi kasar, sebaiknya pakai index tanggal asli
    SAMPLES_PER_MONTH = 22 * 24
    train_size = TRAIN_MONTHS * SAMPLES_PER_MONTH
    test_size = TEST_MONTHS * SAMPLES_PER_MONTH
    step_size = STEP_MONTHS * SAMPLES_PER_MONTH

    start_index = 0
    fold_results = []

    fold_count = 1

    while start_index + train_size + test_size < len(X):
        # Definisikan Slice Data
        train_start = start_index
        train_end = start_index + train_size
        test_end = train_end + test_size

        X_train = X[train_start:train_end]
        y_train = y[train_start:train_end]
        X_test = X[train_end:test_end]
        y_test = y[train_end:test_end]

        # Ambil label tanggal untuk laporan
        period_str = f"{dates[train_end].strftime('%Y-%m')} s/d {dates[test_end].strftime('%Y-%m')}"

        # Skip jika data terlalu sedikit (misal libur panjang)
        if len(X_train) < 100 or len(X_test) < 10:
            start_index += step_size
            continue

        # Setup Model Baru untuk Fold ini (Reset Bobot)
        model = TimeSeriesTransformer(input_dim=4, output_dim=3)

        # Class Weights
        classes = np.unique(y_train)
        if len(classes) < 3:
            # Handle kasus jarang jika di bulan tertentu tidak ada sinyal BUY/SELL
            start_index += step_size
            continue

        class_weights = compute_class_weight("balanced", classes=classes, y=y_train)
        full_weights = torch.ones(3)
        for cls, w in zip(classes, class_weights):
            full_weights[cls] = w

        criterion = nn.CrossEntropyLoss(weight=full_weights)
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Dataloader
        train_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)),
            batch_size=64,
            shuffle=True,
        )
        test_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test)),
            batch_size=64,
            shuffle=False,
        )

        # --- TRAINING FOLD ---
        model.train()
        for _ in range(EPOCHS_PER_FOLD):
            for bx, by in train_loader:
                optimizer.zero_grad()
                out = model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()

        # --- EVALUASI FOLD ---
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for bx, by in test_loader:
                out = model(bx)
                pred = out.argmax(dim=1)
                correct += (pred == by).sum().item()
                total += by.size(0)

        acc = (correct / total) * 100
        logger.info(
            f"📁 Fold {fold_count:02d} | Periode: {period_str} | Acc: {acc:.2f}%"
        )

        fold_results.append({"fold": fold_count, "period": period_str, "accuracy": acc})

        # Geser Jendela
        start_index += step_size
        fold_count += 1

    # --- LAPORAN AKHIR ---
    df_res = pd.DataFrame(fold_results)
    avg_acc = df_res["accuracy"].mean()
    std_dev = df_res["accuracy"].std()

    logger.info("\n" + "=" * 40)
    logger.info(f"📊 HASIL WALK-FORWARD ANALYSIS")
    logger.info(f"=" * 40)
    logger.info(f"Rata-rata Akurasi : {avg_acc:.2f}%")
    logger.info(f"Stabilitas (StdDev): {std_dev:.2f} (Semakin kecil semakin bagus)")
    logger.info(f"Performa Terburuk : {df_res['accuracy'].min():.2f}%")
    logger.info(f"Performa Terbaik  : {df_res['accuracy'].max():.2f}%")
    logger.info(f"=" * 40)

    # Tips untuk User
    if std_dev > 5.0:
        logger.warning(
            "⚠️ Peringatan: Model tidak stabil! Akurasi fluktuatif antar bulan."
        )
        logger.warning("Saran: Perbanyak data training atau kurangi learning rate.")
    else:
        logger.info("✅ Model Stabil. Aman untuk dideploy.")


if __name__ == "__main__":
    asyncio.run(run_walk_forward())
