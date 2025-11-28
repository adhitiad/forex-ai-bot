# ai-engine/config.py

# --- ASSET CONFIG ---
# Gunakan 'XAUUSD=X' untuk Gold Spot di Yahoo Finance
SYMBOL_YF = "GBPUSD=X"
SYMBOL_BINANCE = (
    "XAUUSDT"  # Simbol untuk eksekusi (jika crypto gold PAXG) atau Forex Broker API
)

# --- MODEL CONFIG ---
SEQ_LEN = 30  # Melihat 30 candle ke belakang
PREDICT_WINDOW = 4  # Memprediksi tren 4 candle ke depan
TIMEFRAME = "1h"  # Timeframe H1
TRAIN_PERIOD = "2y"  # Data latih 2 tahun
MODEL_FILE = "trained_model.pth"

# --- TRADING PARAMS (BACKTEST & LIVE) ---
INITIAL_BALANCE = 10000  # Saldo Awal USD
LOT_SIZE = 0.01  # Ukuran Lot
LEVERAGE = 100  # Asumsi Leverage (Standard Forex/Gold)
CONTRACT_SIZE = 100  # 1 Lot Gold = 100 Oz (Standard)

# --- RISK MANAGEMENT ---
# Target Profit & Stop Loss dalam Persentase Harga Entry
# Contoh: Entry 2000. TP 30% = Target 2600.
TAKE_PROFIT_PCT = 0.30  # 30%
STOP_LOSS_PCT = 0.25  # 25%
CONFIDENCE_THRESHOLD = 0.60  # Minimal yakin 60% baru entry

# --- INFLUXDB CONFIG ---
INFLUX_BUCKET = "forex_ai_bot"
