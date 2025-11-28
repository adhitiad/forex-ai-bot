import asyncio

import config  # <--- Menggunakan Config Pusat
import numpy as np
import pandas as pd
import torch
from features import fetcher, process_features
from model import TimeSeriesTransformer


async def run_backtest():
    """
    Run the backtesting process.
    """
    print(f"📊 STARTING BACKTEST: {config.SYMBOL_YF}")
    print(
        f"   Lot: {config.LOT_SIZE} | TP: {config.TAKE_PROFIT_PCT*100}% | SL: {config.STOP_LOSS_PCT*100}%"
    )

    # 1. Fetch Data (60 Hari Terakhir untuk Uji Coba)
    raw_df = await fetcher.fetch_market_data(
        config.SYMBOL_YF, period="60d", interval=config.TIMEFRAME
    )
    full_df, feature_matrix = process_features(raw_df)

    # 2. Load Model
    model = TimeSeriesTransformer(input_dim=feature_matrix.shape[1])
    try:
        model.load_state_dict(torch.load(config.MODEL_FILE))
        model.eval()
    except FileNotFoundError:
        print("❌ Model belum dilatih! Jalankan train.py dulu.")
        return

    # 3. Variables
    balance = config.INITIAL_BALANCE
    position_type = None  # 'BUY' or 'SELL' or None
    entry_price = 0.0
    trades = []

    # Loop Candle
    for i in range(config.SEQ_LEN, len(feature_matrix)):
        # Data saat ini
        current_price = raw_df.iloc[i]["close"]
        high_price = raw_df.iloc[i]["high"]
        low_price = raw_df.iloc[i]["low"]
        timestamp = raw_df.index[i]

        # --- A. LOGIKA MANAJEMEN POSISI (TP/SL) ---
        if position_type is not None:
            pnl_pct = 0.0
            pnl_usd = 0.0
            close_signal = False
            exit_reason = ""
            exec_price = current_price

            # Hitung Profit/Loss Floating
            if position_type == "BUY":
                # Cek apakah High menyentuh TP atau Low menyentuh SL dalam candle ini
                change_high = (high_price - entry_price) / entry_price
                change_low = (low_price - entry_price) / entry_price

                if change_high >= config.TAKE_PROFIT_PCT:
                    exit_reason = "TP Hit (30%)"
                    close_signal = True
                    # Asumsi keluar di harga TP
                    exec_price = entry_price * (1 + config.TAKE_PROFIT_PCT)
                elif change_low <= -config.STOP_LOSS_PCT:
                    exit_reason = "SL Hit (25%)"
                    close_signal = True
                    exec_price = entry_price * (1 - config.STOP_LOSS_PCT)

            # (Logic Short/SELL omitted for brevity, similar to BUY inverted)

            # Eksekusi Close
            if close_signal:
                # Rumus PnL Forex/Gold: (PriceDiff) * ContractSize * Lot
                price_diff = exec_price - entry_price
                pnl_usd = price_diff * config.CONTRACT_SIZE * config.LOT_SIZE

                balance += pnl_usd
                trades.append(
                    {
                        "time": timestamp,
                        "type": "CLOSE_" + position_type,
                        "price": exec_price,
                        "pnl": pnl_usd,
                        "reason": exit_reason,
                        "balance": balance,
                    }
                )
                position_type = None
                continue  # Skip logic entry di candle yang sama saat exit

        # --- B. LOGIKA ENTRY AI ---
        # Hanya entry jika tidak ada posisi
        if position_type is None:
            tensor_in = torch.FloatTensor(
                feature_matrix[i - config.SEQ_LEN : i]
            ).unsqueeze(0)
            with torch.no_grad():
                probs = model(tensor_in).numpy()[0]  # [Buy, Sell, Hold]

            if probs[0] > config.CONFIDENCE_THRESHOLD:  # BUY SIGNAL
                position_type = "BUY"
                entry_price = current_price
                trades.append(
                    {
                        "time": timestamp,
                        "type": "OPEN_BUY",
                        "price": entry_price,
                        "conf": probs[0],
                        "balance": balance,
                    }
                )

            # (Logic Sell/Short bisa ditambahkan disini jika akun support shorting)
            if probs[1] > config.CONFIDENCE_THRESHOLD:  # SELL SIGNAL
                position_type = "SELL"
                entry_price = current_price
                trades.append(
                    {
                        "time": timestamp,
                        "type": "OPEN_SELL",
                        "price": entry_price,
                        "conf": probs[1],
                        "balance": balance,
                    }
                )

    # Report
    print(f"\n🏁 BACKTEST SELESAI. Saldo Akhir: ${balance:.2f}")
    print(f"   Total Trades: {len(trades)}")
    for t in trades[-5:]:  # Show last 5
        print(t)


if __name__ == "__main__":
    asyncio.run(run_backtest())
