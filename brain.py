import asyncio
import datetime
import json
import logging
from collections import deque

import pandas as pd
import torch
import torch.nn.functional as F  # Diperlukan untuk softmax saat inferensi

from config import settings
from database import TradeLog, get_db

# Import fetcher untuk warm-up data
from features import fetcher, processor
from model import TimeSeriesTransformer
from state_manager import state_manager
from stream_manager import streamor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Brain")


class Brain:
    def __init__(self):
        # OPTIMIZATION: Buffer diperbesar ke 300 agar cukup untuk indikator lag (EMA/RSI)
        # 100 terlalu sedikit jika ada dropna() di processor
        self.buffer = deque(maxlen=300)

        self.model = None
        self.latest_sentiment = 0.0
        self.db_gen = get_db()
        self.db = next(self.db_gen)

    async def init(self):
        """Inisialisasi Model, Scaler, dan Data Historis"""
        # 1. Load Scaler
        if not processor.load_scaler():
            logger.warning("⚠️ Scaler not found. Training might be needed.")
            # Tidak return, lanjut warm-up siapa tahu training jalan paralel

        # 2. Load Model
        self.model = TimeSeriesTransformer(input_dim=4)
        try:
            self.model.load_state_dict(torch.load(settings.MODEL_FILE))
            self.model.eval()
            logger.info("🧠 Model Loaded Successfully")
        except FileNotFoundError:
            logger.error("❌ Model file not found!")
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")

        # 3. WARM-UP BUFFER (Critical Fix)
        # Mengambil data historis agar indikator teknikal tidak NaN di awal
        logger.info("⏳ Warming up buffer with historical data...")
        try:
            # Pastikan fetcher terkonfigurasi benar
            fetcher.update_config(settings.ASSET_TYPE, settings.ACTIVE_SYMBOL)

            # Ambil 300 data terakhir
            warmup_data = await fetcher.fetch_market_data(limit=300)

            if not warmup_data.empty:
                count = 0
                for ts, row in warmup_data.iterrows():
                    # Konversi format DataFrame ke format Dictionary (sesuai Redis payload)
                    candle = {
                        "timestamp": str(ts),
                        "open": float(row["open"]),
                        "high": float(row["high"]),
                        "low": float(row["low"]),
                        "close": float(row["close"]),
                        "volume": float(row["volume"]),
                        "source": settings.ASSET_TYPE,
                    }
                    self.buffer.append(candle)
                    count += 1
                logger.info(
                    f"✅ Buffer warmed up with {count} candles. Ready to trade!"
                )
            else:
                logger.warning(
                    "⚠️ Warm-up data empty. Bot will start 'blind' until buffer fills."
                )

        except Exception as e:
            logger.error(f"⚠️ Warm-up failed: {e}")

        # 4. Listen Sentiment
        asyncio.create_task(self.listen_sentiment())

    async def listen_sentiment(self):
        if not streamor.r:
            return
        pubsub = streamor.r.pubsub()
        await pubsub.subscribe("channel_sentiment")
        async for m in pubsub.listen():
            if m["type"] == "message":
                try:
                    data = json.loads(m["data"])
                    self.latest_sentiment = float(data.get("sentiment_score", 0.0))
                except:
                    pass

    async def manage_positions(self, current_price, current_time):
        """Logika untuk menutup posisi (TP/SL)"""
        active_pos = state_manager.get_active_position()
        if not active_pos:
            return

        action = None
        reason = None

        # Logika sederhana TP/SL
        # Gunakan Decimal di production untuk presisi, float untuk contoh ini
        if active_pos["side"] == "BUY":
            if current_price >= active_pos["tp"]:
                action, reason = "SELL", "TP Hit"
            elif current_price <= active_pos["sl"]:
                action, reason = "SELL", "SL Hit"
        elif active_pos["side"] == "SELL":
            if current_price <= active_pos["tp"]:
                action, reason = "BUY", "TP Hit"
            elif current_price >= active_pos["sl"]:
                action, reason = "BUY", "SL Hit"

        if action:
            # Simpan log ke DB
            pnl = (
                (current_price - active_pos["entry_price"])
                if active_pos["side"] == "BUY"
                else (active_pos["entry_price"] - current_price)
            )

            trade = TradeLog(
                symbol=active_pos["symbol"],
                action="CLOSE_" + active_pos["side"],
                entry_price=active_pos["entry_price"],  # Harga entry asli
                tp_price=current_price,  # Gunakan field ini untuk exit price sementara
                sl_price=0,
                status="CLOSED",
            )
            self.db.add(trade)
            self.db.commit()

            # Bersihkan State
            state_manager.clear_active_position()

            # Notifikasi
            msg = {
                "action": "CLOSE",
                "reason": reason,
                "price": current_price,
                "pnl": pnl,
                "timestamp": str(current_time),
            }
            if streamor:
                await streamor.push_signal(msg)
            logger.info(f"💰 Position Closed: {reason} | PnL: {pnl:.2f}")

    async def run(self):
        await self.init()

        logger.info("🚀 Brain Engine Started & Loop Running...")

        while True:
            # Consume data dari Redis
            if streamor and streamor.r:
                # Pastikan stream_manager.py sudah diupdate untuk tidak auto-ack!
                candles = await streamor.consume_market_data()
            else:
                candles = []

            if not candles:
                await asyncio.sleep(0.1)
                continue

            for c in candles:
                # 1. STALE DATA FILTER (Critical Fix)
                # Parse timestamp dari candle
                try:
                    # Handle format string ISO dari ingestor
                    c_time = pd.to_datetime(c["timestamp"])
                    # Cek selisih waktu dengan sekarang (UTC vs Local harus diperhatikan di production)
                    # Asumsi server time sinkron
                    lag = (datetime.datetime.now() - c_time).total_seconds()

                    # Jika data lebih tua dari 5 menit (300 detik), anggap kadaluarsa
                    if lag > 300:
                        logger.warning(f"⏩ Skipping stale data (Lag: {lag:.1f}s)")
                        continue
                except Exception as e:
                    logger.error(f"Timestamp error: {e}")

                current_price = c["close"]

                # 2. Manage Active Positions (TP/SL Check)
                await self.manage_positions(current_price, datetime.datetime.now())

                # 3. Update Buffer & Process
                self.buffer.append(c)
                if len(self.buffer) < settings.SEQ_LEN:
                    continue

                # Konversi buffer ke DataFrame untuk Feature Engineering
                df = pd.DataFrame(self.buffer)
                _, feats = processor.process(df)

                # Cek apakah fitur cukup (Pandas TA dropna bisa mengurangi baris)
                if len(feats) < settings.SEQ_LEN:
                    continue

                if not self.model:
                    continue

                # 4. Predict
                # Ambil sequence terakhir sesuai window model
                tensor = torch.FloatTensor(feats[-settings.SEQ_LEN :]).unsqueeze(0)

                with torch.no_grad():
                    logits = self.model(tensor)
                    # Karena kita menghapus softmax di model.py, kita pakai di sini untuk probabilitas
                    probs = F.softmax(logits, dim=1).numpy()[0]  # [Hold, Buy, Sell]

                # 5. Fusion Logic (AI + Sentiment)
                # Sentiment score (-1.0 to 1.0)
                sentiment_bias = 0.0
                if self.latest_sentiment > 0.5:
                    sentiment_bias = 0.1
                elif self.latest_sentiment < -0.5:
                    sentiment_bias = -0.1

                buy_conf = probs[1] + sentiment_bias
                sell_conf = (
                    probs[2] - sentiment_bias
                )  # Sentiment negatif menambah sell pressure

                # 6. Execute Entry
                action = "HOLD"
                if buy_conf > 0.75:  # Threshold diperketat
                    action = "BUY"
                elif sell_conf > 0.75:
                    action = "SELL"

                # Entry Rule: Hanya jika tidak ada posisi aktif
                if action != "HOLD" and not state_manager.get_active_position():
                    # Hitung TP/SL dinamis (Misal 1% TP, 0.5% SL)
                    tp_pct = settings.TAKE_PROFIT_PCT
                    sl_pct = settings.STOP_LOSS_PCT

                    if action == "BUY":
                        tp = current_price * (1 + tp_pct)
                        sl = current_price * (1 - sl_pct)
                    else:
                        tp = current_price * (1 - tp_pct)
                        sl = current_price * (1 + sl_pct)

                    # Simpan ke DB
                    trade = TradeLog(
                        symbol=c.get("source", "UNK"),
                        action=action,
                        entry_price=current_price,
                        tp_price=tp,
                        sl_price=sl,
                        status="OPEN",
                    )
                    self.db.add(trade)
                    self.db.commit()

                    # Set State (Persistent)
                    state_manager.set_active_position(
                        c.get("source"),
                        action,
                        current_price,
                        settings.LOT_SIZE,
                        tp,
                        sl,
                    )

                    # Push Signal ke WebSocket/Execution
                    if streamor:
                        await streamor.push_signal(
                            {
                                "action": action,
                                "entry_price": current_price,
                                "tp": tp,
                                "sl": sl,
                                "symbol": settings.ACTIVE_SYMBOL,
                                "status": "OPEN",
                                "timestamp": str(datetime.datetime.now()),
                            }
                        )
                    logger.info(
                        f"🚀 {action} Signal at {current_price} | Conf: {max(buy_conf, sell_conf):.2f}"
                    )


if __name__ == "__main__":
    asyncio.run(Brain().run())
