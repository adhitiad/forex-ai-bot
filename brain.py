import asyncio
import json
import logging
import os
from collections import deque

import pandas as pd
import redis.asyncio as redis
import torch
import torch.nn.functional as F

from config import settings
from features import processor
from model import TimeSeriesTransformer
from state_manager import state_manager
from stream_manager import streamor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MultiBrain")


class MultiBrain:
    def __init__(self):
        # Dictionary untuk menyimpan banyak model & buffer
        self.models = {}
        self.buffers = {}

        # Init Buffer untuk setiap pair aktif
        for sym in settings.ACTIVE_SYMBOLS:
            self.buffers[sym] = deque(maxlen=settings.SEQ_LEN + 20)

        self.r = redis.Redis(
            host=settings.REDIS_HOST, port=settings.REDIS_PORT, decode_responses=True
        )

    async def load_all_models(self):
        """Load model untuk setiap pair aktif ke memori"""
        for sym in settings.ACTIVE_SYMBOLS:
            path = settings.get_model_path(sym)
            if os.path.exists(path):
                model = TimeSeriesTransformer(input_dim=4)
                model.load_state_dict(torch.load(path))
                model.eval()
                self.models[sym] = model
                logger.info(f"🧠 Loaded Model: {sym}")
            else:
                logger.warning(f"⚠️ No model found for {sym}")

    async def run(self):
        await self.load_all_models()
        await streamor.connect()
        logger.info(f"🚀 Multi-Brain Running for: {settings.ACTIVE_SYMBOLS}")

        while True:
            # Terima data market (bisa dari berbagai pair)
            candles = await streamor.consume_market_data()
            if not candles:
                await asyncio.sleep(0.01)
                continue

            for c in candles:
                symbol = c["symbol"]

                # Skip jika pair tidak ada di config
                if symbol not in settings.ACTIVE_SYMBOLS:
                    continue

                # Masukkan ke buffer spesifik pair tersebut
                self.buffers[symbol].append(c)

                # --- LOGIKA INFERENCE PER PAIR ---
                # Cek apakah punya posisi di pair ini?
                active_pos = await state_manager.get_active_position(symbol)
                if active_pos:
                    continue  # Skip jika sedang trading pair ini

                # Cek apakah data cukup & model ada
                if (
                    symbol in self.models
                    and len(self.buffers[symbol]) >= settings.SEQ_LEN
                ):
                    try:
                        # Convert buffer ke DataFrame
                        df = pd.DataFrame(list(self.buffers[symbol]))

                        # Process Feature (PENTING: Pass symbol agar scaler benar)
                        _, scaled_data = processor.process(df, symbol=symbol)

                        if len(scaled_data) < settings.SEQ_LEN:
                            continue

                        # Prepare Tensor
                        input_seq = scaled_data[-settings.SEQ_LEN :]
                        tensor = torch.FloatTensor(input_seq).unsqueeze(0)

                        # Predict
                        action = "HOLD"  # Initialize action variable
                        with torch.no_grad():
                            logits = self.models[symbol](tensor)
                            probs = F.softmax(logits, dim=1)
                            top_p, top_class = torch.max(probs, dim=1)

                            pred = top_class.item()
                            prob_pct = top_p.item() * 100

                            if prob_pct > 70:  # Threshold
                                if pred == 1:
                                    action = "BUY"
                                elif pred == 2:
                                    action = "SELL"

                            if action != "HOLD":
                                # --- HITUNG ESTIMASI TP/SL ---
                                # Ambil (jika ada point dari info simbol) atau estimasi 0.0001
                                point = 0.0001
                                if "JPY" in settings.ACTIVE_SYMBOLS[0]:
                                    point = 0.01

                                est_price = c["close"]
                                est_sl = 0.0
                                est_tp = 0.0

                                if action == "BUY":
                                    est_sl = est_price - (settings.SL_PIPS * 10 * point)
                                    est_tp = est_price + (settings.TP_PIPS * 10 * point)
                                elif action == "SELL":
                                    est_sl = est_price + (settings.SL_PIPS * 10 * point)
                                    est_tp = est_price - (settings.TP_PIPS * 10 * point)

                                # --- FORMAT LOG TERBARU ---
                                log_msg = (
                                    f"\n⚡ INTRADAY SIGNAL (M15):\n"
                                    f"• Action       : {action}\n"
                                    f"• Price        : {est_price:.5f}\n"
                                    f"• TP / SL      : {est_tp:.5f} / {est_sl:.5f}\n"
                                    f"• Confidence   : {prob_pct:.2f}%\n"
                                    f"• Reason       : AI prediction above threshold"
                                )
                                logger.info(log_msg)

                                self.is_pending_order = True
                                await streamor.push_signal(
                                    {
                                        "action": action,
                                        "type": "MARKET",
                                        "symbol": settings.ACTIVE_SYMBOLS[
                                            0
                                        ],  # Atau variable symbol jika multi-pair
                                        "entry_price": est_price,
                                        "tp": est_tp,  # <--- Kirim TP
                                        "sl": est_sl,  # <--- Kirim SL
                                        "confidence": prob_pct,
                                        "reason": "AI prediction above threshold",
                                        "source": "BRAIN_M15",
                                    }
                                )

                    except Exception as e:
                        logger.error(f"Error {symbol}: {e}")


if __name__ == "__main__":
    asyncio.run(MultiBrain().run())
