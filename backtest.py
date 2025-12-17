import asyncio
import json
import logging

import numpy as np
import pandas as pd
import torch

from config import settings
from features import fetcher, processor
from model import TimeSeriesTransformer

# Setup Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FusionBacktest")


class FusionSimulator:
    def __init__(self):
        self.model_v1 = None
        # Bobot Simulasi
        self.W_TECH = 0.4
        self.W_LLM = (
            0.3  # Kita simulasi LLM random/bias karena mahal kalau backtest full API
        )
        self.W_RL = 0.3

    def load_brain_v1(self):
        self.model_v1 = TimeSeriesTransformer(input_dim=4, output_dim=3)
        try:
            self.model_v1.load_state_dict(torch.load(settings.MODEL_FILE))
            self.model_v1.eval()
            logger.info("✅ Brain V1 Loaded")
        except:
            logger.error("❌ Model V1 Missing")

    async def run(self):
        logger.info("⏳ Fetching 60 Days Data for Simulation...")
        raw_df = await fetcher.fetch_market_data(days=60)
        df, scaled = processor.process(raw_df)

        if self.model_v1 is None:
            self.load_brain_v1()

        balance = 10000.0
        equity_curve = [balance]
        position = None  # {'side': 'BUY', 'price': 1.0, 'sl': 0.9}
        trades = []

        logger.info("🚀 Starting Event-Driven Simulation...")

        # Loop setiap candle (seperti Live Stream)
        for i in range(settings.SEQ_LEN, len(scaled)):
            current_price = raw_df.iloc[i]["close"]
            current_time = raw_df.index[i]

            # 1. GENERATE VOTES

            # A. Vote Brain V1
            seq = scaled[i - settings.SEQ_LEN : i]
            tensor = torch.FloatTensor(seq).unsqueeze(0)
            with torch.no_grad():
                logits = self.model_v1(tensor)
                pred = torch.argmax(logits, dim=1).item()

            vote_v1 = 1 if pred == 1 else -1 if pred == 2 else 0

            # B. Vote LLM (Simulasi: Random noise + Trend Following)
            # Di real, ini call API. Di backtest, kita pakai Simple MA cross sebagai proxy LLM
            ma50 = raw_df.iloc[i]["close"] > raw_df.iloc[i - 50 : i]["close"].mean()
            vote_llm = 1 if ma50 else -1

            # C. Vote RL (Simulasi: Random untuk demo)
            vote_rl = 0

            # 2. FUSION LOGIC
            final_score = (
                (vote_v1 * self.W_TECH)
                + (vote_llm * self.W_LLM)
                + (vote_rl * self.W_RL)
            )

            action = "HOLD"
            if final_score > 0.4:
                action = "BUY"
            elif final_score < -0.4:
                action = "SELL"

            # 3. EXECUTION LOGIC (Simulasi Broker)

            # Cek Exit Trade (SL/TP)
            if position:
                pnl = 0
                closed = False

                # Logic sederhana TP/SL fixed
                if position["side"] == "BUY":
                    if raw_df.iloc[i]["low"] <= position["sl"]:
                        pnl = (position["sl"] - position["price"]) * 100000  # 1 Lot
                        closed = True
                        reason = "SL"
                    elif (
                        action == "SELL" and final_score < -0.6
                    ):  # Reverse signal strong
                        pnl = (current_price - position["price"]) * 100000
                        closed = True
                        reason = "Signal Reverse"

                elif position["side"] == "SELL":
                    if raw_df.iloc[i]["high"] >= position["sl"]:
                        pnl = (position["price"] - position["sl"]) * 100000
                        closed = True
                        reason = "SL"
                    elif action == "BUY" and final_score > 0.6:
                        pnl = (position["price"] - current_price) * 100000
                        closed = True
                        reason = "Signal Reverse"

                if closed:
                    balance += pnl
                    trades.append({"time": current_time, "pnl": pnl, "reason": reason})
                    position = None

            # Cek Entry Trade
            if position is None and action != "HOLD":
                sl_dist = 0.0020  # 20 pips
                sl = (
                    current_price - sl_dist
                    if action == "BUY"
                    else current_price + sl_dist
                )

                position = {"side": action, "price": current_price, "sl": sl}

            equity_curve.append(balance)

        # REPORT
        win_rate = (
            len([t for t in trades if t["pnl"] > 0]) / len(trades) if trades else 0
        )
        logger.info(f"🏁 Final Balance: ${balance:.2f}")
        logger.info(f"📊 Total Trades: {len(trades)} | Win Rate: {win_rate*100:.1f}%")
        logger.info("Simulation Finished.")


if __name__ == "__main__":
    asyncio.run(FusionSimulator().run())
