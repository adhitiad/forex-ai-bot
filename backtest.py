import asyncio
import logging

import torch

from config import settings
from features import fetcher, processor
from model import TimeSeriesTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Backtest")


async def run_backtest():
    logger.info(f"📊 Backtest: {settings.YFINANCE_SYMBOL}")
    raw_df = await fetcher.fetch_market_data(days=60)
    full_df, feature_matrix = processor.process(raw_df)

    model = TimeSeriesTransformer(input_dim=feature_matrix.shape[1])
    try:
        model.load_state_dict(torch.load(settings.MODEL_FILE))
        model.eval()
    except:
        return

    balance = 10000.0
    position = None
    entry_price = 0.0
    pip_unit = 0.01 if "JPY" in settings.YFINANCE_SYMBOL else 0.0001
    trades = []

    for i in range(settings.SEQ_LEN, len(feature_matrix)):
        price = raw_df.iloc[i]["close"]

        if position:
            close = False
            pnl = 0
            if position == "BUY":
                if raw_df.iloc[i]["high"] >= entry_price + (
                    settings.TAKE_PROFIT_PIPS * pip_unit
                ):
                    pnl = (settings.TAKE_PROFIT_PIPS * pip_unit) * settings.TRADE_UNITS
                    close = True
                elif raw_df.iloc[i]["low"] <= entry_price - (
                    settings.STOP_LOSS_PIPS * pip_unit
                ):
                    pnl = -(settings.STOP_LOSS_PIPS * pip_unit) * settings.TRADE_UNITS
                    close = True

            if close:
                balance += pnl
                trades.append({"pnl": pnl, "bal": balance})
                position = None
                continue

        if not position:
            tensor = torch.FloatTensor(
                feature_matrix[i - settings.SEQ_LEN : i]
            ).unsqueeze(0)
            probs = model(tensor).detach().numpy()[0]
            if probs[1] > 0.5:
                position = "BUY"
                entry_price = price
            elif probs[2] > 0.5:
                position = "SELL"
                entry_price = price

    print(f"🏁 Final Balance: ${balance:.2f} | Trades: {len(trades)}")


if __name__ == "__main__":
    asyncio.run(run_backtest())
