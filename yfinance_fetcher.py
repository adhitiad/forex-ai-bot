import asyncio
import logging
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

logger = logging.getLogger("YFinanceFetcher")


class YFinanceDataFetcher:
    """Alternatif data fetcher menggunakan Yahoo Finance"""

    async def fetch_market_data(self, symbol="BTC-USD", days=365, interval="1h"):
        """
        Download data crypto dari Yahoo Finance

        Args:
            symbol: Symbol Yahoo Finance (BTC-USD, ETH-USD, dll)
            days: Jumlah hari historis
            interval: 1h, 1d, dll
        """
        try:
            logger.info(
                f"📥 Downloading {symbol} data from Yahoo Finance ({days} days)..."
            )

            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            # Download data
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, interval=interval)

            if df.empty:
                logger.error("❌ No data fetched from Yahoo Finance")
                return pd.DataFrame()

            # Rename columns to match CCXT format
            df = df.rename(
                columns={
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                }
            )

            # Keep only needed columns
            df = df[["open", "high", "low", "close", "volume"]]

            # Ensure index is datetime
            df.index.name = "timestamp"

            logger.info(f"✅ Downloaded {len(df)} candles from Yahoo Finance")
            return df

        except Exception as e:
            logger.error(f"❌ Error downloading from Yahoo Finance: {e}")
            return pd.DataFrame()


# Create instance
yfinance_fetcher = YFinanceDataFetcher()
