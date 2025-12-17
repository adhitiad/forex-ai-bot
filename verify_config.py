import logging
import sys
import os

# Redirect output to a file
with open("verify_output.txt", "w") as f:
    try:
        f.write("Starting verification...\n")

        from config import settings

        f.write(f"Active Symbols: {settings.ACTIVE_SYMBOLS}\n")
        f.write(f"YFinance Symbol: {settings.YFINANCE_SYMBOL}\n")

        import feature_engine_test_dummy  # This fails if imports are wrong

        f.write("Imports successful.\n")

    except Exception as e:
        f.write(f"Error: {e}\n")
