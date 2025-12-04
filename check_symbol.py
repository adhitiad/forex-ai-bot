import yfinance as yf


def check_symbol(symbol):
    print(f"🔍 Checking {symbol}...")
    try:
        ticker = yf.Ticker(symbol)
        # Coba ambil data hari ini
        hist = ticker.history(period="1d")

        if hist.empty:
            print(f"❌ INVALID: Simbol '{symbol}' tidak ditemukan atau tidak ada data.")
            return False
        else:
            price = hist["Close"].iloc[-1]
            print(f"✅ VALID: '{symbol}' ditemukan. Harga terakhir: {price}")
            return True
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False


# --- DAFTAR YANG MAU DITES ---
test_symbols = [
    "BTC-USD",  # Crypto Benar
    "BTCUSD",  # Crypto Salah (kurang strip)
    "XAUUSD=X",  # Gold Benar
    "BBCA.JK",  # Saham Indo Benar
    "BBCA",  # Saham Indo Salah (kurang .JK)
    "AAPL",  # Saham US Benar
    "GOTO.JK",
]

if __name__ == "__main__":
    print("--- VALIDASI SIMBOL YFINANCE ---")
    for sym in test_symbols:
        check_symbol(sym)
        print("-" * 20)
