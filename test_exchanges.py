import asyncio
import ccxt.async_support as ccxt


async def test_exchange(exchange_id):
    """Test if an exchange is accessible"""
    try:
        print(f"\n🔍 Testing {exchange_id}...")
        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class(
            {
                "enableRateLimit": True,
                "timeout": 10000,
            }
        )

        # Try to fetch ticker
        ticker = await exchange.fetch_ticker("BTC/USDT")
        print(f"✅ {exchange_id} - SUCCESS! BTC/USDT price: ${ticker['last']}")
        await exchange.close()
        return True

    except Exception as e:
        print(f"❌ {exchange_id} - FAILED: {str(e)[:100]}")
        try:
            await exchange.close()
        except:
            pass
        return False


async def main():
    """Test multiple exchanges to find which ones work"""
    exchanges_to_test = [
        "binance",
        "kraken",
        "bybit",
        "kucoin",
        "gateio",
        "mexc",
        "bitget",
        "okx",
    ]

    print("=" * 60)
    print("Testing Exchange Connectivity from Indonesia")
    print("=" * 60)

    working_exchanges = []

    for exchange_id in exchanges_to_test:
        if await test_exchange(exchange_id):
            working_exchanges.append(exchange_id)
        await asyncio.sleep(1)  # Small delay between tests

    print("\n" + "=" * 60)
    print(
        f"✅ Working Exchanges: {', '.join(working_exchanges) if working_exchanges else 'None'}"
    )
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
