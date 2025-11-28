import Binance, { OrderType } from "binance-api-node";

let client: ReturnType<typeof Binance>;

export function initExchange() {
  const apiKey = process.env.BINANCE_API_KEY;
  const secret = process.env.BINANCE_SECRET_KEY;
  const useTestnet = process.env.USE_TESTNET === "true";

  const options: any = {
    apiKey,
    apiSecret: secret,
  };

  if (useTestnet) {
    options.httpBase = "https://testnet.binance.vision";
  }

  client = Binance(options);

  console.log("✅ Binance Client Ready");
}

export async function placeMarketOrder(
  symbol: string,
  side: "BUY" | "SELL",
  qty: number
): Promise<{ orderId: string; fillPrice: number }> {
  const order = await client.order({
    symbol,
    side,
    quantity: qty.toString(),
    type: OrderType.MARKET,
  });

  const fillPrice = parseFloat(order.fills?.[0]?.price || "0");
  return { orderId: order.orderId.toString(), fillPrice };
}
