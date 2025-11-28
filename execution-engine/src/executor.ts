import * as fs from "fs";
import { placeMarketOrder } from "./exchangeClient";
import { riskManager } from "./risk";

export interface Order {
  symbol: string;
  action: string;
  confidence: number;
}

const LOG_FILE = "trade_logs.json";

export async function executeOrder(order: Order) {
  // 1. Risk Check
  if (!riskManager.canTrade(order.confidence)) {
    console.log("🛡️ Risk Manager Blocked Trade");
    return;
  }

  console.log(`⚡ EXECUTING: ${order.action} ${order.symbol}`);

  // 2. Binance API Call
  // Qty hardcoded 0.001 for safety in demo
  const qty = 0.001;
  const { orderId, fillPrice } = await placeMarketOrder(
    order.symbol,
    order.action as "BUY" | "SELL",
    qty
  );

  // 3. Log to file
  const logEntry = {
    symbol: order.symbol,
    action: order.action,
    price: fillPrice,
    quantity: qty,
    confidence: order.confidence,
    executed_at: new Date().toISOString(),
    orderId,
  };

  let logs: any[] = [];
  if (fs.existsSync(LOG_FILE)) {
    logs = JSON.parse(fs.readFileSync(LOG_FILE, "utf-8"));
  }
  logs.push(logEntry);
  fs.writeFileSync(LOG_FILE, JSON.stringify(logs, null, 2));

  riskManager.recordTrade();
  console.log(`✅ Success! ID: ${orderId} Avg: ${fillPrice.toFixed(2)}`);
}
