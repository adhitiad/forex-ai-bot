import { createClient } from "redis";
import { initExchange } from "./exchangeClient";
import { executeOrder, Order } from "./executor";

async function main() {
  initExchange();

  const rdb = createClient({
    username: "default",
    password: "fNzirXEiYdNFVFA3tbHGPoTJA00q2jX9",
    socket: {
      host: "redis-16018.c334.asia-southeast2-1.gce.cloud.redislabs.com",
      port: 16018,
    },
  });
  await rdb.connect();

  const subscriber = rdb.duplicate();
  await subscriber.connect();
  await subscriber.subscribe("trade_signals", (message) => {
    const order: Order = JSON.parse(message);
    executeOrder(order);
  });

  console.log("🎧 TS Executor Listening...");
}

main().catch(console.error);
