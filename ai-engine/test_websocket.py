import asyncio
import json

import websockets


async def test_websocket():
    uri = "ws://localhost:8000/ws/realtime"

    try:
        async with websockets.connect(uri) as websocket:
            print("✅ Connected to WebSocket")

            # Send ping
            await websocket.send("ping")
            response = await websocket.recv()
            print(f"📡 Ping response: {response}")

            # Wait for some messages (if any)
            try:
                for _ in range(5):
                    message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    data = json.loads(message)
                    print(f"📊 Received: {data}")
            except asyncio.TimeoutError:
                print("⏰ No more messages received within timeout")

    except (websockets.exceptions.WebSocketException, OSError) as e:
        print(f"❌ WebSocket test failed: {e}")


if __name__ == "__main__":
    asyncio.run(test_websocket())
