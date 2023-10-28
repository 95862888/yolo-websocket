import asyncio
from websockets.server import serve


async def handler(websocket):
    async for message in websocket:
        print('recv msg', message)
        # await websocket.recv(message)


async def main():
    async with serve(handler, '0.0.0.0', 8765):
        await asyncio.Future()  # run forever

asyncio.run(main())
