import websockets
import json
import asyncio
import time


class Comm():
    def __init__(self, abii_connected: bool):
        self.abii_connected = abii_connected

    def send_fer_class(self, ferclass: int):
        """
        Send websocket message.
        1 -> Happy
        2 -> Sad
        3 -> Party
        4 -> Introduction
        """
        if not self.abii_connected:
            # Skip sending to ABii since we are running this locally.
            return -1
        print(f'LOG --> In send_fer_class')
        ws_server = "ws://192.168.42.1:4811/fer_listener"
        req = json.dumps({"FERCLASS": ferclass})

        async def inner():
            async with websockets.connect(ws_server) as ws:
                await ws.send(req)
        return asyncio.get_event_loop().run_until_complete(inner())
