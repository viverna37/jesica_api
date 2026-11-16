import json
import wave
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, WebSocket
from fastapi.concurrency import run_in_threadpool

import logging

import numpy as np

from utils.tts import stt

logger = logging.getLogger(__name__)

ws_router = APIRouter()

RECORDINGS = Path("recordings")
RECORDINGS.mkdir(exist_ok=True)

SAMPLE_RATE = 16000


def save_wav(pcm: bytes) -> Path:
    path = RECORDINGS / f"{datetime.now():%Y%m%d_%H%M%S}.wav"
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)          # моно
        f.setsampwidth(2)          # 16 бит
        f.setframerate(SAMPLE_RATE)
        f.writeframes(pcm)
    return path


@ws_router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    buffer = bytearray()
    recording = False

    while True:
        message = await websocket.receive()
        logger.debug(f"Received message: {message}")
        if message["type"] == "websocket.disconnect":
            break

        if message.get("text") is not None:          # управляющее сообщение
            event = json.loads(message["text"])

            if event["type"] == "start":
                buffer.clear()
                recording = True
                await websocket.send_text(json.dumps({"type": "listening"}))

            elif event["type"] == "end" and recording:
                recording = False
                response = await run_in_threadpool(stt, bytes(buffer))
                logger.debug(f"Received response: {response}")

                # path = save_wav(bytes(buffer))
                # await websocket.send_text(json.dumps({
                #     "type": "saved", "file": path.name, "bytes": len(buffer),
                # }))

        elif message.get("bytes") is not None:      # кусок звука
            if recording:
                buffer.extend(message["bytes"])
            # байты без start просто игнорируем