import asyncio
import json
import logging
import os
import re
import time
import wave
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, WebSocket
from fastapi.concurrency import run_in_threadpool

from utils.llm import ask
from utils.stt import stt
from utils.tts import TTS_SAMPLE_RATE, tts

logger = logging.getLogger(__name__)

ws_router = APIRouter()

RECORDINGS = Path("recordings")
RECORDINGS.mkdir(exist_ok=True)

SAMPLE_RATE = 16000

# Отвечать только на фразы с именем. Когда имя начнёт ловить ESP — поставить false в .env
REQUIRE_WAKE_WORD = os.getenv("REQUIRE_WAKE_WORD", "true").lower() == "true"
# Сколько секунд после ответа можно говорить без имени
FOLLOW_UP_SECONDS = float(os.getenv("FOLLOW_UP_SECONDS", "20"))
# Сколько последних сообщений помнить (пара реплик = 2 сообщения)
HISTORY_LIMIT = 20

AUDIO_CHUNK_BYTES = TTS_SAMPLE_RATE * 2 // 10    # звук ответа шлём кусками по 100 мс
AUDIO_LEAD_SECONDS = 0.4                         # насколько можно опережать проигрывание (буфер ESP маленький)

WAKE = re.compile(r"д?жесс?ик|jess?ica", re.IGNORECASE)


def save_wav(pcm: bytes) -> Path:
    path = RECORDINGS / f"{datetime.now():%Y%m%d_%H%M%S}.wav"
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)          # моно
        f.setsampwidth(2)          # 16 бит
        f.setframerate(SAMPLE_RATE)
        f.writeframes(pcm)
    return path


async def send_json(websocket: WebSocket, **data) -> None:
    # ensure_ascii=False — чтобы кириллица шла буквами, а не \u0414...
    await websocket.send_text(json.dumps(data, ensure_ascii=False))


async def send_audio(websocket: WebSocket, pcm: bytes) -> None:
    """Шлёт звук в темпе речи и ждёт, пока устройство его доиграет."""
    bytes_per_second = TTS_SAMPLE_RATE * 2
    play_until = time.monotonic()                # когда устройство доиграет уже отправленное
    for offset in range(0, len(pcm), AUDIO_CHUNK_BYTES):
        chunk = pcm[offset:offset + AUDIO_CHUNK_BYTES]
        await websocket.send_bytes(chunk)
        now = time.monotonic()
        play_until = max(play_until, now) + len(chunk) / bytes_per_second
        ahead = play_until - now
        if ahead > AUDIO_LEAD_SECONDS:
            await asyncio.sleep(ahead - AUDIO_LEAD_SECONDS)
    await asyncio.sleep(max(0.0, play_until - time.monotonic()))


async def speak(websocket: WebSocket, text: str) -> None:
    """Озвучивает текст: audio_start -> бинарные куски -> audio_end. Ошибка синтеза не роняет ответ."""
    try:
        pcm = await run_in_threadpool(tts, text)
    except Exception:
        logger.exception("tts failed for %r", text)
        return
    if not pcm:
        return
    await send_json(websocket, type="audio_start", sample_rate=TTS_SAMPLE_RATE)
    await send_audio(websocket, pcm)
    await send_json(websocket, type="audio_end")


@ws_router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    # Всё это — своё у каждого подключения (у каждой ESP)
    buffer = bytearray()
    recording = False
    history: list[dict] = []        # память диалога: только текст реплик
    active_until = 0.0              # до этого момента имя называть не нужно

    while True:
        message = await websocket.receive()
        if message["type"] == "websocket.disconnect":
            break

        if message.get("text") is not None:          # управляющее сообщение
            event = json.loads(message["text"])
            logger.debug("event: %s", event)          # байты звука не логируем — их слишком много

            if event["type"] == "start":
                buffer.clear()
                recording = True
                await send_json(websocket, type="listening")

            elif event["type"] == "end" and recording:
                recording = False
                pcm = bytes(buffer)
                buffer.clear()
                try:
                # 1. Голос -> текст
                    text = (await run_in_threadpool(stt, pcm)).strip()
                    logger.info("heard: %r", text)
                    await send_json(websocket, type="heard", text=text)
                    if not text:
                        continue

                    # 2. Это обращение к Джесике?
                    addressed = (
                        not REQUIRE_WAKE_WORD
                        or WAKE.search(text)
                        or time.monotonic() < active_until
                    )
                    if not addressed:
                        continue

                    # 3. Текст -> Claude
                    await send_json(websocket, type="thinking")
                    try:
                        meta, answer = await ask(history, text)
                    except Exception:
                        logger.exception("claude request failed")
                        await send_json(websocket, type="reply", emotion="sad",
                                        text="Ой, я что-то задумалась. Повтори, пожалуйста?")
                        await speak(websocket, "Ой, я что-то задумалась. Повтори, пожалуйста?")
                        continue

                    logger.info("said (%s): %r", meta["emotion"], answer)

                    # 4. Запоминаем реплики (ответ — без JSON-строки)
                    history.append({"role": "user", "content": text})
                    history.append({"role": "assistant", "content": answer})
                    del history[:-HISTORY_LIMIT]

                    # 5. Разговор продолжается — следующие секунды имя не нужно
                    active_until = time.monotonic() + FOLLOW_UP_SECONDS

                    await send_json(websocket, type="reply",
                                    emotion=meta["emotion"], text=answer)

                    # 6. Озвучка (весь ответ целиком)
                    await speak(websocket, answer)
                finally:
                    await send_json(websocket, type="done")

        elif message.get("bytes") is not None:      # кусок звука
            if recording:
                buffer.extend(message["bytes"])
            # байты без start просто игнорируем