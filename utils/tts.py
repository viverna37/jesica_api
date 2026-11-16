"""Синтез речи: Silero, локально на CPU."""
import logging
import os
import threading

import torch

from utils.text import has_letters, normalize_for_tts, split_for_tts

logger = logging.getLogger(__name__)

TTS_MODEL = os.getenv("TTS_MODEL", "v5_5_ru")
TTS_VOICE = os.getenv("TTS_VOICE", "baya")
TTS_SAMPLE_RATE = int(os.getenv("TTS_SAMPLE_RATE", "24000"))
TTS_THREADS = int(os.getenv("TTS_THREADS", "4"))

model = None
_lock = threading.Lock()  # модель нельзя вызывать из двух потоков сразу (две ESP — два потока)


def load_model() -> None:
    """Грузит Silero. Кэш — в TORCH_HOME (в Docker это /models/torch)."""
    global model
    torch.set_num_threads(TTS_THREADS)
    logger.info("Loading Silero model '%s' (voice=%s)...", TTS_MODEL, TTS_VOICE)
    model, _ = torch.hub.load(
        repo_or_dir="snakers4/silero-models",
        model="silero_tts",
        language="ru",
        speaker=TTS_MODEL,          # у Silero "speaker" здесь — это имя модели
        trust_repo=True,
    )
    model.to(torch.device("cpu"))
    logger.info("Silero model loaded")


def tts(text: str) -> bytes:
    """Синхронный синтез: текст -> PCM 16 бит моно, TTS_SAMPLE_RATE.
    Вызывать через run_in_threadpool. Пустые байты, если озвучивать нечего."""
    if model is None:
        raise RuntimeError("Silero model is not loaded")

    clean = normalize_for_tts(text)
    if not has_letters(clean):
        return b""

    pieces = []
    with _lock:
        for part in split_for_tts(clean):
            if has_letters(part):
                pieces.append(model.apply_tts(text=part, speaker=TTS_VOICE, sample_rate=TTS_SAMPLE_RATE))
    if not pieces:
        return b""
    audio = torch.cat(pieces)
    # Тензор -1..1 -> int16
    return (audio.clamp(-1, 1) * 32767).to(torch.int16).numpy().tobytes()
