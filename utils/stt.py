import logging
import os

import numpy as np
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

MODEL_SIZE = os.getenv("WHISPER_MODEL", "small")
DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

model: WhisperModel | None = None


def load_model() -> None:
    global model
    logger.info("Loading Whisper model '%s' (device=%s, compute_type=%s)...", MODEL_SIZE, DEVICE, COMPUTE_TYPE)
    model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    logger.info("Whisper model loaded")


def stt(pcm: bytes) -> str:
    if model is None:
        raise RuntimeError("Whisper model is not loaded")
    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    segments, info = model.transcribe(audio, beam_size=1, language="ru", vad_filter=True)
    text = "".join(segment.text for segment in segments).strip()
    logger.info("STT (%.2fs audio): %s", info.duration, text)
    return text
