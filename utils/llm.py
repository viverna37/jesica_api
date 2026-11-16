import json
import os
from pathlib import Path

from anthropic import AsyncAnthropic

# Ключ и ANTHROPIC_BASE_URL библиотека берёт из окружения сама
client = AsyncAnthropic()
MODEL = os.getenv("CLAUDE_MODEL", "claude-haiku-5-5")

# Промпт читаем один раз при старте — правишь файл, перезапускаешь сервер
SYSTEM_PROMPT = Path("static/system_prompt.txt").read_text(encoding="utf-8")

EMOTIONS = {"neutral", "happy", "smile", "sad", "angry", "surprised", "sleepy"}


def parse_reply(raw: str) -> tuple[dict, str]:
    """Отделяет первую строку-JSON от текста, который Джесика скажет вслух."""
    first, _, rest = raw.strip().partition("\n")
    try:
        meta = json.loads(first)
        text = rest.strip()
    except json.JSONDecodeError:
        meta, text = {}, raw.strip()      # модель забыла JSON — не падаем, просто без эмоции
    if meta.get("emotion") not in EMOTIONS:
        meta["emotion"] = "neutral"
    return meta, text


async def ask(history: list[dict], user_text: str) -> tuple[dict, str]:
    response = await client.messages.create(
        model=MODEL,
        max_tokens=600,
        system=SYSTEM_PROMPT,
        messages=[*history, {"role": "user", "content": user_text}],
    )
    raw = response.content[0].text
    return parse_reply(raw)