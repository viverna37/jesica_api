import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool

from utils.stt import load_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
for noisy in ("httpx", "httpx2", "huggingface_hub"):
    logging.getLogger(noisy).setLevel(logging.WARNING)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await run_in_threadpool(load_model)
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="Jesica API",
        version="1.0.0",
        lifespan=lifespan,
    )
    from routers.websocket import ws_router
    app.include_router(ws_router)

    return app

app = create_app()
