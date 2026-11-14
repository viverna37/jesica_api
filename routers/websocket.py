from fastapi import FastAPI, APIRouter
from starlette.websockets import WebSocket

ws_router = APIRouter(
    prefix="/v1",
    tags=["websocket",],
)

@ws_router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        await websocket.send_text(f"Message text was: {data}")
