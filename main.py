from fastapi import WebSocket, FastAPI

def create_app() -> FastAPI:
    app = FastAPI(
        title="Jesica API",
        version="1.0.0",
    )
    from routers.websocket import ws_router
    app.include_router(ws_router)

    return app

app = create_app()



