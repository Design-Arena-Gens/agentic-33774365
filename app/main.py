from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from .routers import api
from .utils.logger import get_logger


APP_TITLE = "Agentic Coder"
APP_DESCRIPTION = (
    "Autonomous AI coder agent with memory persistence, semantic search, and human-in-the-loop controls."
)

logger = get_logger(__name__)


def create_app() -> FastAPI:
    app = FastAPI(
        title=APP_TITLE,
        description=APP_DESCRIPTION,
        default_response_class=HTMLResponse,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api.router, prefix="/api")

    static_index = Path(__file__).resolve().parent.parent / "static" / "index.html"

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def root() -> HTMLResponse:
        try:
            return HTMLResponse(static_index.read_text(encoding="utf-8"))
        except FileNotFoundError:
            logger.error("Static index.html not found at %s", static_index)
            return HTMLResponse(
                "<h1>Agent UI not yet generated</h1>",
                status_code=500,
            )

    return app


app = create_app()
