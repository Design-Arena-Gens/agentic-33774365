from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, Body, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from ..memory.memory_manager import MemoryManager
from ..services.agent import AgentService
from ..services.openrouter_client import OpenRouterClient
from ..services.vector_store import SemanticStore
from ..state.conversation_store import ConversationStore
from ..utils.logger import get_logger


router = APIRouter()
logger = get_logger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
memory_manager = MemoryManager(
    scratchpad_path=BASE_DIR / "scratchpad.md",
    plan_path=BASE_DIR / "main-plan.md",
)
conversation_store = ConversationStore(BASE_DIR / "state" / "conversations.json")
semantic_store = SemanticStore(BASE_DIR / "vector_store" / "store.json")
openrouter_client = OpenRouterClient()
agent_service = AgentService(
    memory_manager=memory_manager,
    conversation_store=conversation_store,
    semantic_store=semantic_store,
    openrouter_client=openrouter_client,
)


@router.get("/health")
async def health_check() -> Dict[str, str]:
    return {"status": "ok"}


@router.get("/models")
async def list_models() -> Dict[str, Any]:
    raw_models = await openrouter_client.list_models()
    data = raw_models.get("data", [])
    categorized = []
    for model in data:
        pricing = model.get("pricing", {})
        prompt_cost = pricing.get("prompt", 0.0) or 0.0
        completion_cost = pricing.get("completion", 0.0) or 0.0
        plan = "free" if (prompt_cost == 0 and completion_cost == 0) else "paid"
        categorized.append(
            {
                "id": model.get("id"),
                "name": model.get("name", model.get("id")),
                "pricing": pricing,
                "context_length": model.get("context_length"),
                "category": plan,
            }
        )
    return {"models": categorized}


@router.get("/memory")
async def read_memory() -> Dict[str, str]:
    snapshot = memory_manager.snapshot()
    return {"scratchpad": snapshot["scratchpad"], "plan": snapshot["plan"]}


@router.post("/memory")
async def update_memory(
    kind: str = Body(..., embed=True),
    content: str = Body(..., embed=True),
) -> Dict[str, str]:
    if kind not in {"scratchpad", "plan"}:
        raise HTTPException(status_code=400, detail="Invalid memory kind")
    memory_manager.write(kind, content)
    return {"status": "updated", "kind": kind}


@router.post("/chat")
async def chat(
    payload: Dict[str, Any],
) -> JSONResponse:
    session_id = payload.get("session_id", "default")
    message = payload.get("message")
    if not message:
        raise HTTPException(status_code=400, detail="Message is required")
    model = payload.get("model")
    if not model:
        raise HTTPException(status_code=400, detail="Model selection is required")
    temperature = float(payload.get("temperature", 0.2))
    max_tokens: Optional[int] = payload.get("max_tokens")
    response = await agent_service.handle_message(
        session_id=session_id,
        user_message=message,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return JSONResponse(response)


@router.post("/ingest")
async def ingest_document(
    source: Optional[str] = Body(None),
    content: Optional[str] = Body(None),
    file: Optional[UploadFile] = File(None),
) -> Dict[str, Any]:
    text_content = content
    metadata: Dict[str, str] = {}
    if file:
        file_bytes = await file.read()
        text_content = file_bytes.decode("utf-8")
        metadata["source"] = file.filename or source or "upload"
    if not text_content:
        raise HTTPException(status_code=400, detail="No content provided for ingestion")
    if source and "source" not in metadata:
        metadata["source"] = source
    doc_id = agent_service.ingest_document(text_content, metadata)
    return {"status": "indexed", "document_id": doc_id}


@router.post("/session/reset")
async def reset_session(
    session_id: str = Body("default", embed=True),
) -> Dict[str, str]:
    agent_service.reset_session(session_id)
    return {"status": "reset", "session_id": session_id}
