from __future__ import annotations

import json
import textwrap
from typing import Any, Dict, List, Optional, Tuple

from ..memory.memory_manager import MemoryManager
from ..services.openrouter_client import OpenRouterClient
from ..services.vector_store import SemanticStore
from ..state.conversation_store import ConversationStore
from ..utils.logger import get_logger


SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an autonomous senior software engineer tasked with building production-ready applications.
    Maintain two explicit workspaces:
    1. Short-term scratchpad for implementation details, open tasks, debugging notes.
    2. Long-term main plan that captures architecture decisions, milestones, and delivery status.

    When responding, produce JSON with the following schema:
    {
      "reply": "<message for human collaborators>",
      "scratchpad_append": "<optional text to append to scratchpad>",
      "plan_append": "<optional text to append to long-term plan>",
      "metadata": {
          "requires_human_confirmation": false,
          "context_used": ["<doc summary>", ...]
      }
    }

    Only include keys with meaningful content. Prefer concise yet actionable replies. Ask for human
    confirmation before high-risk changes or when instructions are unclear. Adhere to previously agreed plans unless instructed.
    """
).strip()


class AgentService:
    def __init__(
        self,
        memory_manager: MemoryManager,
        conversation_store: ConversationStore,
        semantic_store: SemanticStore,
        openrouter_client: OpenRouterClient,
    ) -> None:
        self.memory_manager = memory_manager
        self.conversation_store = conversation_store
        self.semantic_store = semantic_store
        self.client = openrouter_client
        self.logger = get_logger(self.__class__.__name__)

    def _prepare_messages(
        self,
        session_id: str,
        user_message: str,
        context_docs: List[Dict[str, Any]],
    ) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "system",
                "content": f"Long-term plan memory:\\n{self.memory_manager.read('plan')}",
            },
            {
                "role": "system",
                "content": f"Scratchpad memory:\\n{self.memory_manager.read('scratchpad')}",
            },
        ]
        if context_docs:
            for doc in context_docs:
                source = doc.get("metadata", {}).get("source", "semantic-memory")
                messages.append(
                    {
                        "role": "system",
                        "content": f"Retrieved context ({source}): {doc['content']}",
                    }
                )
        history = self.conversation_store.get_history(session_id)
        for item in history[-20:]:
            messages.append({"role": item["role"], "content": item["content"]})
        messages.append({"role": "user", "content": user_message})
        return messages

    @staticmethod
    def _parse_response(content: str) -> Tuple[str, Optional[str], Optional[str], Dict[str, Any]]:
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].strip()
        try:
            payload = json.loads(cleaned)
            reply = payload.get("reply", "")
            scratchpad_append = payload.get("scratchpad_append")
            plan_append = payload.get("plan_append")
            metadata = payload.get("metadata", {})
            return reply, scratchpad_append, plan_append, metadata
        except json.JSONDecodeError:
            metadata: Dict[str, Any] = {"requires_human_confirmation": False}
            return content, None, None, metadata

    async def handle_message(
        self,
        session_id: str,
        user_message: str,
        model: str,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        context_docs_raw = self.semantic_store.similarity_search(user_message, top_k=5)
        context_docs = [
            {
                "id": doc.id,
                "content": doc.content,
                "metadata": doc.metadata,
            }
            for doc in context_docs_raw
        ]
        messages = self._prepare_messages(session_id, user_message, context_docs)
        self.conversation_store.append_message(session_id, "user", user_message)
        completion = await self.client.chat_completion(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        choice = completion.get("choices", [{}])[0]
        assistant_message = choice.get("message", {}).get("content", "")
        reply, scratchpad_append, plan_append, metadata = self._parse_response(
            assistant_message
        )
        if scratchpad_append:
            self.memory_manager.append("scratchpad", scratchpad_append.strip())
        if plan_append:
            self.memory_manager.append("plan", plan_append.strip())
        self.conversation_store.append_message(session_id, "assistant", reply)
        response = {
            "reply": reply,
            "context": context_docs,
            "scratchpad": self.memory_manager.read("scratchpad"),
            "plan": self.memory_manager.read("plan"),
            "metadata": metadata,
        }
        return response

    def ingest_document(self, content: str, metadata: Optional[Dict[str, str]] = None) -> str:
        return self.semantic_store.add_document(content, metadata)

    def reset_session(self, session_id: str) -> None:
        self.conversation_store.reset_session(session_id)
