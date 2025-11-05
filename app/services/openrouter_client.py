from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import httpx

from ..utils.logger import get_logger


class OpenRouterClient:
    def __init__(self) -> None:
        self.base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.headers = {
            "HTTP-Referer": os.getenv("OPENROUTER_APP_URL", "https://agentic.local"),
            "X-Title": os.getenv("OPENROUTER_APP_NAME", "Agentic Coder"),
        }
        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"
        self.logger = get_logger(self.__class__.__name__)

    async def list_models(self) -> Dict[str, Any]:
        url = f"{self.base_url}/models"
        if not self.api_key:
            self.logger.warning("OPENROUTER_API_KEY not set; returning mock model list")
            return {
                "data": [
                    {
                        "id": "mock/free-coder",
                        "name": "Mock Free Coder",
                        "pricing": {"prompt": 0.0, "completion": 0.0},
                        "context_length": 4096,
                    },
                    {
                        "id": "mock/paid-coder",
                        "name": "Mock Paid Coder",
                        "pricing": {"prompt": 0.001, "completion": 0.002},
                        "context_length": 16384,
                    },
                ]
            }
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, headers=self.headers)
            response.raise_for_status()
            return response.json()

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        url = f"{self.base_url}/chat/completions"
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens
        headers = {**self.headers, **(extra_headers or {})}
        if not self.api_key:
            self.logger.warning("OPENROUTER_API_KEY not set; returning mock completion")
            content = (
                "Mock response generated because OPENROUTER_API_KEY is missing. "
                "Set the API key to receive live model outputs."
            )
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": content,
                        }
                    }
                ]
            }
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()
