from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from .config import Settings, settings


class OllamaError(RuntimeError):
    """Ollama 调用失败。"""


class OllamaClient:
    def __init__(self, app_settings: Settings = settings) -> None:
        self.settings = app_settings
        self.timeout = httpx.Timeout(app_settings.ollama_timeout_seconds)

    async def health_check(self) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(f"{self.settings.ollama_base_url}/api/tags")
                response.raise_for_status()
                payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            return {
                "available": False,
                "models": [],
                "required_models": [self.settings.llm_model, self.settings.embedding_model],
                "chat_model": self.settings.llm_model,
                "embedding_model": self.settings.embedding_model,
                "vision_model": self.settings.vision_model,
                "base_url": self.settings.ollama_base_url,
                "openai_base_url": self.settings.ollama_openai_base_url,
                "error": str(exc),
            }

        models = [item.get("name", "") for item in payload.get("models", [])]
        required = [self.settings.llm_model, self.settings.embedding_model]
        return {
            "available": True,
            "models": models,
            "required_models": required,
            "chat_model": self.settings.llm_model,
            "embedding_model": self.settings.embedding_model,
            "vision_model": self.settings.vision_model,
            "base_url": self.settings.ollama_base_url,
            "openai_base_url": self.settings.ollama_openai_base_url,
            "missing_models": [model for model in required if model not in models],
        }

    async def chat(self, messages: list[dict[str, str]], max_tokens: int = 384) -> str:
        failures = []
        # Native first allows us to explicitly disable Qwen thinking for responsive replies.
        for native in (True, False):
            url = f"{self.settings.ollama_base_url}/api/chat" if native else f"{self.settings.ollama_openai_base_url}/chat/completions"
            payload = {"model": self.settings.llm_model, "messages": messages, "stream": False}
            if native:
                payload.update({"think": False, "options": {"temperature": .6, "top_p": .9, "num_predict": max_tokens}})
            else:
                payload.update({"temperature": .6, "top_p": .9, "max_tokens": max_tokens})
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(url, json=payload)
                    response.raise_for_status()
                    data = response.json()
                    content = data["message"]["content"] if native else data["choices"][0]["message"]["content"]
                    if content and content.strip():
                        return content.strip()
            except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
                failures.append(str(exc))
        raise OllamaError("Ollama 未返回可用回复：" + "; ".join(failures))

    async def stream_chat(self, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        """Stream tokens from Ollama's native NDJSON chat endpoint."""
        emitted = False
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream(
                    "POST",
                    f"{self.settings.ollama_base_url}/api/chat",
                    json={
                        "model": self.settings.llm_model,
                        "messages": messages,
                        "stream": True,
                        "think": False,
                        "options": {"temperature": 0.75, "top_p": 0.9, "num_predict": 384},
                    },
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        payload = json.loads(line)
                        content = payload.get("message", {}).get("content", "")
                        if content:
                            emitted = True
                            yield content
        except (httpx.HTTPError, json.JSONDecodeError, TypeError, ValueError) as exc:
            if emitted:
                raise OllamaError(f"Ollama 流式响应中断: {exc}") from exc
            try:
                answer = await self.chat(messages)
            except OllamaError as fallback_exc:
                raise OllamaError(f"Ollama 流式调用失败: {exc}; 回退失败: {fallback_exc}") from fallback_exc
            if answer:
                yield answer

    async def embed(self, texts: str | list[str]) -> list[list[float]]:
        inputs = [texts] if isinstance(texts, str) else texts
        if not inputs:
            return []

        openai_error: Exception | None = None
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.settings.ollama_openai_base_url}/embeddings",
                    json={"model": self.settings.embedding_model, "input": inputs},
                )
                response.raise_for_status()
                data = sorted(response.json()["data"], key=lambda item: item["index"])
                return [item["embedding"] for item in data]
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
            openai_error = exc

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.settings.ollama_base_url}/api/embed",
                    json={"model": self.settings.embedding_model, "input": inputs},
                )
                response.raise_for_status()
                return response.json()["embeddings"]
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
            raise OllamaError(
                f"Embedding OpenAI-compatible API 失败: {openai_error}; native API 失败: {exc}"
            ) from exc

    async def vision_chat(self, prompt: str, image_base64: str) -> str:
        if not self.settings.enable_vision:
            return "多模态功能已预留，当前版本暂未启用。"
        if not image_base64:
            raise OllamaError("启用图片理解时必须提供 image_base64")
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.settings.ollama_base_url}/api/chat",
                    json={
                        "model": self.settings.vision_model,
                        "messages": [
                            {"role": "user", "content": prompt, "images": [image_base64]}
                        ],
                        "stream": False,
                    },
                )
                response.raise_for_status()
                return response.json()["message"]["content"].strip()
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
            raise OllamaError(f"图片理解调用失败: {exc}") from exc
