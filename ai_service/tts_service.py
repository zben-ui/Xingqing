from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path
import re
import uuid

from websockets.asyncio.client import connect
from .config import Settings, settings


class TTSError(RuntimeError):
    pass


class TTSService:
    """Sambert voice adapter; credentials are never returned to the browser."""

    def __init__(self, app_settings: Settings = settings):
        self.settings = app_settings
        self.cache: dict[str, bytes] = {}
        self.lock = asyncio.Lock()

    def _key(self) -> str:
        key = os.getenv("DASHSCOPE_API_KEY", "").strip()
        if key:
            return key
        if self.settings.tts_reference_server:
            try:
                reference = Path(self.settings.tts_reference_server).read_text(encoding="utf-8-sig")
                match = re.search(r"(?:const|let|var)\s+API_KEY\s*=\s*['\"]([^'\"]+)['\"]", reference)
                return match.group(1) if match else ""
            except OSError:
                pass
        return ""

    def status(self) -> dict[str, object]:
        return {
            "enabled": self.settings.enable_tts, "configured": bool(self._key()),
            "provider": "aliyun", "model": self.settings.tts_model,
            "message": "仅将朗读的回复文本发送至阿里云，聊天推理仍使用本地 Ollama。",
        }

    async def synthesize(self, text: str) -> bytes:
        key = self._key()
        if not self.settings.enable_tts or not key:
            raise TTSError("阿里云语音未配置，可在后端设置 DASHSCOPE_API_KEY")
        digest = hashlib.sha256(text.encode()).hexdigest()
        async with self.lock:
            if digest in self.cache:
                return self.cache[digest]
            task_id = uuid.uuid4().hex
            task = {
                "header": {"action": "run-task", "task_id": task_id, "streaming": "out"},
                "payload": {
                    "task_group": "audio", "task": "tts", "function": "SpeechSynthesizer",
                    "model": self.settings.tts_model,
                    "parameters": {"format": "mp3", "sample_rate": 16000, "volume": 45,
                                   "rate": 1.0, "pitch": 1.05},
                    "input": {"text": text},
                },
            }
            audio = bytearray()
            async def collect():
                async with connect(self.settings.tts_ws_url,
                                   additional_headers={"Authorization": f"Bearer {key}"},
                                   max_size=10_000_000, open_timeout=12) as websocket:
                    await websocket.send(json.dumps(task, ensure_ascii=False))
                    async for message in websocket:
                        if isinstance(message, bytes):
                            audio.extend(message)
                            if len(audio) > 10_000_000:
                                raise TTSError("语音回复过长，请缩短内容后重试")
                            continue
                        event = json.loads(message).get("header", {}).get("event")
                        if event == "task-failed":
                            raise TTSError("阿里云语音合成失败，请检查后端密钥与额度")
                        if event == "task-finished":
                            if not audio:
                                raise TTSError("语音服务没有返回音频")
                            return bytes(audio)
                raise TTSError("语音服务提前断开，请重试")
            try:
                result = await asyncio.wait_for(collect(), timeout=55)
                if len(self.cache) >= 8:
                    self.cache.pop(next(iter(self.cache)))
                self.cache[digest] = result
                return result
            except TTSError:
                raise
            except Exception:
                raise TTSError("语音连接暂不可用，请稍后重试") from None
