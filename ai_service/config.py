from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _path_from_env(name: str, default: str) -> Path:
    value = Path(os.getenv(name, default))
    return value if value.is_absolute() else PROJECT_ROOT / value


@dataclass(frozen=True)
class Settings:
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    ollama_openai_base_url: str = os.getenv(
        "OLLAMA_OPENAI_BASE_URL", "http://localhost:11434/v1"
    ).rstrip("/")
    llm_provider: str = os.getenv("LLM_PROVIDER", "ollama")
    llm_model: str = os.getenv("LLM_MODEL", "qwen3.5:9b")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "qwen3-embedding:0.6b")
    vision_model: str = os.getenv("VISION_MODEL", "llava:latest")
    rag_top_k: int = int(os.getenv("RAG_TOP_K", "4"))
    rag_chunk_size: int = int(os.getenv("RAG_CHUNK_SIZE", "500"))
    rag_chunk_overlap: int = int(os.getenv("RAG_CHUNK_OVERLAP", "80"))
    memory_max_turns: int = int(os.getenv("MEMORY_MAX_TURNS", "10"))
    digital_human_name: str = os.getenv("DIGITAL_HUMAN_NAME", "小艾")
    enable_vision: bool = _as_bool(os.getenv("ENABLE_VISION"), False)
    enable_tts: bool = _as_bool(os.getenv("ENABLE_TTS"), False)
    tts_model: str = os.getenv("TTS_MODEL", "sambert-zhiwei-v1")
    tts_ws_url: str = os.getenv(
        "TTS_WS_URL", "wss://dashscope.aliyuncs.com/api-ws/v1/inference/"
    )
    tts_reference_server: str = os.getenv("TTS_REFERENCE_SERVER", "")
    agent_data_dir: Path = _path_from_env("AGENT_DATA_DIR", "Agent")
    ollama_timeout_seconds: float = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "120"))
    app_host: str = os.getenv("APP_HOST", "127.0.0.1")
    app_port: int = int(os.getenv("APP_PORT", "8000"))
    knowledge_base_dir: Path = _path_from_env("KNOWLEDGE_BASE_DIR", "knowledge_base")
    vector_store_path: Path = _path_from_env("VECTOR_STORE_PATH", "data/vector_store.json")
    database_path: Path = _path_from_env("DATABASE_PATH", "data/xiaoai.db")
    system_prompt_path: Path = _path_from_env(
        "SYSTEM_PROMPT_PATH", "ai_service/prompts/xiaoai_system.md"
    )


settings = Settings()
