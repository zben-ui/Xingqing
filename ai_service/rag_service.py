from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, asdict, field
from pathlib import Path

from .config import Settings, settings
from .ollama_client import OllamaClient, OllamaError
from .agent_knowledge_service import AgentKnowledgeService


@dataclass
class KnowledgeChunk:
    file: str
    chunk: int
    text: str
    fingerprint: str
    embedding: list[float] | None = None
    metadata: dict[str, str] = field(default_factory=dict)


class RagService:
    def __init__(self, ollama: OllamaClient, app_settings: Settings = settings) -> None:
        self.ollama = ollama
        self.settings = app_settings
        self.corpus = AgentKnowledgeService(app_settings.agent_data_dir)
        self.chunks: list[KnowledgeChunk] = []
        self.last_mode = "keyword"
        self.settings.knowledge_base_dir.mkdir(parents=True, exist_ok=True)
        self.settings.vector_store_path.parent.mkdir(parents=True, exist_ok=True)
        self.load_store()

    def _files(self) -> list[Path]:
        files: list[Path] = []
        for pattern in ("*.txt", "*.md"):
            files.extend(self.settings.knowledge_base_dir.rglob(pattern))
        return sorted(files)

    def _split_text(self, text: str) -> list[str]:
        clean = re.sub(r"\r\n?", "\n", text)
        clean = re.sub(r"[ \t]+", " ", clean).strip()
        size = max(100, self.settings.rag_chunk_size)
        overlap = min(max(0, self.settings.rag_chunk_overlap), size // 2)
        chunks: list[str] = []
        start = 0
        while start < len(clean):
            end = min(len(clean), start + size)
            if end < len(clean):
                boundary = max(clean.rfind("\n", start, end), clean.rfind("。", start, end))
                if boundary > start + size // 2:
                    end = boundary + 1
            piece = clean[start:end].strip()
            if piece:
                chunks.append(piece)
            if end >= len(clean):
                break
            start = max(start + 1, end - overlap)
        return chunks

    def read_documents(self) -> list[KnowledgeChunk]:
        result: list[KnowledgeChunk] = []
        for path in self._files():
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                text = path.read_text(encoding="gb18030", errors="ignore")
            relative = path.relative_to(self.settings.knowledge_base_dir).as_posix()
            for index, piece in enumerate(self._split_text(text), start=1):
                digest = hashlib.sha256(f"{relative}:{index}:{piece}".encode("utf-8")).hexdigest()
                result.append(KnowledgeChunk(relative, index, piece, digest))
        for index, record in enumerate(self.corpus.records, start=1):
            # Cluster-specific research descriptions are not applicable to our custom self-test.
            if record.get("applicable_clusters"):
                continue
            piece = f"主题：{record.get('topic', '')}\n{record['content']}"
            relative = "Agent/knowledge/psych_rag_kb.csv"
            metadata = {key: record.get(key, "") for key in ("id", "topic", "source_type", "source")}
            digest = hashlib.sha256(f"{relative}:{index}:{piece}".encode()).hexdigest()
            result.append(KnowledgeChunk(relative, index, piece, digest, metadata=metadata))
        return result

    def load_store(self) -> None:
        path = self.settings.vector_store_path
        if not path.exists():
            self.chunks = self.read_documents()
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            cached = {item["fingerprint"]: item for item in payload.get("chunks", [])}
            self.chunks = self.read_documents()
            for chunk in self.chunks:
                chunk.embedding = cached.get(chunk.fingerprint, {}).get("embedding")
            self.last_mode = payload.get("mode", "keyword")
            # A partial old index must not hide newly added Agent/private documents.
            if any(chunk.embedding is None for chunk in self.chunks):
                for chunk in self.chunks:
                    chunk.embedding = None
                self.last_mode = "keyword"
        except (OSError, ValueError, TypeError):
            self.chunks = self.read_documents()

    async def rebuild(self) -> dict[str, object]:
        chunks = self.read_documents()
        mode = "embedding"
        error: str | None = None
        if chunks:
            try:
                batch_size = 16
                for start in range(0, len(chunks), batch_size):
                    batch = chunks[start : start + batch_size]
                    vectors = await self.ollama.embed([item.text for item in batch])
                    if len(vectors) != len(batch):
                        raise OllamaError("Embedding 返回数量与文本数量不一致")
                    for item, vector in zip(batch, vectors):
                        item.embedding = vector
            except OllamaError as exc:
                mode = "keyword"
                error = str(exc)
                for item in chunks:
                    item.embedding = None

        self.chunks = chunks
        payload = {
            "embedding_model": self.settings.embedding_model,
            "mode": mode,
            "chunks": [asdict(item) for item in chunks],
        }
        temp_path = self.settings.vector_store_path.with_suffix(".tmp")
        temp_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        temp_path.replace(self.settings.vector_store_path)
        self.last_mode = mode
        return {"documents": len(self._files()) + int(bool(self.corpus.records)), "chunks": len(chunks), "mode": mode, "error": error}

    @staticmethod
    def _cosine(left: list[float], right: list[float]) -> float:
        if not left or len(left) != len(right):
            return 0.0
        dot = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0

    @staticmethod
    def _keywords(text: str) -> set[str]:
        lowered = text.lower()
        latin = set(re.findall(r"[a-z0-9_]{2,}", lowered))
        chinese_runs = re.findall(r"[\u4e00-\u9fff]+", lowered)
        chinese: set[str] = set()
        for run in chinese_runs:
            if len(run) == 1:
                chinese.add(run)
            else:
                chinese.update(run[index : index + 2] for index in range(len(run) - 1))
        return latin | chinese

    @staticmethod
    def _visible_chunks(chunks: list[KnowledgeChunk], user_id: int | None) -> list[KnowledgeChunk]:
        prefix = f"users/{user_id}/" if user_id is not None else ""
        return [
            chunk
            for chunk in chunks
            if not chunk.file.startswith("users/") or (prefix and chunk.file.startswith(prefix))
        ]

    def _keyword_search(
        self, query: str, top_k: int, chunks: list[KnowledgeChunk] | None = None
    ) -> list[dict[str, object]]:
        query_terms = self._keywords(query)
        ranked: list[tuple[float, KnowledgeChunk]] = []
        source_chunks = chunks if chunks is not None else self.chunks
        for chunk in source_chunks:
            terms = self._keywords(chunk.text)
            overlap = len(query_terms & terms)
            score = overlap / max(1, len(query_terms))
            if score > 0:
                ranked.append((score, chunk))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return [self._source(chunk, score) for score, chunk in ranked[:top_k]]

    @staticmethod
    def _source(chunk: KnowledgeChunk, score: float) -> dict[str, object]:
        excerpt = chunk.text[:180] + ("…" if len(chunk.text) > 180 else "")
        return {
            "file": chunk.file,
            "chunk": chunk.chunk,
            "score": round(float(score), 4),
            "excerpt": excerpt,
            "text": chunk.text,
            "metadata": chunk.metadata,
        }

    async def search(
        self, query: str, top_k: int | None = None, user_id: int | None = None
    ) -> list[dict[str, object]]:
        limit = top_k or self.settings.rag_top_k
        if not self.chunks:
            self.chunks = self.read_documents()
        if not self.chunks:
            return []

        visible_chunks = self._visible_chunks(self.chunks, user_id)
        embedded_chunks = [chunk for chunk in visible_chunks if chunk.embedding]
        if embedded_chunks:
            try:
                query_vector = (await self.ollama.embed(query))[0]
                ranked = [
                    (self._cosine(query_vector, chunk.embedding or []), chunk)
                    for chunk in embedded_chunks
                ]
                ranked.sort(key=lambda item: item[0], reverse=True)
                self.last_mode = "embedding"
                return [self._source(chunk, score) for score, chunk in ranked[:limit] if score > 0.2]
            except (OllamaError, IndexError):
                pass

        self.last_mode = "keyword"
        return self._keyword_search(query, limit, visible_chunks)
