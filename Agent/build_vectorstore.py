# -*- coding: utf-8 -*-
"""
从 knowledge/psych_rag_kb.csv 构建本地 Chroma 向量库（持久化到 chroma_db/，collection: psych_kb）。

依赖：pandas、chromadb、ollama
前置：Ollama 已启动，并已拉取 qwen3-embedding:0.6b
    ollama pull qwen3-embedding:0.6b
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

# -----------------------------------------------------------------------------
# 可配置项（环境变量可覆盖，便于在 Windows/CI/远程 Ollama 下使用）
# -----------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
# 亦可用环境变量 PSYCH_KB_CSV 指定 CSV 的绝对路径
DEFAULT_CSV = SCRIPT_DIR / "knowledge" / "psych_rag_kb.csv"

CHROMA_PERSIST_DIR = Path(os.environ.get("CHROMA_PERSIST_DIR", str(SCRIPT_DIR / "chroma_db"))).resolve()
COLLECTION_NAME = os.environ.get("CHROMA_COLLECTION_PSYCH_KB", "psych_kb")

# 与本机 `ollama list` 中名称一致，含 :tag
OLLAMA_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "qwen3-embedding:0.6b")

EMBED_BATCH_SIZE = max(1, int(os.environ.get("EMBED_BATCH_SIZE", "8")))
CHROMA_ADD_BATCH = max(1, int(os.environ.get("CHROMA_ADD_BATCH", "32")))

# 语义检索常用 cosine；如效果异常可换 l2 做对比
HNSW_SPACE = os.environ.get("CHROMA_HNSW_SPACE", "cosine")

REQUIRED_COLUMNS: tuple[str, ...] = (
    "id",
    "topic",
    "source_type",
    "applicable_clusters",
    "applicable_flags",
    "risk_level",
    "content",
    "source",
)

METADATA_COLS: tuple[str, ...] = (
    "id",
    "topic",
    "source_type",
    "applicable_clusters",
    "applicable_flags",
    "risk_level",
    "source",
)


def _scalar_for_chroma(value: Any) -> str | int | float | bool:
    """将单元格值转为 Chroma 允许的 metadata 标量，禁止 None / 非标准类型。"""
    if value is None:
        return ""
    try:
        import math

        if isinstance(value, float) and math.isnan(value):
            return ""
    except (TypeError, ValueError):
        pass
    try:
        import pandas as pd

        if pd.isna(value):
            return ""
    except (ImportError, TypeError, ValueError):
        pass
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        s = value.strip()
        if s == "" or s.lower() in ("nan", "none", "<na>"):
            return ""
        return s
    if isinstance(value, (int,)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, float) and not isinstance(value, bool):
        return float(value)
    return str(value)


def _validate_columns(df: Any) -> None:
    import pandas as pd

    if not isinstance(df, pd.DataFrame):
        raise TypeError("内部错误：DataFrame 类型不正确。")
    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV 缺少必要列: {missing}。需要: {list(REQUIRED_COLUMNS)}"
        )


def _embeddings_from_ollama_response(response: Any) -> list[list[float]]:
    """
    将 ollama.embed 的返回统一为 list[list[float]]，与「输入条数」一致。

    兼容: dict/类 dict 的 embeddings；单数 embedding；以及只带 .embedding 属性的响应对象。
    """
    if response is None:
        raise ValueError("embed 响应为 None。")

    embs: Any = None
    if isinstance(response, Mapping):
        m = dict(response)
        if m.get("embeddings") is not None:
            embs = m["embeddings"]
        elif "embedding" in m:
            embs = [m["embedding"]]
    if embs is None:
        embs = getattr(response, "embeddings", None)
    if embs is None and hasattr(response, "embedding"):
        one = getattr(response, "embedding", None)
        if one is not None:
            embs = [one]
    if embs is None:
        raise TypeError(f"无法解析 embed 响应: {type(response)!r}")
    if not isinstance(embs, (list, tuple)) or not embs:
        raise ValueError("embed 结果为空或不是列表。")

    first = embs[0]
    if not isinstance(first, (list, tuple)) and all(
        isinstance(x, (int, float)) for x in embs
    ):
        return [list(map(float, cast(Sequence[int | float], embs)))]
    return [list(map(float, row)) for row in embs]  # type: ignore[union-attr]


def _call_ollama_embed(model: str, inputs: str | list[str]) -> list[list[float]]:
    try:
        import ollama
    except ImportError as e:
        raise ImportError("请先安装: pip install ollama") from e

    try:
        response = ollama.embed(model=model, input=inputs)
    except Exception as e:  # noqa: BLE001
        host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434 (默认)")
        raise RuntimeError(
            f"Ollama 请求失败: {e}\n"
            f"  可检查: 1) 服务已启动; 2) 已 ollama pull {model}"
            f"; 3) 若远程部署请设置 OLLAMA_HOST; 当前参考: {host}"
        ) from e
    return _embeddings_from_ollama_response(response)


def _embed_texts_batched(model: str, texts: list[str]) -> list[list[float]]:
    """分批向 Ollama 请求；整批失败则对该批退化为逐条。"""
    n = len(texts)
    if n == 0:
        return []
    all_vecs: list[list[float]] = []
    i = 0
    while i < n:
        end = min(i + EMBED_BATCH_SIZE, n)
        batch = texts[i:end]
        try:
            vecs = _call_ollama_embed(model, batch)
        except (RuntimeError, TypeError, ValueError, KeyError):
            if len(batch) == 1:
                raise
            for t in batch:
                all_vecs.append(_call_ollama_embed(model, t)[0])
            i = end
            continue
        if len(vecs) == len(batch):
            all_vecs.extend(vecs)
            i = end
            continue
        for t in batch:
            all_vecs.append(_call_ollama_embed(model, t)[0])
        i = end
    if len(all_vecs) != n:
        raise RuntimeError(
            f"嵌入条数与文本不一致: 得到 {len(all_vecs)} 条, 需要 {n} 条。"
        )
    return all_vecs


def _rebuild_or_create_collection() -> Any:
    """删除同名 collection（若存在）后重新创建。返回 chromadb 的 Collection 对象。"""
    import chromadb

    client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
    names = {c.name for c in client.list_collections()}
    if COLLECTION_NAME in names:
        client.delete_collection(COLLECTION_NAME)
    return client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": HNSW_SPACE},
    )


def _chroma_add_in_batches(
    collection: Any,
    ids: list[str],
    documents: list[str],
    metadatas: list[dict[str, str | int | float | bool]],
    embeddings: list[list[float]],
) -> None:
    n = len(ids)
    for s in range(0, n, CHROMA_ADD_BATCH):
        e = min(s + CHROMA_ADD_BATCH, n)
        collection.add(
            ids=ids[s:e],
            documents=documents[s:e],
            metadatas=metadatas[s:e],  # type: ignore[arg-type]
            embeddings=embeddings[s:e],
        )


def build_vectorstore() -> int:
    """
    读 CSV、嵌入、写入 Chroma。成功返回写入条数，失败则抛错由 main 捕获。
    """
    import pandas as pd

    csv_path = Path(os.environ.get("PSYCH_KB_CSV", str(DEFAULT_CSV))).resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"找不到知识库文件: {csv_path}")

    # Windows 下用 utf-8-sig 以兼容带 BOM 的 Excel 导出
    df = pd.read_csv(csv_path, encoding="utf-8-sig", dtype=object)
    _validate_columns(df)
    if df.empty:
        raise ValueError("CSV 无数据行，无法建库。")

    # 行级 metadata + 去空 content
    doc_ids: list[str] = []
    documents: list[str] = []
    metas: list[dict[str, str | int | float | bool]] = []

    for _idx, row in df.iterrows():
        raw = row.get("content")
        if raw is None or (isinstance(raw, float) and pd.isna(raw)):
            continue
        text = str(raw).strip()
        if not text:
            continue
        row_id = row.get("id")
        if row_id is None or (isinstance(row_id, float) and pd.isna(row_id)):
            raise ValueError("存在行缺少非空 id。")
        sid = str(row_id).strip()
        if not sid:
            raise ValueError("存在行 id 为空字符串。")
        m: dict[str, str | int | float | bool] = {
            c: _scalar_for_chroma(row.get(c)) for c in METADATA_COLS
        }
        doc_ids.append(sid)
        documents.append(text)
        metas.append(m)

    if not doc_ids:
        raise ValueError("无有效行（需至少一条非空 content）。")

    print(f"正在用 Ollama 模型 {OLLAMA_EMBED_MODEL!r} 计算 {len(documents)} 条向量…")
    embeddings = _embed_texts_batched(OLLAMA_EMBED_MODEL, documents)
    if len(embeddings) != len(documents):
        raise RuntimeError("嵌入数量与有效文档数不一致。")

    # 与 content 一一对应；若 Ollama 对顺序有歧义，上面已逐批保证顺序
    for i, emb in enumerate(embeddings):
        if len(emb) == 0:
            raise ValueError(f"第 {i+1} 条嵌入为空向量。")

    print("正在重建 Chroma collection 并写入（含持久化）…")
    collection = _rebuild_or_create_collection()
    _chroma_add_in_batches(collection, doc_ids, documents, metas, embeddings)
    n_out = int(collection.count())
    if n_out != len(doc_ids):
        raise RuntimeError(f"Chroma 计数异常: count={n_out}, 预期={len(doc_ids)}")
    return n_out


def main() -> int:
    try:
        n = build_vectorstore()
    except (OSError, ValueError, FileNotFoundError, RuntimeError, ImportError) as e:
        print(f"建库失败: {e}", file=sys.stderr)
        return 1
    print("-" * 60)
    print(f"成功写入 {n} 条知识到 collection {COLLECTION_NAME!r}。")
    print(f"向量库保存位置: {CHROMA_PERSIST_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
