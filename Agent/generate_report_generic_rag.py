# -*- coding: utf-8 -*-
"""
普通 RAG 版：KMeans 分群 + 群体画像 + Chroma 向量检索（**仅**由激活心理 flags
与自杀意念信号构成检索问句，**不使用**分群名、**不使用**课程类别定向逻辑）+ 本地 LLM 生成 + 安全审阅。

用法（项目根目录）:
  python generate_report_generic_rag.py  students_eval/student1.json

环境变量（与 run_agent 一致，另可选）:
  OLLAMA_LLM, OLLAMA_EMBED, OLLAMA_HOST, LLM_TEMPERATURE,
  CHROMA_DIR, CHROMA_COLLECTION, MODELS_DIR, KMEANS_MODEL_PATH, SCALER_PATH, FEATURE_ORDER_PATH,
  CLUSTER_KNOWLEDGE_PATH, RISK_RULES_PATH, REPORT_STYLE_PATH,
  GENERIC_RAG_N_RESULTS（单次检索取回条数上界，默认 32）
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

SCRIPT_DIR = Path(__file__).resolve().parent

PROMPT_VERSION = "generic_rag_v1"
METHOD = "kmeans_chroma_generic_rag"
DEFAULT_OUT_DIR = SCRIPT_DIR / "outputs_eval" / "generic_rag"
DEFAULT_LOG_DIR = SCRIPT_DIR / "logs_eval" / "generic_rag"

RAG_PREAMBLE = (
    "（本次为**普通 RAG**：检索问句**仅**由当前激活的心理筛查 flags 与"
    "（若存在）自杀意念信号共同构成，**未**在问句中使用分群名称，也**未**使用"
    "课程类别支持感知/定向检索逻辑。以下片段与【输入信息】、写作与边界要求一并理解。）\n\n"
)


def _kb_fingerprint(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for p in sorted(paths, key=lambda x: f"{x.name}\0{str(x)}"):
        h.update(p.resolve().as_posix().encode("utf-8", errors="replace"))
        h.update(b":")
        if p.is_file():
            h.update(p.read_bytes())
        else:
            h.update(b"MISSING")
    return f"sha256:{h.hexdigest()[:20]}"


def _iso_timestamp() -> str:
    return (
        datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    )


def _prepare_student(data: dict[str, Any]) -> dict[str, Any]:
    s = {**data}
    s.pop("name", None)
    return s


def _meta_score_flags_only(
    student: Mapping[str, Any],
    active_flags: list[str],
    meta: Mapping[str, Any] | None,
) -> float:
    """
    重排分：不使用 cluster；仅 applicable_flags 与活动 flags 的交集、
    以及自杀相关 topic / risk（与 run_agent 一致思路，但去掉分群项）。
    """
    from run_agent import _empty_meta_str

    if not meta:
        return 0.0
    from utils import is_suicidal_ideation_active

    score = 0.0
    afm = meta.get("applicable_flags", "")
    topic = str(meta.get("topic", "")).lower()
    risk = str(meta.get("risk_level", "")).lower()

    if _empty_meta_str(afm):
        score += 0.8
    else:
        parts = re.split(r"[,;，；\s]+", str(afm).strip()) if str(afm).strip() else []
        for p in parts:
            p = p.strip()
            if p and p in active_flags:
                score += 2.0

    if is_suicidal_ideation_active(cast(Any, student)) and (
        "suicide" in topic
        or "suicide" in str(meta)
        or "自杀" in str(meta.get("applicable_flags", ""))
    ):
        score += 1.2
    if risk in ("high",) and is_suicidal_ideation_active(cast(Any, student)):
        score += 0.5
    return score


def build_generic_rag_query(
    active_flags: list[str],
    flag_terms: dict[str, str],
) -> str:
    """
    构造**唯一**检索问句：仅由激活的筛查维度用语组成（含 suicidal_ideation 对应用语）。
    不得包含分群名、课程类型（数学/核心/语言）定向语。
    """
    if not active_flags:
        return "高校学生 心理健康教育 心理支持 一般性知识 非临床诊断 支持性解释"

    terms: list[str] = []
    for f in sorted(active_flags):
        if f in flag_terms:
            terms.append(flag_terms[f])
        else:
            t = f.replace("_", " ").strip()
            if t:
                terms.append(t)
    seen: set[str] = set()
    uniq: list[str] = []
    for t in terms:
        if t and t not in seen:
            seen.add(t)
            uniq.append(t)
    if not uniq:
        return "高校学生 心理健康教育 心理支持 一般性知识 非临床诊断 支持性解释"
    return (
        "心理教育知识；高校学生支持；个体化信息；SCL-90 与相关筛查；非诊断化表述；\n"
        f"当前在筛查层面需关注的心理维度与信号：{'；'.join(uniq)}。"
    )


def _chroma_query_generic(
    col: Any,
    query_text: str,
    student: Mapping[str, Any],
    active_flags: list[str],
    n_results: int,
) -> list[tuple[float, str, dict[str, Any]]]:
    """单次 Chroma 查询 + 分数字段融合 + 按 id/正文去重。"""
    from run_agent import _dedupe_id, embed_query_text

    if n_results < 1:
        return []
    qvec = embed_query_text(query_text)
    try:
        res = col.query(
            query_embeddings=[qvec],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"Chroma 查询失败: {e}") from e
    docs = (res.get("documents") or [[]])[0] or []
    metas = (res.get("metadatas") or [[]])[0] or []
    dists = (res.get("distances") or [[]])[0] or []
    n = min(len(docs), len(metas))
    if len(dists) < n:
        dists = list(dists) + [0.0] * (n - len(dists))
    out: list[tuple[float, str, dict[str, Any]]] = []
    local_seen: set[str] = set()
    for i in range(n):
        doc = str(docs[i] or "")
        me = metas[i] if i < len(metas) and isinstance(metas[i], dict) else {}
        d = dists[i] if i < len(dists) else 0.0
        rscore = _meta_score_flags_only(student, active_flags, me)
        dist_part = 1.0 / (1.0 + float(abs(d)))
        total = rscore * 0.5 + dist_part
        kid = _dedupe_id(me, doc)
        if kid in local_seen:
            continue
        local_seen.add(kid)
        m2 = {**me, "retrieval_stage": "generic_rag"}
        out.append((total, doc, m2))
    out.sort(key=lambda x: -x[0])
    return out


def format_generic_rag_for_prompt(
    items: list[tuple[float, str, dict[str, Any]]],
) -> str:
    if not items:
        return "（本档未命中外部检索片段；请依据学生数据与 `cluster_profile_knowledge` 撰写。）"
    lines: list[str] = []
    for i, (sc, text, m) in enumerate(items, 1):
        if not (text and str(text).strip()):
            continue
        st = m.get("retrieval_stage", "generic_rag")
        lines.append(
            f"【片段 {i}】 阶段={st} | 分 {sc:.3f} | topic={m.get('topic', '')!s} | "
            f"risk={m.get('risk_level', '')!s} | source={m.get('source', '')!s}\n{str(text).strip()}\n"
        )
    if not lines:
        return "（本档未命中有效检索正文；请依据学生数据与群体画像知识撰写。）"
    return "\n".join(lines)


def _unique_topics(items: list[tuple[float, str, dict[str, Any]]]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for _, _, m in items:
        t = m.get("topic")
        s = str(t).strip() if t is not None else ""
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def retrieve_generic_rag(
    student: Mapping[str, Any],
    active_flags: list[str],
) -> tuple[str, list[tuple[float, str, dict[str, Any]]], int, int, str]:
    """
    普通 RAG：单问句、单次检索、结果已按 _dedupe_id 去重。

    返回: (retrieval_query, 排序后的条列表, chroma 文档条数, n_results 请求数, chroma_dir 字符串)
    """
    import chromadb  # 延迟导入

    from run_agent import CHROMA_COLLECTION, CHROMA_DIR, RETRIEVAL_TOP_K, _FLAG_QUERY_TERMS

    q = build_generic_rag_query(list(active_flags), _FLAG_QUERY_TERMS)
    chroma_dir_s = str(CHROMA_DIR.resolve().as_posix())

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        col = client.get_collection(CHROMA_COLLECTION)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            f"无法打开 Chroma 集合 {CHROMA_COLLECTION!r}，路径: {CHROMA_DIR}。请先运行建库。{e}"
        ) from e
    try:
        count_fn = col.count
        n_docs = int(count_fn() if callable(count_fn) else count_fn)
    except (TypeError, ValueError, AttributeError):
        n_docs = 0
    if n_docs <= 0:
        raise RuntimeError("Chroma 集合为空，请先完成向量库构建。")
    n_req = int(os.environ.get("GENERIC_RAG_N_RESULTS", "32"))
    n_req = max(1, min(n_req, n_docs))
    n_final_cap = max(1, int(RETRIEVAL_TOP_K))
    raw = _chroma_query_generic(
        col, q, student, list(active_flags), n_results=n_req
    )
    merged = raw[: min(n_final_cap, len(raw))]
    return q, merged, n_docs, n_req, chroma_dir_s


def main() -> int:
    from run_agent import (
        CLUSTER_KNOWLEDGE_PATH,
        OLLAMA_LLM,
        RISK_RULES_PATH,
        REPORT_STYLE_PATH,
        build_system_prompt,
        build_user_prompt,
        call_llm_markdown,
        classify_student,
        find_cluster_block,
        safety_review,
    )
    from utils import (
        ensure_dir,
        get_active_flags,
        get_psychological_profile,
        load_json,
        load_cluster_knowledge,
        normalize_text,
        save_json,
        save_markdown,
    )

    parser = argparse.ArgumentParser(
        description="普通 RAG：分群 + 仅 flags 问句的 Chroma 检索 + 报告生成",
    )
    parser.add_argument("student_json", type=Path, help="仅心理版学生 JSON 路径")
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUT_DIR, help="报告输出目录"
    )
    parser.add_argument(
        "--log-dir", type=Path, default=DEFAULT_LOG_DIR, help="运行日志 JSON 目录"
    )
    args = parser.parse_args()
    spath = args.student_json.resolve()
    out_dir = args.output_dir.resolve()
    log_dir = args.log_dir.resolve()

    if not spath.is_file():
        print(f"未找到学生文件: {spath}", file=sys.stderr)
        return 1
    try:
        raw = load_json(spath)
    except (OSError, json.JSONDecodeError, ValueError) as e:
        print(f"读取学生 JSON 失败: {e}", file=sys.stderr)
        return 1
    if not isinstance(raw, dict):
        print("学生档案必须是 JSON 对象。", file=sys.stderr)
        return 1

    student: dict[str, Any] = _prepare_student(raw)
    student_id = str(student.get("student_id", "unknown")).strip() or "unknown"
    if not get_psychological_profile(cast(Any, student)):
        print("需包含非空的 psychological_profile（仅心理输入版）。", file=sys.stderr)
        return 1

    # 1) 分群（cluster_name 仅写入档案供画像与审阅/风险规则，**不**进入检索问句）
    try:
        cls_out = classify_student(cast(Any, student))
    except (FileNotFoundError, ValueError, OSError, RuntimeError) as e:
        print(f"分群预测失败: {e}", file=sys.stderr)
        return 1
    student = {
        **student,
        "cluster": cls_out["cluster_id"],
        "cluster_name": cls_out["cluster_name"],
    }
    cname = str(cls_out["cluster_name"]).strip()

    kb_paths = [CLUSTER_KNOWLEDGE_PATH, RISK_RULES_PATH]
    if REPORT_STYLE_PATH.is_file():
        kb_paths.append(REPORT_STYLE_PATH)
    knowledge_base_version = _kb_fingerprint([Path(p) for p in kb_paths])

    try:
        cluster_data = load_cluster_knowledge(CLUSTER_KNOWLEDGE_PATH)
    except (OSError, TypeError, ValueError) as e:
        print(f"读取分群知识库失败: {e}", file=sys.stderr)
        return 1
    cblock = find_cluster_block(cluster_data, cname)
    sc_raw = cluster_data.get("study_context", {})
    study_context = (
        json.dumps(sc_raw, ensure_ascii=False, indent=2) if sc_raw else "（无）"
    )
    try:
        cluster_json_text = json.dumps(
            cluster_data, ensure_ascii=False, indent=2, allow_nan=False
        )
    except (TypeError, ValueError) as e:
        print(f"序列化分群知识失败: {e}", file=sys.stderr)
        return 1

    try:
        risk_raw = load_json(RISK_RULES_PATH)
    except (OSError, json.JSONDecodeError) as e:
        print(f"读取 risk_rules.json 失败: {e}", file=sys.stderr)
        return 1
    if not isinstance(risk_raw, list):
        print("risk_rules.json 期望为 JSON 数组。", file=sys.stderr)
        return 1

    _style: dict[str, Any] | None = None
    if REPORT_STYLE_PATH.is_file():
        try:
            so = load_json(REPORT_STYLE_PATH)
            if isinstance(so, dict):
                _style = so
        except (OSError, json.JSONDecodeError):
            _style = None

    # 2) 激活 flags（含 suicidal_ideation=1 时）→ 唯一检索问句
    active = get_active_flags(cast(Any, student))
    try:
        retrieval_query, ranked, n_chroma_docs, n_requested, chroma_dir_s = (
            retrieve_generic_rag(student, active)
        )
    except RuntimeError as e:
        print(f"检索阶段失败: {e}", file=sys.stderr)
        return 1

    rag_body = format_generic_rag_for_prompt(ranked)
    rag_text = RAG_PREAMBLE + rag_body

    system_prompt = build_system_prompt()
    system_prompt += (
        "\n【本运行：普通 RAG】检索问句仅由心理筛查 flags 与（若存在）自杀意念信号构成，"
        "未使用分群名或课程类别构造检索。写作须：不编造个体真实学业成绩；不将分群写为"
        "临床诊断；不虚构姓名；若 suicidal_ideation=1 须优先提示专业/危机支持。"
    )
    if _style and "writing_constraints" in _style:
        wc = _style.get("writing_constraints") or []
        if wc:
            system_prompt += "\n附加写作约束: " + "；".join(str(x) for x in list(wc)[:8])
    if _style and isinstance(_style.get("tone"), dict):
        t = _style.get("tone")
        if isinstance(t, dict) and t.get("overall"):
            system_prompt += f"\n语气参考: {t.get('overall')}"

    try:
        user_prompt = build_user_prompt(
            cast(Any, student),
            study_context,
            cblock,
            cluster_json_text,
            rag_text,
        )
        report = call_llm_markdown(user_prompt, system_prompt)
    except RuntimeError as e:
        print(f"生成阶段失败: {e}", file=sys.stderr)
        return 1

    try:
        report = safety_review(report, student, risk_raw)
        report = normalize_text(report)
    except (OSError, TypeError) as e:
        print(f"后处理失败: {e}", file=sys.stderr)
        return 1

    ensure_dir(out_dir)
    ensure_dir(log_dir)
    report_path = out_dir / f"{student_id}_generic_rag.md"
    log_path = log_dir / f"{student_id}_generic_rag.json"
    try:
        save_markdown(report_path, report)
    except OSError as e:
        print(f"保存报告失败: {e}", file=sys.stderr)
        return 1

    topics = _unique_topics(ranked)

    log_obj: dict[str, Any] = {
        "student_id": student_id,
        "method": METHOD,
        "cluster_id": cls_out["cluster_id"],
        "cluster_name": cls_out["cluster_name"],
        "generation_model": OLLAMA_LLM,
        "prompt_version": PROMPT_VERSION,
        "knowledge_base_version": knowledge_base_version,
        "report_file": str(report_path.resolve().as_posix()),
        "timestamp": _iso_timestamp(),
        "retrieval_query": retrieval_query,
        "retrieved_doc_count": len(ranked),
        "retrieved_topics": topics,
        "chroma_dir": chroma_dir_s,
        "chroma_collection_doc_count": n_chroma_docs,
        "retrieval_n_requested": n_requested,
    }
    try:
        save_json(log_path, log_obj, ensure_ascii=False)
    except OSError as e:
        print(f"保存日志失败: {e}", file=sys.stderr)
        return 1

    print("完成。")
    print(f"  学生: {student_id}")
    print(f"  分群: cluster_id={cls_out['cluster_id']}  {cls_out['cluster_name']}")
    print(f"  检索问句: {retrieval_query[:80]}…" if len(retrieval_query) > 80 else f"  检索问句: {retrieval_query}")
    print(f"  检索条数: {len(ranked)}  topic 数: {len(topics)}")
    print(f"  报告: {report_path}")
    print(f"  日志: {log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
