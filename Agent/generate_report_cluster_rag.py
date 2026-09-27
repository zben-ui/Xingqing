# -*- coding: utf-8 -*-
"""
分群驱动 + 课程类别支持感知 RAG：KMeans 分群 + 四阶段 Chroma 检索
（cluster / feature / course-impact / risk）+ 五段式报告 + 安全审阅。

与 ``run_agent.retrieve_psych_knowledge`` 检索逻辑一致；输出与日志落盘在
``outputs_eval/cluster_rag``、``logs_eval/cluster_rag``。

用法:
  python generate_report_cluster_rag.py  students_eval/student1.json

环境变量（与 run_agent 一致，可选）:
  OLLAMA_LLM, OLLAMA_EMBED, OLLAMA_HOST, LLM_TEMPERATURE,
  CHROMA_DIR, CHROMA_COLLECTION, RETRIEVAL_PER_STAGE, RETRIEVAL_TOP_K,
  MODELS_DIR, KMEANS_MODEL_PATH, SCALER_PATH, FEATURE_ORDER_PATH,
  CLUSTER_KNOWLEDGE_PATH, RISK_RULES_PATH, REPORT_STYLE_PATH
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

SCRIPT_DIR = Path(__file__).resolve().parent
KNOWLEDGE_DIR = SCRIPT_DIR / "knowledge"
PSYCH_RAG_CSV = KNOWLEDGE_DIR / "psych_rag_kb.csv"

PROMPT_VERSION = "cluster_rag_v1"
METHOD = "kmeans_chroma_four_stage_cluster_course_rag"
DEFAULT_OUT_DIR = SCRIPT_DIR / "outputs_eval" / "cluster_rag"
DEFAULT_LOG_DIR = SCRIPT_DIR / "logs_eval" / "cluster_rag"

CLUSTER_RAG_RAG_PREAMBLE = (
    "（以下检索为**四阶段**合并结果：1）分群向；2）心理筛查 flags 向；"
    "3）课程影响/研究性群体规律向；4）风险向（含自杀意念时的高优先级条）。\n"
    "同一文献按优先级去重。）\n\n"
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


def _unique_topics(
    items: list[tuple[float, str, dict[str, Any]]],
) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for _, _, m in items:
        t = m.get("topic")
        s = str(t).strip() if t is not None else ""
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def build_retrieval_query_log(
    student: Mapping[str, Any],
    active_flags: list[str],
) -> dict[str, str | None]:
    """与四阶段实际使用的问句一致（与 ``retrieve_psych_knowledge`` 内调用对齐）。"""
    from run_agent import (
        build_cluster_query_text,
        build_course_impact_query_text,
        build_feature_query_text,
        build_risk_query_text,
    )

    cname = str(student.get("cluster_name", "") or "").strip()
    q_cluster = build_cluster_query_text(cname) if cname else None
    q_feature = build_feature_query_text(list(active_flags))
    q_course = build_course_impact_query_text(
        student, list(active_flags), cname
    )
    q_risk = build_risk_query_text(student)
    return {
        "cluster_aware": q_cluster,
        "feature_aware": q_feature,
        "course_impact_aware": q_course,
        "risk_aware": q_risk,
    }


def main() -> int:
    from run_agent import (
        CLUSTER_KNOWLEDGE_PATH,
        OLLAMA_EMBED,
        OLLAMA_LLM,
        RISK_RULES_PATH,
        REPORT_STYLE_PATH,
        build_system_prompt,
        build_user_prompt,
        call_llm_markdown,
        classify_student,
        find_cluster_block,
        format_rag_for_prompt,
        retrieve_psych_knowledge,
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
        description="四阶段分群+课程感知 RAG 与心理支持报告生成",
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

    kb_paths = [Path(CLUSTER_KNOWLEDGE_PATH), Path(RISK_RULES_PATH)]
    if Path(REPORT_STYLE_PATH).is_file():
        kb_paths.append(Path(REPORT_STYLE_PATH))
    if PSYCH_RAG_CSV.is_file():
        kb_paths.append(PSYCH_RAG_CSV)
    knowledge_base_version = _kb_fingerprint(kb_paths)

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
    if Path(REPORT_STYLE_PATH).is_file():
        try:
            so = load_json(REPORT_STYLE_PATH)
            if isinstance(so, dict):
                _style = so
        except (OSError, json.JSONDecodeError):
            _style = None

    active = get_active_flags(cast(Any, student))
    retrieval_query = build_retrieval_query_log(cast(Any, student), active)

    try:
        ranked = retrieve_psych_knowledge(cast(Any, student), active)
    except (RuntimeError, ValueError) as e:
        print(f"四阶段检索失败: {e}", file=sys.stderr)
        return 1

    rag_body = format_rag_for_prompt(ranked)
    rag_text = CLUSTER_RAG_RAG_PREAMBLE + rag_body

    system_prompt = build_system_prompt()
    system_prompt += (
        "\n【本运行：四阶段 RAG】检索已按分群、心理 flags、课程影响研究规律、"
        "风险（含自杀意念）分阶段完成并去重。写作须非诊断化、不编造个体成绩；"
        "边界与求助提醒自然写入「提醒与边界说明」，勿在文末逐条堆叠系统规则句。"
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
    report_path = out_dir / f"{student_id}_cluster_rag.md"
    log_path = log_dir / f"{student_id}_cluster_rag.json"

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
        "embedding_model": OLLAMA_EMBED,
        "prompt_version": PROMPT_VERSION,
        "knowledge_base_version": knowledge_base_version,
        "retrieval_query": retrieval_query,
        "retrieved_doc_count": len(ranked),
        "retrieved_topics": topics,
        "report_file": str(report_path.resolve().as_posix()),
        "timestamp": _iso_timestamp(),
    }
    try:
        save_json(log_path, log_obj, ensure_ascii=False)
    except OSError as e:
        print(f"保存日志失败: {e}", file=sys.stderr)
        return 1

    print("完成。")
    print(f"  学生: {student_id}")
    print(f"  分群: cluster_id={cls_out['cluster_id']}  {cls_out['cluster_name']}")
    print(f"  合并检索条数: {len(ranked)}  去重后 topic 数: {len(topics)}")
    print(f"  报告: {report_path}")
    print(f"  日志: {log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
