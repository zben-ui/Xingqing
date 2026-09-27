# -*- coding: utf-8 -*-
"""
无 RAG 版：本地 KMeans 分群 + 群体画像（cluster_profile_knowledge.json）+
本地 Ollama 生成五段式报告 + 安全审阅。

不调用 Chroma/向量检索，不依赖 LangChain。

用法（项目根目录）:
  python generate_report_no_rag.py  students_eval/student1.json

环境变量（与 run_agent 一致，可选）:
  OLLAMA_LLM, OLLAMA_HOST, LLM_TEMPERATURE,
  MODELS_DIR, KMEANS_MODEL_PATH, SCALER_PATH, FEATURE_ORDER_PATH,
  CLUSTER_KNOWLEDGE_PATH, RISK_RULES_PATH, REPORT_STYLE_PATH
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

SCRIPT_DIR = Path(__file__).resolve().parent

# 与 run_agent 中 REPORT_GENERATION 占位符约定一致
PROMPT_VERSION = "no_rag_v1"
METHOD = "kmeans_local_no_rag"

# 无 RAG：明确告知模型不要期待检索片段
RETRIEVED_KNOWLEDGE_NO_RAG = (
    "（本流程为**无 RAG**模式，未进行知识库向量检索。请仅依据下方【输入信息】中的"
    "学生心理数据、KMeans 分群结果，以及 `cluster_profile_knowledge` 中的"
    "研究背景与当前分群群体画像撰写；不得编造未在输入与画像知识中出现的事实。）"
)

DEFAULT_OUT_DIR = SCRIPT_DIR / "outputs_eval" / "no_rag"
DEFAULT_LOG_DIR = SCRIPT_DIR / "logs_eval" / "no_rag"


def _kb_fingerprint(paths: list[Path]) -> str:
    """多文件内容拼接后的短 sha256，用于日志中的 knowledge_base_version。"""
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
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _prepare_student(data: dict[str, Any]) -> dict[str, Any]:
    """与 run_agent 一致：去掉 name，不进入模型与审阅。"""
    s = {**data}
    s.pop("name", None)
    return s


def main() -> int:
    from run_agent import (  # 延迟导入，避免脚本未在根目录时过早失败
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
        get_psychological_profile,
        load_json,
        load_cluster_knowledge,
        normalize_text,
        save_json,
        save_markdown,
    )

    parser = argparse.ArgumentParser(
        description="无 RAG：分群 + 群体画像 + 本地 LLM 生成心理支持报告",
    )
    parser.add_argument(
        "student_json",
        type=Path,
        help="仅心理版学生 JSON 路径",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"报告输出目录（默认: {DEFAULT_OUT_DIR}）",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=DEFAULT_LOG_DIR,
        help=f"运行日志 JSON 目录（默认: {DEFAULT_LOG_DIR}）",
    )
    args = parser.parse_args()
    spath: Path = args.student_json.resolve()
    out_dir: Path = args.output_dir.resolve()
    log_dir: Path = args.log_dir.resolve()

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

    # 1) 分群
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

    # 2) 群体画像知识（全量 JSON + 匹配子块 + study_context）
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

    # 3) 风险规则
    try:
        risk_raw = load_json(RISK_RULES_PATH)
    except (OSError, json.JSONDecodeError) as e:
        print(f"读取 risk_rules.json 失败: {e}", file=sys.stderr)
        return 1
    if not isinstance(risk_raw, list):
        print("risk_rules.json 期望为 JSON 数组。", file=sys.stderr)
        return 1

    # 4) 可选写作风格
    _style: dict[str, Any] | None = None
    if REPORT_STYLE_PATH.is_file():
        try:
            so = load_json(REPORT_STYLE_PATH)
            if isinstance(so, dict):
                _style = so
        except (OSError, json.JSONDecodeError):
            _style = None

    # 5) 提示词（无 RAG：retrieved 占位为说明性文字）
    system_prompt = build_system_prompt()
    if _style and "writing_constraints" in _style:
        wc = _style.get("writing_constraints") or []
        if wc:
            system_prompt += "\n附加写作约束: " + "；".join(str(x) for x in list(wc)[:8])
    if _style and "tone" in _style and isinstance(_style.get("tone"), dict):
        t = _style.get("tone")
        if isinstance(t, dict) and t.get("overall"):
            system_prompt += f"\n语气参考: {t.get('overall')}"

    try:
        user_prompt = build_user_prompt(
            cast(Any, student),
            study_context,
            cblock,
            cluster_json_text,
            RETRIEVED_KNOWLEDGE_NO_RAG,
        )
        report = call_llm_markdown(user_prompt, system_prompt)
    except RuntimeError as e:
        print(f"生成阶段失败: {e}", file=sys.stderr)
        return 1

    # 6) 安全审阅（与 run_agent 一致：自杀高优先级、边界、风险规则、禁止诊断化替换）
    try:
        report = safety_review(report, student, risk_raw)
        report = normalize_text(report)
    except (OSError, TypeError) as e:
        print(f"后处理失败: {e}", file=sys.stderr)
        return 1

    # 7) 落盘
    ensure_dir(out_dir)
    ensure_dir(log_dir)
    report_path = out_dir / f"{student_id}_no_rag.md"
    log_path = log_dir / f"{student_id}_no_rag.json"

    try:
        save_markdown(report_path, report)
    except OSError as e:
        print(f"保存报告失败: {e}", file=sys.stderr)
        return 1

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
    }
    try:
        save_json(log_path, log_obj, ensure_ascii=False)
    except OSError as e:
        print(f"保存日志失败: {e}", file=sys.stderr)
        return 1

    print("完成。")
    print(f"  学生: {student_id}")
    print(f"  分群: cluster_id={cls_out['cluster_id']}  {cls_out['cluster_name']}")
    print(f"  报告: {report_path}")
    print(f"  日志: {log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
