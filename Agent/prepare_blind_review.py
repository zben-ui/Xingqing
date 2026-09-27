# -*- coding: utf-8 -*-
"""
将 ``outputs_eval/`` 下三种子目录中的报告复制到 ``blind_reports/``，按
``case{序号}_{A|B|C}.md`` 命名，**不暴露方法名**；同一学生三份与 A/B/C 的对应关系
**每次运行随机**（可设环境变量 ``BLIND_REVIEW_SEED`` 以固定顺序便于复现）。

同时生成 ``evaluation/blind_mapping.csv``，记录 ``case`` 与 ``student_id``、
真实 ``method``、原始相对路径，仅供组织者保管，**勿随盲评材料分发给评阅人**。

用法（项目根目录）:
  python prepare_blind_review.py
  set BLIND_REVIEW_SEED=42  # Windows: set BLIND_REVIEW_SEED=42
"""

from __future__ import annotations

import csv
import os
import random
import re
import shutil
import sys
from pathlib import Path

# 项目根 = 本脚本所在目录
ROOT = Path(__file__).resolve().parent
OUTPUTS_EVAL = ROOT / "outputs_eval"
BLIND_DIR = ROOT / "blind_reports"
EVALUATION_DIR = ROOT / "evaluation"
MAPPING_CSV = EVALUATION_DIR / "blind_mapping.csv"

# 子目录与 ``method`` 字段取值（与生成脚本、测评脚本一致）
SUBDIRS: tuple[str, str, str] = ("no_rag", "generic_rag", "cluster_rag")


def _parse_filename(name: str) -> tuple[str | None, str | None]:
    """自 ``{student_id}_no_rag.md`` 等解析 student_id 与 method。"""
    m = re.match(
        r"^(.+?)_(no_rag|generic_rag|cluster_rag)\.md$",
        name,
        re.IGNORECASE,
    )
    if not m:
        return None, None
    return m.group(1), m.group(2).lower()


def collect_students_with_three_reports() -> dict[str, dict[str, Path]]:
    """
    扫描 ``outputs_eval/*/{pattern}.md``，按 student_id 聚合。

    仅当该学生在三个子目录中**各有一份**规范命名文件时，才纳入（避免盲评缺页）。
    """
    by_sid: dict[str, dict[str, Path]] = {}
    for sub in SUBDIRS:
        d = OUTPUTS_EVAL / sub
        if not d.is_dir():
            continue
        for p in d.glob("*.md"):
            sid, meth = _parse_filename(p.name)
            if not sid or not meth or meth not in SUBDIRS:
                print(f"跳过无法解析或方法名不认识的文件: {d.name}/{p.name}", file=sys.stderr)
                continue
            by_sid.setdefault(sid, {})[meth] = p
    out: dict[str, dict[str, Path]] = {}
    for sid, mmap in by_sid.items():
        if all(k in mmap for k in SUBDIRS):
            out[sid] = {k: mmap[k] for k in SUBDIRS}
        else:
            missing = [k for k in SUBDIRS if k not in mmap]
            print(
                f"学生 {sid!r} 缺报告（少: {missing}），不参与盲评复制。",
                file=sys.stderr,
            )
    return out


def _init_rng() -> None:
    """可重复：若设置 ``BLIND_REVIEW_SEED``（整数字符串）则固定随机分配。"""
    s = os.environ.get("BLIND_REVIEW_SEED", "").strip()
    if s.lstrip("-").isdigit():
        random.seed(int(s))
    # 否则使用系统熵，每次运行 A/B/C 与方法的对应不同


def _clear_blind_case_files() -> None:
    """避免上次运行遗留的 case 文件与本次不一致，删除 ``blind_reports`` 下 ``case*.md``。"""
    if not BLIND_DIR.is_dir():
        return
    for p in BLIND_DIR.glob("case*.md"):
        try:
            p.unlink()
        except OSError as e:
            print(f"无法删除旧文件 {p}: {e}", file=sys.stderr)


def main() -> int:
    by_student = collect_students_with_three_reports()
    if not by_student:
        print(
            "没有可供盲评的完整三件套报告（需 outputs_eval 下三子目录各有同名 student 的 md）。",
            file=sys.stderr,
        )
        EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
        with MAPPING_CSV.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["blind_id", "student_id", "method", "original_file"],
            )
            w.writeheader()
        print(f"已写出空表头: {MAPPING_CSV}")
        return 0

    _init_rng()
    _clear_blind_case_files()
    BLIND_DIR.mkdir(parents=True, exist_ok=True)
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

    # 稳定序号：按 student_id 排序后 case01, case02, …
    sorted_sids = sorted(by_student.keys(), key=str)
    labels = ("A", "B", "C")
    rows: list[dict[str, str]] = []
    n_copied = 0

    for case_idx, sid in enumerate(sorted_sids, start=1):
        meth_paths = by_student[sid]
        # 三份 (method, path) 打乱后，按顺序命名为 A、B、C
        triple: list[tuple[str, Path]] = [
            ("no_rag", meth_paths["no_rag"]),
            ("generic_rag", meth_paths["generic_rag"]),
            ("cluster_rag", meth_paths["cluster_rag"]),
        ]
        random.shuffle(triple)

        case_base = f"case{case_idx:02d}"
        for (method, src_path), lab in zip(triple, labels):
            blind_id = f"{case_base}_{lab}"
            out_name = f"{blind_id}.md"
            dest = BLIND_DIR / out_name
            try:
                shutil.copy2(src_path, dest)
            except OSError as e:
                print(f"复制失败 {src_path} -> {dest}: {e}", file=sys.stderr)
                return 1
            n_copied += 1
            try:
                orig_rel = src_path.resolve().relative_to(ROOT)
            except ValueError:
                orig_rel = src_path
            rows.append(
                {
                    "blind_id": blind_id,
                    "student_id": sid,
                    "method": method,
                    "original_file": str(orig_rel).replace(os.sep, "/"),
                }
            )

    with MAPPING_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["blind_id", "student_id", "method", "original_file"],
        )
        w.writeheader()
        w.writerows(rows)

    print(f"映射表: {MAPPING_CSV.resolve()}")
    print(
        f"共准备 {n_copied} 份盲评报告"
        f"（{len(sorted_sids)} 个学生 × 3 份），输出目录: {BLIND_DIR.resolve()}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
