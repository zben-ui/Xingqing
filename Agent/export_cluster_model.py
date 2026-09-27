# -*- coding: utf-8 -*-
"""
将已训练好的 KMeans 分群模型与 StandardScaler 导出到本项目的 models/ 目录。

典型用法 A（在 notebook / 本脚本中已有变量）::

    from export_cluster_model import export_cluster_artifacts

    # 假设你已有训练好的 kmeans_model, scaler
    export_cluster_artifacts(kmeans_model, scaler)

典型用法 B（从任意路径的 joblib 读入，再按统一文件名写出）::

    python export_cluster_model.py  path/to/your_kmeans.joblib  path/to/your_scaler.joblib

将写入（相对项目根）::

    models/kmeans_model.joblib
    models/scaler.joblib
    models/feature_order.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib

# 与主程序、心理档案 psychological_profile 字段一致；设计矩阵列顺序须与训练时一致
FEATURE_NAMES: list[str] = [
    "upi_flag",
    "scl90_depression_flag",
    "scl90_anxiety_flag",
    "scl90_interpersonal_sensitivity_flag",
    "scl90_obsessive_flag",
    "scl90_psychoticism_flag",
    "scl90_paranoid_flag",
    "scl90_hostility_flag",
    "suicidal_ideation",
]

# 默认与 run_agent 中 models 目录一致；可通过参数覆盖
_DEFAULT_MODELS_DIR = Path(__file__).resolve().parent / "models"


def export_cluster_artifacts(
    kmeans_model: Any,
    scaler: Any,
    models_dir: str | Path | None = None,
) -> None:
    """
    将 KMeans 与 Scaler 用 joblib 落盘，并写出特征顺序 JSON。

    :param kmeans_model: 已 fit 的聚类器（如 ``sklearn.cluster.KMeans``）
    :param scaler: 已 fit 的标准化器（如 ``sklearn.preprocessing.StandardScaler``）
    :param models_dir: 输出目录，默认本脚本所在目录下的 ``models/``；可传 str 或 Path
    """
    out_dir = (
        Path(models_dir).resolve()
        if models_dir is not None
        else _DEFAULT_MODELS_DIR.resolve()
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    kpath = out_dir / "kmeans_model.joblib"
    spath = out_dir / "scaler.joblib"
    fpath = out_dir / "feature_order.json"

    joblib.dump(kmeans_model, kpath)
    joblib.dump(scaler, spath)

    with fpath.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(FEATURE_NAMES, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print("已导出：")
    print(f"  {kpath}")
    print(f"  {spath}")
    print(f"  {fpath}")
    print(f"  特征维数: {len(FEATURE_NAMES)}")


def _main_cli() -> int:
    """从命令行传入两个 joblib 文件路径时，读入后按统一文件名写到 models/。"""
    parser = argparse.ArgumentParser(
        description="从两个 joblib 读入 kmeans 与 scaler，并写出到 models/（含 feature_order.json）"
    )
    parser.add_argument("kmeans_joblib", nargs="?", help="已保存的 KMeans 的 .joblib 路径")
    parser.add_argument("scaler_joblib", nargs="?", help="已保存的 StandardScaler 的 .joblib 路径")
    parser.add_argument(
        "-o",
        "--output-dir",
        default=None,
        help="输出目录，默认为脚本旁 models/",
    )
    args = parser.parse_args()

    if not args.kmeans_joblib or not args.scaler_joblib:
        print(
            "用法:\n"
            "  1) 在代码中: from export_cluster_model import export_cluster_artifacts; "
            "export_cluster_artifacts(kmeans_model, scaler)\n"
            "  2) 或命令行: python export_cluster_model.py <kmeans.joblib> <scaler.joblib> [-o 输出目录]\n",
            file=sys.stderr,
        )
        return 1

    k_path = Path(args.kmeans_joblib)
    s_path = Path(args.scaler_joblib)
    if not k_path.is_file():
        print(f"未找到 kmeans 文件: {k_path}", file=sys.stderr)
        return 1
    if not s_path.is_file():
        print(f"未找到 scaler 文件: {s_path}", file=sys.stderr)
        return 1

    try:
        km = joblib.load(k_path)
        sc = joblib.load(s_path)
    except Exception as e:  # noqa: BLE001
        print(f"加载 joblib 失败: {e}", file=sys.stderr)
        return 1

    out: Path | None = Path(args.output_dir).resolve() if args.output_dir else None
    export_cluster_artifacts(km, sc, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main_cli())
