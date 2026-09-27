# -*- coding: utf-8 -*-
"""
联调/演示用：按 ``models/feature_order.json`` 的维数，用随机数据拟合
``StandardScaler`` + ``KMeans(n_clusters=2)`` 并保存为 joblib。

正式环境请使用真实训练集拟合的模型**替换**同路径文件，否则分群无实际意义。

用法（项目根）:
  python build_example_cluster_models.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
FP = SCRIPT_DIR / "models" / "feature_order.json"
OUT_S = SCRIPT_DIR / "models" / "scaler.joblib"
OUT_K = SCRIPT_DIR / "models" / "kmeans_model.joblib"


def main() -> int:
    if not FP.is_file():
        print(f"缺少: {FP}", file=sys.stderr)
        return 1
    with FP.open("r", encoding="utf-8", newline="") as f:
        order = json.load(f)
    n = len(order) if isinstance(order, list) else 0
    if n < 1:
        print("feature_order 为空。", file=sys.stderr)
        return 1
    rng = np.random.default_rng(0)
    X = rng.random((100, n))
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    km = KMeans(n_clusters=2, random_state=0, n_init=10)
    km.fit(Xs)
    OUT_S.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, OUT_S)
    joblib.dump(km, OUT_K)
    print(f"已写入（示例，请替换为真实训练模型）:\n  {OUT_S}\n  {OUT_K}\n  特征维数: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
