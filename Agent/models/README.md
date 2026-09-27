# 分群模型文件（与训练时保持一致）

本目录应包含以下文件，供 `run_agent.classify_student()` 加载：

| 文件 | 说明 |
|------|------|
| `kmeans_model.joblib` | 训练好的 `sklearn.cluster.KMeans`（或其它兼容 `predict` 的聚类器） |
| `scaler.joblib` | 训练时使用的 `sklearn.preprocessing.StandardScaler`（与特征顺序、维度一致） |
| `feature_order.json` | 一维特征名列表，顺序必须与训练时输入矩阵列顺序一致，且能从 `student.psychological_profile` 中按名取值（缺省为 0） |

`feature_order.json` 可为根级字符串数组，例如：

```json
["upi_flag", "scl90_depression_flag", ...]
```

或对象形式，例如 `{"feature_order": ["upi_flag", ...]}`。

聚类标签 `0` / `1` 在 `run_agent.py` 的 `CLUSTER_ID_TO_NAME` 中映射为中文分群名，与 `knowledge/cluster_profile_knowledge.json` 的 `cluster_name_zh` 对齐；若你的模型输出其它编号，请同步修改 `CLUSTER_ID_TO_NAME`。

环境变量可覆盖路径：`MODELS_DIR`、`KMEANS_MODEL_PATH`、`SCALER_PATH`、`FEATURE_ORDER_PATH`。
