# 索引用文本文档 ↔ 源 JSON/CSV 对应表

本表说明 `knowledge/` 下各「索引用 `*.txt`」与**权威数据源**（程序实际加载的文件）的逐层对应关系。  
**运行时**：`run_agent.py` / `build_vectorstore.py` 只读 **JSON/CSV**；`txt` 为人工/模型索引的副本，**不**自动与源文件同步，修改源文件后需自行更新对应 `txt` 或重新整理。

---

## 1. 总览：一图对应

| 索引用/合并 txt | 主要对应源文件 | 在源文件中的位置（摘要） |
|-----------------|----------------|--------------------------|
| `索引用_01_研究背景与分群画像.txt` | `cluster_profile_knowledge.json` | `study_context` + `clusters` 全数组 |
| `索引用_02_RAG心理教育知识_kb1-32.txt` | `psych_rag_kb.csv` | 行 `id=1`～`id=32`（等同 CSV 首列 id） |
| `索引用_03_RAG课程类别支持_kb33-46.txt` | `psych_rag_kb.csv` | 行 `id=33`～`id=46` |
| `索引用_04_安全与合规规则.txt` | `risk_rules.json` | JSON 数组，每项一条规则（见 §4） |
| `索引用_05_写作风格参考.txt` | `report_style_rules.json` | 根对象下 `tone` / `structure` / `writing_constraints` 等（见 §5） |
| `心理支持报告Agent_知识库全文_索引用.txt` | 上述**全部** | 全文 = 分卷 01～05 的线性拼接（同目录大卷） |
| `索引用_00_说明与文件目录.txt` | （无源文件） | 纯说明，仅描述分卷与路径 |

---

## 2. `cluster_profile_knowledge.json` ↔ `索引用_01_…`

| txt 中章节/标签 | JSON 路径 | 说明 |
|-----------------|-----------|------|
| § 一、1.1 研究对象与分群 | `study_context.population` / `sample_size` / `clustering_basis` / `optimal_clustering` | 分群技术说明 |
| § 1.2 课程与成绩研究结论 | `study_context.course_outcome_research_summary` | 若 JSON 中键名是数组 `course_outcome_research_summary`，与 txt 中三条一一对应 |
| § 1.3 元说明 | `study_context.notes` | 与数组元素顺序一致 |
| § 二、分群 0 | `clusters[0]` 且 `cluster_id === 0` 或 `cluster_name_zh === "低心理负荷相对平稳型"` | 子键：`core_summary` / `psychological_characteristics` / `academic_characteristics` / `course_research_interpretation` / `interpretation_guidance` / `report_tone` |
| § 三、分群 1 | `clusters[1]` 且 `cluster_id === 1` 或 `cluster_name_zh === "高心理负荷型"` | 同上 |

> **注意**：若你本地 JSON 的键名与当前 repo 稍异（如旧版无 `course_outcome_research_summary`），以磁盘上的 `cluster_profile_knowledge.json` 为准，并同步改 `索引用_01` 的整理文。

---

## 3. `psych_rag_kb.csv` ↔ `索引用_02_…` / `索引用_03_…`

- CSV **表头**固定为：`id, topic, source_type, applicable_clusters, applicable_flags, risk_level, content, source`
- 索引用中每条以 **`[KB_ID:N]`** 标出，**N 即 CSV 第一列 `id`**

| 索引用文件 | `id`（CSV 首列） | `source_type` 在 txt 中的主要分布 |
|------------|------------------|-----------------------------------|
| `索引用_02_RAG心理教育知识_kb1-32.txt` | 1–32 | `concept_explanation` / `support_recommendation` / `academic_support` / `help_seeking_guidance` / `boundary_statement` 等（**不含** 纯课程专卷） |
| `索引用_03_RAG课程类别支持_kb33-46.txt` | 33–46 | 对应 CSV 中 `source_type = course_category_support`（及 `source = course_insight_2024` 等列） |

**向量库**：`build_vectorstore.py` 读**同一 CSV** 写入 `chroma_db/`，`metadata` 中 `id` 与上表一致。索引用 `txt` 为同一知识的可读副本。

---

## 4. `risk_rules.json` ↔ `索引用_04_…`

- 源结构：**JSON 数组** `[{ "rule_name", "priority", "trigger", "must_include", "must_not_include", "style_constraints" }, …]`
- 索引用中每条为 **`[规则|rule_name|…]`** 块

| 索引用中的 `rule_name` | JSON 中 | 说明 |
|------------------------|---------|------|
| `must_add_boundary_statement` | 数组中 `rule_name` 同名字段 | `trigger.always: true` |
| `suicidal_ideation_high_priority_support` | 同上 | `trigger.suicidal_ideation: 1` |
| `high_burden_cluster_requires_supportive_tone` | 同上 | `trigger.cluster_name: 高心理负荷型` |
| `low_burden_cluster_no_over_pathologizing` | 同上 | `trigger.cluster_name: 低心理负荷相对平稳型` |
| `no_clinical_diagnosis_language` | 同上 | `trigger.always: true` |
| `cluster_is_not_fixed_label` | 同上 | `trigger.always: true` |

`run_agent.safety_review()` 直接加载 `risk_rules.json`；`索引用_04` 为同一逻辑的人类可读版。

---

## 5. `report_style_rules.json` ↔ `索引用_05_…`

| 索引用段落 | JSON 根键 | 说明 |
|------------|------------|------|
| 整体语气 / 禁用风格 | `tone.overall` / `tone.forbidden_styles` | 数组在 txt 中拆成行 |
| 原五段式 structure（可忽略） | `structure` | 仅心理版已改为提示词内五段；索引用 05 有注释 |
| 写作约束 | `writing_constraints` | 列表 → txt 中 `writing_constraints:` 下逐条 |
| 推荐开头用语 | `recommended_phrases` | 一句话串在 txt 中，分号分隔 |

`run_agent` 若存在 `report_style_rules.json` 会读入并拼到 system 侧约束（以代码为准）；`索引用_05` 与其对齐。

---

## 6. 合并全卷 `心理支持报告Agent_知识库全文_索引用.txt`

- **内容** = `索引用_01` + `索引用_02` + `索引用_03` + `索引用_04` + `索引用_05`（顺序与分卷号一致，中间带一级标题如「## 四、RAG…」）
- **对应源**：即 **§2 + §3 + §4 + §5** 所涉全部文件，无额外数据源。

---

## 7. 维护建议

1. **改知识**：先改 `cluster_profile_knowledge.json` / `psych_rag_kb.csv` / `risk_rules.json` / `report_style_rules.json`，再跑 `build_vectorstore.py`（若动 CSV），**然后**用同一内容更新或重生成各 `索引用_*.txt` 与「全文」`txt`。
2. **查一条 RAG 来自哪行**：在 CSV 里用 `id` 列搜 `[KB_ID:N]` 中的 N。
3. **分群文字**：在 JSON 的 `clusters[]` 里用 `cluster_name_zh` 或 `cluster_id` 定位，与 `索引用_01` 中「分群 0/1」节对应。

---

*对应表与项目 `knowledge/` 目录中文件同步维护；源文件为唯一事实源。*
