# 基于心理分群结果驱动的个体化报告生成 Agent

## 1. 项目简介

本项目是**仅心理输入版**的**本地可运行**辅助工具。学生 JSON **不自带分群**；主程序用 ``models/`` 中的 **KMeans + StandardScaler**（与 ``feature_order.json`` 同训练）预测 ``cluster_id`` / ``cluster_name`` 后，再读群体画像、做 Chroma 检索并由本地大语言模型成文。**不依赖**个体 GPA、挂科、成绩轨迹等学业字段。

典型流程包括：

1. **分群预测**：按 ``psychological_profile`` 与 ``feature_order.json`` 构造特征 → 标准化 → KMeans 得到 ``cluster_name``（与 `knowledge/cluster_profile_knowledge.json` 中 `cluster_name_zh` 对应）；
2. 将领域知识表嵌入并写入 Chroma 持久化向量库（``build_vectorstore.py``，首次/更新语料时）；
3. 按「**分群**—心理 flags（激活时）—**风险**（`suicidal_ideation=1` 时强制）」多阶段检索并去重；
4. 经提示词与 ``risk_rules`` 成稿，**``safety_review``** 做边界与禁止用语、自杀高优先级等处理。

首次无模型文件时，可运行 ``python build_example_cluster_models.py`` 生成**仅供联调**的示例 joblib，正式使用请**替换**为你方真实训练导出的 `kmeans_model.joblib` 与 `scaler.joblib`（见 `models/README.md`）。

**定位**：教育与支持信息整理，**不**用于临床诊断、**不**替代专业心理咨询或危机干预。适用于课程作业、研究原型与论文配套演示（demo）。

---

## 2. 项目目录结构

```
项目根目录/
├── README.md                 # 本说明
├── requirements.txt          # Python 依赖
├── run_agent.py              # 主程序：检索 + 提示词 + Ollama 生成 + 安全审阅
├── build_vectorstore.py      # 从 CSV 构建 Chroma 向量库
├── utils.py                  # JSON/路径/去噪文本等工具函数
├── student1.json             # 示例学生（无 cluster，含 psychological_profile 与基本信息）
├── models/                   # 分群：kmeans_model.joblib, scaler.joblib, feature_order.json
├── build_example_cluster_models.py  # 可选：生成示例 KMeans+Scaler（联调用）
├── classify_student.py       # 仅打印分群结果（不跑 LLM）
├── knowledge/                # 静态知识与规则
│   ├── psych_rag_kb.csv            # 心理教育知识库（RAG 语料源）
│   ├── cluster_profile_knowledge.json  # 聚类群体画像与解释辅助知识
│   ├── risk_rules.json            # 报告合规与安全规则（触发条件、必须出现/禁止用语）
│   └── report_style_rules.json   # 写作与结构约束（可选，供主程序强化提示词）
├── chroma_db/                # Chroma 持久化目录（运行 build 后生成，勿手改）
└── outputs/                  # 生成报告输出目录（运行 Agent 后生成）
    └── {student_id}_psych_only_report.md
```

> 若本地尚未执行建库，则无 `chroma_db/`；首次运行前请先完成「构建向量库」步骤。

---

## 3. 环境准备

| 项目 | 说明 |
|------|------|
| 操作系统 | Windows / macOS / Linux 均可；路径以本仓库**项目根目录**为工作目录 |
| Python | **3.10+**（推荐 3.10 或 3.11 稳定线） |
| 硬件 | 建议具备足够内存以本地运行 9B 级模型；向量构建阶段需能调用 Ollama 嵌入接口 |
| 服务 | 本机需安装并启动 **[Ollama](https://ollama.com/)**，并保证可访问其 HTTP 端点（默认 `http://127.0.0.1:11434`；远程可设环境变量，见各脚本内注释） |

---

## 4. Ollama 模型准备

在终端中拉取与运行下列两个模型（名称需与程序内常量或环境变量一致）：

| 用途 | 模型 | 建议命令（示例） |
|------|------|------------------|
| 报告正文生成 | `qwen3.5:9b` | `ollama pull qwen3.5:9b` |
| 知识库与查询向量嵌入 | `qwen3-embedding:0.6b` | `ollama pull qwen3-embedding:0.6b` |

拉取完成后可用 `ollama list` 确认；若你使用本地自建标签名，请通过环境变量 `OLLAMA_LLM` / `OLLAMA_EMBED` 与 `run_agent.py`、`build_vectorstore.py` 中的配置对齐。

---

## 5. 安装依赖

在项目根目录执行：

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate

pip install -r requirements.txt
```

主要依赖包括：`pandas`、`chromadb`、`ollama`、`pydantic`、`numpy`、`scikit-learn`、`joblib`（分群与 ``joblib.load``）；`python-dotenv` 与 `tqdm` 为可选增强。

---

## 6. 如何构建向量库

1. 确认 `knowledge/psych_rag_kb.csv` 列完整（`id, topic, source_type, applicable_clusters, applicable_flags, risk_level, content, source` 等，以脚本内校验为准）。
2. 确认 Ollama 已运行且已拉取 **qwen3-embedding:0.6b**。
3. 在项目根目录执行：

```bash
python build_vectorstore.py
```

成功后将：

- 在 `chroma_db/` 下持久化数据库；
- 使用集合名 **`psych_kb`**（若已存在同名集则按脚本逻辑重建）。

可通过环境变量调整持久化目录、集合名、批大小等（见 `build_vectorstore.py` 顶部注释）。

---

## 7. 如何运行 Agent

1. 完成「向量库构建」与「Ollama 中 **qwen3.5:9b** 可用」。
2. 准备或修改学生档案：默认读取项目根下 **`student1.json`**（亦可通过环境变量 `STUDENT_JSON` 指定绝对路径）。
3. 确保 `knowledge/` 下 **`cluster_profile_knowledge.json`**、**`risk_rules.json`** 等文件可解析且与分群/规则设计一致（论文复现时请与实验设定同步）。
4. 在项目根目录执行：

```bash
python run_agent.py
```

程序将：加载学生与知识 → 多阶段 Chroma 检索（仅基于分群、心理 flags、自杀意念）→ 调用 `qwen3.5:9b` 生成 Markdown → 执行 `safety_review` 等后处理 → 将终稿写入 **`outputs/{student_id}_psych_only_report.md`**。

---

## 8. 输入输出说明

### 输入（主要）

| 类型 | 路径/说明 |
|------|-----------|
| 学生档案 | 默认 `student1.json`：须含匿名 `student_id`、`gender`、`age`、`grade` 等（可按需增删）及 **`psychological_profile`**；**不要**包含真实姓名（不要 `name` 字段）；**不要**在档案中写 `cluster` / `cluster_name`（由程序预测并写回后参与检索与成文）；报告仅通过「学生代号：{student_id}」显示标识 |
| 分群与群体知识 | `knowledge/cluster_profile_knowledge.json` |
| 规则库 | `knowledge/risk_rules.json`（JSON 数组，含触发与必须/禁止表述） |
| 可选风格约束 | `knowledge/report_style_rules.json` |
| 检索依赖 | 已构建的 `chroma_db/` 中 `psych_kb` 集合，语料源为 `knowledge/psych_rag_kb.csv` |

### 输出

| 类型 | 说明 |
|------|------|
| 主输出 | `outputs/{student_id}_psych_only_report.md`：单篇 Markdown，固定五个 `##` 节：心理画像、分群解释、**相关风险与支持需求**、个体化建议、提醒与边界（见 `run_agent.py` 模板） |
| 控制台 | 简要提示学生 ID、输出路径、检索条数等 |

环境变量可覆盖 Chroma 路径、集合名、模型名、输出目录等，详见 `run_agent.py` 与 `build_vectorstore.py` 文件头部常量区。

---

## 9. 注意事项

1. **先后次序**：需先 `build_vectorstore.py` 成功，再 `run_agent.py`；否则 Chroma 无数据或集合缺失将导致检索失败。
2. **Ollama 占用**：9B 模型对显存/内存有要求，低配机器可适当减小并发或换用经评估的小模型，并同步调整环境变量与提示词中的期望。
3. **数据与伦理**：仅使用脱敏、合规的示例数据作 demo；真实施测数据须符合伦理审批与校规。
4. **分群解释**：聚类结果属于**研究性/统计性分组描述**，在报告中**不得**写成疾病诊断或固定人格标签；写作约束已写入提示词与安全审阅逻辑，但仍建议人工抽查。
5. **高风险内容**：当档案中提示自伤/自杀相关信号时，系统会尝试在生成与安全审阅阶段强化**专业求助**表述；**不能**替代现实世界的危机评估与现场处置。
6. **版本与复现**：论文投稿或附录中建议写明 Python、依赖版本、`ollama` 与具体模型 `tag`、以及随机种子/温度（见 `run_agent.py` 中 `LLM_TEMPERATURE` 等）。

---

## 10. 本项目边界声明

本仓库中的脚本与生成内容**仅用于辅助性、教育性、研究演示性**的文本整理与支持性建议组织，**不构成**医学或心理学上的临床诊断、治疗建议或危机处置方案。

- 筛查、聚类与自动报告**不能替代**有资质专业人员的面询、评估、诊断或干预。  
- **不得**将本工具输出作为唯一依据作出重大生活或医疗决定。  
- 若存在明显心理痛苦、自伤/自杀念头或其他紧急情况，请**立即**联系学校心理中心、医疗应急（如 120）或当地危机干预与心理援助资源。

> **本报告/工具仅用于辅助支持，不替代专业诊断、心理咨询或医学评估；在论文与演示材料中建议原文或等价引用上述原则。**

---

## 引用与复现（论文配套建议）

在方法或附录中可简要说明：本地 Ollama 部署、嵌入模型与生成模型 ID、Chroma 持久化路径与集合名、知识库文件版本、以及 `safety_review` 与 `risk_rules.json` 的合规后处理。欢迎在本 README 版本信息的基础上补充你论文中的**数据集编号与伦理批件**说明。

*README 与仓库代码版本以你本地 Git 或归档日期为准。*
