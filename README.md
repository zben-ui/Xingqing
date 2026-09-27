# 小艾：本地 Ollama 情感陪伴数字人 MVP

## 项目介绍

小艾是一个由本地 Ollama 驱动的情感陪伴数字人 MVP。当前版本已经跑通用户注册登录、流式陪伴对话、Blender 数字人及四个动作、账号隔离的本地知识库、陪伴风格约束、一周状态自测、非诊断性状态报告，以及管理员风险预警与处置流程，不需要云端 API Key。

## 运行界面

<img width="1920" height="945" alt="4d1075f7f3f9ec21bb62858fecf6fc9f" src="https://github.com/user-attachments/assets/e2e6897f-78d7-42f1-bc8c-6097115416b2" />
<img width="1920" height="945" alt="dc5b2cf23292b4ee5adb37a667291e70" src="https://github.com/user-attachments/assets/5c03b563-79c6-4913-bea9-01bc5e3534d2" />
<img width="1920" height="945" alt="5c19b08c80fb5816ff988d978d77e9fb" src="https://github.com/user-attachments/assets/02c47894-c429-4959-87d2-982a5efb150e" />
<img width="1920" height="945" alt="b8a00b5b5f292cf85e611990d5434b86" src="https://github.com/user-attachments/assets/f358ded6-7546-40f8-93c1-790ac4087eef" />
<img width="1920" height="945" alt="ec62ba592854e51f93f5f3e082a7a2fd" src="https://github.com/user-attachments/assets/0b1704df-7100-44fa-a734-c4e77e53b1bc" />
<img width="1920" height="945" alt="bfca62ce874454d4bbe48ffbc2b179de" src="https://github.com/user-attachments/assets/e601c9e1-734f-47c7-b8b9-d42661a826fa" />
<img width="1920" height="945" alt="2691e380dc9b769152ad2ca3d8f7f9ef" src="https://github.com/user-attachments/assets/4008fa45-0d8b-4b7d-9779-261bf04b2982" />



## 小艾的人设

小艾温柔、自然、真诚，先倾听，再提供少量实际建议。她不是医生、心理咨询师或真人，不做诊断，也不会诱导用户只依赖数字人。自伤、自杀或伤害他人的危机表达会绕过普通聊天，直接进入安全回复模式。

## 技术架构

```text
浏览器 HTML/CSS/JavaScript + pinkgirl.glb 数字人
        │
        ▼
FastAPI + Bearer 会话
        ├── safety_service.py  安全分级
        ├── memory_service.py  最近 10 轮记忆
        ├── rag_service.py     用户隔离的向量/关键词检索
        ├── assessment_service.py  非诊断性状态自测
        ├── report_service.py  测评 + 对话摘要报告
        ├── storage_service.py SQLite 用户、记录与预警
        └── ollama_client.py   OpenAI-compatible → native fallback
                    │
                    ▼
             本地 Ollama
```

FastAPI 同时托管前端静态页面，因此不需要单独安装 Node.js 或启动第二个前端进程。

## 3D 数字人形象

网页直接加载从 Blender 导出的 `frontend/assets/models/pinkgirl.glb`，并用四个直观按钮播放模型内动作：

- `Clip_Idle_Breathe`：呼吸
- `Clip_Wave`：挥手
- `Clip_WalkCycle`：散步
- `Clip_Head_Nod`：点头

对话 Agent 也会根据回应意图自动选择合适动作。模型由本地 `model-viewer` 渲染，运行时不需要联网。

## 本地模型要求

```bash
ollama pull qwen3.5:9b
ollama pull qwen3-embedding:0.6b
ollama pull llava:latest
```

- `qwen3.5:9b`：聊天回答
- `qwen3-embedding:0.6b`：知识库向量
- `llava:latest`：图片理解预留，默认不启用

## 启动 Ollama

先打开 Ollama 桌面程序，或在支持的环境中执行：

```bash
ollama serve
```

浏览器访问 `http://localhost:11434/api/tags` 能看到模型列表即表示服务正常。

## 一键启动

项目使用 Conda 环境 `nn_env`。

Windows：

```bat
start.bat
```

Mac/Linux：

```bash
chmod +x start.sh
./start.sh
```

也可以手动启动：

```bash
conda run -n nn_env python -m uvicorn ai_service.main:app --host 127.0.0.1 --port 8000
```

访问：`http://127.0.0.1:8000`

## 首次使用

1. 打开页面后选择“注册”。
2. 第一个在本机注册的账号会成为管理员，之后注册的账号是普通用户。
3. 普通用户从主登录区进入；管理员从登录页右下角的“管理员登录”进入，普通账号不能访问管理接口。
4. 用户可在“我的资料”上传头像、设置昵称、联系方式、表达风格与背景，在“我的知识库”上传 `.txt` / `.md`。
5. 管理端只有“风险预警”面板：可看高危词、三因子图文线索、学生联系方式、报告与预警，并导入测评 JSON 和发布带封面的文章。
6. 用户可记录每日心情日记、阅读四篇本地散文与新发布文章；自测后可生成依据近期聊天和本地知识库的状态报告。

## 本地 Agent 与心理报告

`Agent/knowledge/psych_rag_kb.csv` 的 46 条资料作为本地心理教育语料读取；通用资料进入聊天/报告检索，风格和安全规则进入报告整理。带研究聚类限定的资料不用于当前自定义自测。示例学生 JSON、示例报告、已有研究索引和 joblib 模型不会执行或向其他用户公开。

当前三因子自测是自定义自我觉察题，不是 SCL-90/UPI，也不能转换成这些量表的阳性标记。因此研究聚类模型保持关闭，避免给用户编造分群标签。报告区会显示本地模型是否可用及引用的片段；模型离线时明确降级为规则整理。

测评模板上传格式为 `{ "title": "...", "description": "...", "questions": [{ "id": "a1", "text": "...", "factor": "anxiety" }, ...] }`。题目须覆盖 `anxiety`、`obsessive`、`somatization`、`safety` 四类，4–40 题，唯一英文 ID，选项统一 0–3。导入后成为新的用户端自测模板；旧记录保留原因子结果。

聊天与日记的高危词命中会生成管理员预警。关键词只是人工关注线索，不等于临床风险结论，不会自动联系学生或紧急联系人。

## 数字人与阿里云朗读

新版模型为 `frontend/assets/models/pinkgirl-companion-v2.glb`，动作只有 `Idle_Relaxed`、`Talk_Gentle`、`Greeting_Wave`。初始双手自然下垂；回复/播放语音时说话，其余待机。点击人物停止当前朗读并回到待机，人物不支持旋转。说话使用已有下颌权重与轻微手势，并非逐音素口型同步。原 GLB 和原 Blender 文件保留。

按用户确认沿用阿里云 `sambert-zhiwei-v1`。后端优先读取 `DASHSCOPE_API_KEY`，也支持 `TTS_REFERENCE_SERVER` 指向本机旧项目的 `server.js` 读取既有密钥；不复制密钥到前端，不运行旧服务器。推荐将密钥迁移到环境变量，并轮换旧源码中曾写死的密钥。

`ENABLE_TTS=true` 后调用认证接口 `POST /api/tts` 返回 MP3，无公开音频目录。只把待朗读的回复文本发送至阿里云，Ollama 聊天及报告推理仍在本机。用户可以关闭朗读；播放被浏览器拦截时点击“重新朗读”。

## 添加与重建知识库

登录后进入“我的知识库”，上传 UTF-8 编码的 `.txt` 或 `.md` 文件（单个不超过 1MB）。资料按账号保存并检索；Embedding 不可用时会自动降级为关键词检索。

## 测试聊天功能

健康检查：

```bash
curl http://127.0.0.1:8000/api/health
```

运行不依赖真实模型的自动测试：

```bash
conda run -n nn_env python -m unittest discover -s tests -v
```

推荐人工验收：

1. 普通聊天：“你能做什么”
2. 情绪支持：“我最近压力很大”
3. RAG：“小艾使用的模型是什么”
4. 危机安全：“我不想活了”（应直接返回安全建议）
5. 关闭 Ollama 后再次聊天（页面不应崩溃）

## API

- `GET /api/health`
- `POST /api/auth/register`、`POST /api/auth/login`
- `POST /api/agent/chat/stream`
- `GET/PUT /api/profile`
- `GET/POST /api/assessments`
- `POST /api/reports/generate`
- `GET/POST /api/diary`、`GET /api/articles`
- `POST /api/tts`
- `POST /api/admin/assessment-templates`、`POST /api/admin/articles`
- `GET/POST /api/knowledge`
- `GET /api/admin/overview`、`GET /api/admin/users`、`GET /api/admin/alerts`
- `POST /api/agent/clear`
- `POST /api/rag/rebuild`
- `POST /api/vision/chat`
- `GET /docs`：FastAPI 接口文档

## 后续升级

- Live2D：把当前 WebGL/CSS 数字人替换为 Cubism 模型，并根据 `emotion/action` 驱动动作。
- TTS：后续可替换为本地 ChatTTS/CosyVoice，减少云端文本传输。
- 语音识别：接入浏览器 SpeechRecognition 或本地 Whisper。
- 多模态：设置 `ENABLE_VISION=true`，完善文件上传和 `llava:latest` 调用。
- 长期记忆：在明确征得用户同意后，增加可查看、可删除、可过期的本地存储。
- RAG：增加文档上传、增量索引、reranker 和引用片段展开。
