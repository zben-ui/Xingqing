#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

if ! command -v conda >/dev/null 2>&1; then
  echo "[错误] 未找到 conda，请先安装或初始化 Conda。"
  exit 1
fi

if ! conda run -n nn_env python --version >/dev/null 2>&1; then
  echo "[错误] 未找到 Conda 环境 nn_env。"
  exit 1
fi

[ -f .env ] || cp .env.example .env

echo "[小艾] 检查本地 Ollama..."
if curl -fsS --max-time 5 http://localhost:11434/api/tags >/dev/null 2>&1; then
  echo "[正常] Ollama 已启动。请确认已有 qwen3.5:9b、qwen3-embedding:0.6b、llava:latest。"
else
  echo "[提示] Ollama 当前未连接，网页仍会启动；启动 Ollama 后刷新即可。"
fi

if ! conda run -n nn_env python -c "import fastapi,uvicorn,httpx,dotenv,pydantic; from websockets.asyncio.client import connect" >/dev/null 2>&1; then
  conda run -n nn_env python -m pip install -r requirements.txt
fi

echo "[小艾] 访问 http://127.0.0.1:8000"
conda run -n nn_env python -m uvicorn ai_service.main:app --host 127.0.0.1 --port 8000
