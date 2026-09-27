@echo off
setlocal EnableExtensions
title Xiaoai
cd /d "%~dp0"

echo [Xiaoai] starting...
echo [Xiaoai] working dir: %CD%
echo.

set "CONDA_ROOT="
if exist "D:\Program Files\anaconda3\Scripts\activate.bat" set "CONDA_ROOT=D:\Program Files\anaconda3"
if exist "%USERPROFILE%\anaconda3\Scripts\activate.bat" set "CONDA_ROOT=%USERPROFILE%\anaconda3"
if exist "%USERPROFILE%\miniconda3\Scripts\activate.bat" set "CONDA_ROOT=%USERPROFILE%\miniconda3"
if exist "%USERPROFILE%\Miniconda3\Scripts\activate.bat" set "CONDA_ROOT=%USERPROFILE%\Miniconda3"
if exist "C:\ProgramData\anaconda3\Scripts\activate.bat" set "CONDA_ROOT=C:\ProgramData\anaconda3"
if exist "C:\ProgramData\miniconda3\Scripts\activate.bat" set "CONDA_ROOT=C:\ProgramData\miniconda3"

if not defined CONDA_ROOT (
  echo [ERROR] Conda not found. Install Anaconda/Miniconda, or edit CONDA_ROOT in start.bat.
  echo.
  pause
  exit /b 1
)

echo [Xiaoai] conda: %CONDA_ROOT%
call "%CONDA_ROOT%\Scripts\activate.bat" nn_env
if errorlevel 1 (
  echo [ERROR] Failed to activate conda env nn_env.
  echo.
  pause
  exit /b 1
)

python --version
if errorlevel 1 (
  echo [ERROR] Python in nn_env is not available.
  echo.
  pause
  exit /b 1
)

if not exist ".env" copy ".env.example" ".env" >nul

echo [Xiaoai] checking Ollama...
powershell -NoProfile -Command "try { $r = Invoke-RestMethod -Uri 'http://localhost:11434/api/tags' -TimeoutSec 5; $names = @($r.models.name); Write-Host '[OK] Ollama is running'; foreach ($m in @('qwen3.5:9b','qwen3-embedding:0.6b','llava:latest')) { if ($names -contains $m) { Write-Host ('  [found] ' + $m) } else { Write-Host ('  [missing] ' + $m) } } } catch { Write-Host '[HINT] Ollama is not connected. The page can still start.' }"

echo [Xiaoai] checking Python packages...
python -c "import fastapi,uvicorn,httpx,dotenv,pydantic; from websockets.asyncio.client import connect"
if errorlevel 1 (
  echo [Xiaoai] installing requirements...
  python -m pip install -r requirements.txt
  if errorlevel 1 (
    echo [ERROR] pip install failed.
    echo.
    pause
    exit /b 1
  )
)

echo.
echo [Xiaoai] open http://127.0.0.1:8000
echo [Xiaoai] press Ctrl+C to stop.
echo.
start "" cmd /c "timeout /t 3 /nobreak >nul & start http://127.0.0.1:8000"

python -m uvicorn ai_service.main:app --host 127.0.0.1 --port 8000
set "ERR=%ERRORLEVEL%"

echo.
if not "%ERR%"=="0" (
  echo [ERROR] server exited with code %ERR%.
  echo If the port is already in use, open http://127.0.0.1:8000 directly.
)
echo Press any key to close this window.
pause >nul
exit /b %ERR%
