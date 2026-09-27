from __future__ import annotations

from contextlib import asynccontextmanager
import json
from pathlib import Path
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.responses import FileResponse, StreamingResponse, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles

from .agent_service import AgentService
from .assessment_service import AssessmentService
from .config import PROJECT_ROOT, settings
from .content_service import ContentService
from .memory_service import MemoryService
from .ollama_client import OllamaClient, OllamaError
from .rag_service import RagService
from .report_service import ReportService
from .safety_service import SafetyService
from .screening_classifier import ScreeningClassifier
from .schemas import (
    ArticleRequest,
    AssessmentRequest,
    AssessmentTemplateRequest,
    AuthRequest,
    ChatRequest,
    ChatResponse,
    ClearRequest,
    DiaryRequest,
    KnowledgeUploadRequest,
    ProfileRequest,
    TTSRequest,
    VisionRequest,
)
from .storage_service import StorageService
from .tts_service import TTSService, TTSError


ollama = OllamaClient()
memory = MemoryService()
safety = SafetyService()
rag = RagService(ollama)
agent = AgentService(ollama, rag, memory, safety)
tts = TTSService()
storage = StorageService()
assessment = AssessmentService()
reports = ReportService(ollama, storage, rag)
content = ContentService(PROJECT_ROOT / "image")
bearer = HTTPBearer(auto_error=False)


@asynccontextmanager
async def lifespan(_: FastAPI):
    if rag._files() and not settings.vector_store_path.exists():
        await rag.rebuild()
    yield


app = FastAPI(
    title="小艾情感陪伴数字人",
    description="Local Ollama powered emotional companion MVP",
    version="0.1.0",
    lifespan=lifespan,
)


def _credentials_token(credentials: HTTPAuthorizationCredentials | None) -> str:
    """从 Authorization: Bearer 取出 token，缺失则 401。"""
    if not credentials or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="请先登录")
    return credentials.credentials


def current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
) -> dict[str, object]:
    """普通接口鉴权：用 token 查 sessions 表，过期则 401。"""
    user = storage.user_from_token(_credentials_token(credentials))
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="登录已失效")
    return user


def admin_user(user: dict[str, object] = Depends(current_user)) -> dict[str, object]:
    """管理端鉴权：必须先登录且 users.role == admin，否则 403。"""
    if user.get("role") != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="需要管理员权限")
    return user


def _record_crisis_alert(user_id: int, text: str, source: str = "chat_keyword") -> None:
    detected = safety.detect(text)
    matches = list(detected.matched_keywords)
    if detected.level != "crisis":
        return
    visible = "、".join(matches[:4]) if matches else "即时安全信号"
    storage.create_alert(
        user_id,
        source,
        "high",
        f"优先安全核实 · 高危词感知：{visible}",
        {"matched_keywords": matches, "requires_human_review": True,
         "attention_priority": "urgent", "evidence_excerpt": text[:200],
         "note": "关键词命中不等于本人存在自杀意图，须结合语境人工核实。"},
    )


def _chat_profile(user_id: int):
    profile = storage.get_profile(user_id)
    latest = storage.latest_assessment(user_id)
    if latest:
        profile["assessment_context"] = json.dumps(
            {"created_at": latest.get("created_at"),
             "classification": ScreeningClassifier().model_context(latest)},
            ensure_ascii=False,
        )
    return profile


@app.post("/api/auth/register")
async def register(request: AuthRequest) -> dict[str, object]:
    """注册：空库第一个账号自动成为管理员，返回 user + token。"""
    try:
        user, token = storage.register(request.username, request.password)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"token": token, "user": user}


@app.post("/api/auth/login")
async def login(request: AuthRequest) -> dict[str, object]:
    """用户登录：校验密码哈希，写入 sessions 并返回 token。"""
    try:
        user, token = storage.authenticate(request.username, request.password)
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    return {"token": token, "user": user}


@app.post("/api/auth/admin-login")
async def admin_login(request: AuthRequest) -> dict[str, object]:
    """管理员登录：密码正确但非 admin 会立刻作废 token。"""
    try:
        user, token = storage.authenticate(request.username, request.password)
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    if user.get("role") != "admin":
        storage.revoke_session(token)
        raise HTTPException(status_code=403, detail="该账号没有管理员权限")
    return {"token": token, "user": user}


@app.get("/api/auth/me")
async def me(user: dict[str, object] = Depends(current_user)) -> dict[str, object]:
    """刷新页面时用已有 token 恢复登录态。"""
    return {"user": user}


@app.post("/api/auth/logout")
async def logout(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
    _: dict[str, object] = Depends(current_user),
) -> dict[str, bool]:
    """退出登录：删除 sessions 行。"""
    storage.revoke_session(_credentials_token(credentials))
    return {"success": True}


@app.get("/api/health")
async def health() -> dict[str, object]:
    ollama_health = await ollama.health_check()
    return {
        "status": "ok" if ollama_health.get("available") else "degraded",
        "name": settings.digital_human_name,
        "ollama": ollama_health,
        "rag": {"chunks": len(rag.chunks), "mode": rag.last_mode},
        "vision_enabled": settings.enable_vision,
        "tts": tts.status(),
        "agent": rag.corpus.status(),
    }


@app.post("/api/agent/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest, user: dict[str, object] = Depends(current_user)
) -> dict[str, object]:
    user_id = int(user["id"])
    level = safety.detect(request.user_message).level
    storage.save_message(user_id, request.session_id, "user", request.user_message, level)
    _record_crisis_alert(user_id, request.user_message)
    profile = _chat_profile(user_id)
    result = await agent.chat(
        request.user_message,
        f"user-{user_id}:{request.session_id}",
        user_id=user_id,
        profile=profile,
    )
    storage.save_message(user_id, request.session_id, "assistant", str(result["answer"]), str(result["safety_level"]))
    return result


@app.post("/api/agent/chat/stream")
async def chat_stream(
    request: ChatRequest, user: dict[str, object] = Depends(current_user)
) -> StreamingResponse:
    user_id = int(user["id"])
    level = safety.detect(request.user_message).level
    storage.save_message(user_id, request.session_id, "user", request.user_message, level)
    _record_crisis_alert(user_id, request.user_message)
    profile = _chat_profile(user_id)

    async def events():
        async for event in agent.chat_stream(
            request.user_message,
            f"user-{user_id}:{request.session_id}",
            user_id=user_id,
            profile=profile,
        ):
            if event.get("type") == "done":
                level = str(event["safety_level"])
                storage.save_message(
                    user_id, request.session_id, "assistant", str(event["answer"]), level
                )
            yield json.dumps(event, ensure_ascii=False) + "\n"

    return StreamingResponse(events(), media_type="application/x-ndjson")


@app.post("/api/agent/clear")
async def clear(
    request: ClearRequest, user: dict[str, object] = Depends(current_user)
) -> dict[str, object]:
    removed = memory.clear(f"user-{int(user['id'])}:{request.session_id}")
    return {"success": True, "cleared": removed, "session_id": request.session_id}


@app.post("/api/rag/rebuild")
async def rebuild_rag(_: dict[str, object] = Depends(current_user)) -> dict[str, object]:
    result = await rag.rebuild()
    return {"success": True, **result}


@app.get("/api/profile")
async def get_profile(user: dict[str, object] = Depends(current_user)) -> dict[str, str]:
    """读取当前用户 profiles 表记录。"""
    return storage.get_profile(int(user["id"]))


@app.put("/api/profile")
async def update_profile(
    request: ProfileRequest, user: dict[str, object] = Depends(current_user)
) -> dict[str, str]:
    """保存昵称、头像、联系方式与陪伴偏好。"""
    return storage.update_profile(
        int(user["id"]),
        request.nickname,
        request.avatar_data,
        request.chat_style,
        request.background_notes,
        request.real_name,
        request.student_id,
        request.department,
        request.phone,
        request.emergency_contact,
        request.emergency_phone,
    )


@app.get("/api/assessments/questions")
async def assessment_questions(_: dict[str, object] = Depends(current_user)) -> dict[str, object]:
    return assessment.questions()


@app.post("/api/assessments")
async def submit_assessment(
    request: AssessmentRequest, user: dict[str, object] = Depends(current_user)
) -> dict[str, object]:
    question_set = assessment.questions()
    try:
        result = assessment.score(request.answers, list(question_set["questions"]))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    user_id = int(user["id"])
    packed_scores = {
        "factors": dict(result["factor_scores"]),
        "elevated_factors": result.get("elevated_factors") or [],
        "classification": result.get("classification") or {},
        "references": result.get("references") or [],
    }
    saved = storage.save_assessment(
        user_id,
        int(result["score"]),
        str(result["risk_level"]),
        request.answers,
        packed_scores,
        str(result["summary"]),
        question_set.get("id"),
        bool(result["safety_signal"]),
    )
    saved["factor_scores"] = dict(result["factor_scores"])
    saved["elevated_factors"] = packed_scores["elevated_factors"]
    saved["classification"] = packed_scores["classification"]
    saved.update({key: value for key, value in result.items() if key not in saved})
    if result["risk_level"] in {"medium", "high"}:
        reason = "状态测评出现安全题信号" if result["safety_signal"] else "状态测评分值需要关注"
        storage.create_alert(
            user_id,
            "assessment",
            str(result["risk_level"]),
            reason,
            {
                "factor_scores": result["factor_scores"],
                "safety_signal": result["safety_signal"],
                "requires_human_review": True,
            },
        )
    return saved


@app.get("/api/assessments/latest")
async def latest_assessment(user: dict[str, object] = Depends(current_user)) -> dict[str, object]:
    return {"assessment": storage.latest_assessment(int(user["id"]))}


@app.get("/api/diary")
async def list_diary(user: dict[str, object] = Depends(current_user)) -> dict[str, object]:
    return {"entries": storage.list_diary(int(user["id"]))}


@app.post("/api/diary")
async def save_diary(
    request: DiaryRequest, user: dict[str, object] = Depends(current_user)
) -> dict[str, object]:
    user_id = int(user["id"])
    saved = storage.save_diary(user_id, request.mood, request.note, request.entry_date)
    _record_crisis_alert(user_id, request.note, "diary_keyword")
    return saved


@app.get("/api/articles")
async def list_articles(_: dict[str, object] = Depends(current_user)) -> dict[str, object]:
    return {"articles": storage.list_articles() + content.articles()}


@app.post("/api/tts")
async def synthesize_voice(request: TTSRequest, _: dict[str, object] = Depends(current_user)):
    try:
        audio = await tts.synthesize(request.text)
    except TTSError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return Response(audio, media_type="audio/mpeg", headers={"Cache-Control": "no-store"})


@app.post("/api/reports/generate")
async def generate_report(user: dict[str, object] = Depends(current_user)) -> dict[str, object]:
    try:
        return await reports.generate(int(user["id"]))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/api/knowledge")
async def list_knowledge(user: dict[str, object] = Depends(current_user)) -> dict[str, object]:
    return {"files": storage.list_knowledge_files(int(user["id"]))}


@app.post("/api/knowledge")
async def upload_knowledge(
    request: KnowledgeUploadRequest, user: dict[str, object] = Depends(current_user)
) -> dict[str, object]:
    filename = Path(request.filename).name
    if Path(filename).suffix.lower() not in {".txt", ".md"}:
        raise HTTPException(status_code=415, detail="仅支持 .txt 和 .md 文件")
    user_id = int(user["id"])
    folder = settings.knowledge_base_dir / "users" / str(user_id)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / filename
    path.write_text(request.content, encoding="utf-8")
    relative = path.relative_to(settings.knowledge_base_dir).as_posix()
    storage.upsert_knowledge_file(user_id, filename, relative)
    rebuild = await rag.rebuild()
    return {"success": True, "filename": filename, "rag": rebuild}


@app.post("/api/vision/chat")
async def vision_chat(request: VisionRequest) -> dict[str, object]:
    if not settings.enable_vision:
        return {"enabled": False, "answer": "多模态功能已预留，当前版本暂未启用。"}
    try:
        answer = await ollama.vision_chat(request.prompt, request.image_base64 or "")
        return {"enabled": True, "answer": answer}
    except OllamaError as exc:
        return {"enabled": True, "answer": "图片理解暂时不可用，请稍后再试。", "error": str(exc)}


@app.get("/api/admin/overview")
async def admin_overview(_: dict[str, object] = Depends(admin_user)) -> dict[str, object]:
    """管理端顶部统计：待处理预警、高风险人数等。"""
    return storage.admin_overview()


@app.get("/api/admin/alerts")
async def admin_alerts(_: dict[str, object] = Depends(admin_user)) -> dict[str, object]:
    """预警流水，供右侧列表展示。"""
    return {"alerts": storage.list_alerts()}


@app.post("/api/admin/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(
    alert_id: int, _: dict[str, object] = Depends(admin_user)
) -> dict[str, bool]:
    """人工核实：只改 alerts.status，不自动联系学生。"""
    if not storage.acknowledge_alert(alert_id):
        raise HTTPException(status_code=404, detail="预警不存在")
    return {"success": True}


@app.get("/api/admin/users")
async def admin_users(_: dict[str, object] = Depends(admin_user)) -> dict[str, object]:
    """风险队列：联系方式 + 最近一次测评因子。"""
    return {"users": storage.list_users()}


@app.get("/api/admin/users/{user_id}")
async def admin_user_detail(
    user_id: int, _: dict[str, object] = Depends(admin_user)
) -> dict[str, object]:
    """单用户核实详情：资料、测评、消息、报告、预警。"""
    detail = storage.user_detail(user_id)
    if not detail:
        raise HTTPException(status_code=404, detail="用户不存在")
    return detail


@app.post("/api/admin/assessment-templates")
async def upload_assessment_template(
    request: AssessmentTemplateRequest,
    user: dict[str, object] = Depends(admin_user),
) -> dict[str, object]:
    questions = [item.model_dump() for item in request.questions]
    try:
        assessment.validate_questions(questions)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return storage.save_assessment_template(
        int(user["id"]), request.title, request.description, questions
    )


@app.post("/api/admin/articles")
async def publish_article(
    request: ArticleRequest,
    user: dict[str, object] = Depends(admin_user),
) -> dict[str, object]:
    return storage.save_article(
        int(user["id"]),
        request.title,
        request.summary,
        request.content,
        request.cover_data,
    )


FRONTEND_DIR = PROJECT_ROOT / "frontend"
MODEL_MANIFEST_PATH = FRONTEND_DIR / "assets" / "models" / "xiaoai_manifest.json"


@app.get("/api/digital-human/model")
async def digital_human_model() -> dict[str, object]:
    manifest: dict[str, object] = {}
    if MODEL_MANIFEST_PATH.exists():
        manifest = json.loads(MODEL_MANIFEST_PATH.read_text(encoding="utf-8"))
    ollama_health = await ollama.health_check()
    manifest["ollama"] = ollama_health
    manifest["resolved"] = {
        "active_model_url": manifest.get(
            "active_model_url", "/static/assets/models/pinkgirl.glb"
        ),
        "manifest_url": "/static/assets/models/xiaoai_manifest.json",
        "runtime": manifest.get("active_runtime", "model-viewer"),
    }
    return manifest


app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")
