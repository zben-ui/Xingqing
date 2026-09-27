from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ChatRequest(BaseModel):
    user_message: str = Field(min_length=1, max_length=4000)
    session_id: str = Field(min_length=1, max_length=128)


class SourceItem(BaseModel):
    file: str
    chunk: int
    score: float
    excerpt: str = ""
    metadata: dict[str, str] = Field(default_factory=dict)


class TTSRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4000)


class ChatResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    answer: str
    emotion: Literal["neutral", "friendly", "happy", "thinking", "comforting"]
    action: Literal["idle", "talk", "comfort", "encourage", "explain"]
    sources: list[SourceItem] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    safety_level: Literal["normal", "emotional_support", "crisis"] = "normal"
    model_available: bool = True


class ClearRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)


class VisionRequest(BaseModel):
    prompt: str = Field(default="请描述这张图片", max_length=2000)
    image_base64: str | None = None


class AuthRequest(BaseModel):
    """登录/注册请求体：用户名 2–32 位，密码至少 8 位。"""
    username: str = Field(min_length=2, max_length=32, pattern=r"^[\w\u4e00-\u9fff.-]+$")
    password: str = Field(min_length=8, max_length=128)


class ProfileRequest(BaseModel):
    """个人资料保存请求：身份信息 + 陪伴偏好，供聊天风格和管理端联系人使用。"""
    nickname: str = Field(default="", max_length=24)
    avatar_data: str = Field(default="", max_length=900_000)
    chat_style: str = Field(default="", max_length=1000)
    background_notes: str = Field(default="", max_length=4000)
    real_name: str = Field(default="", max_length=40)
    student_id: str = Field(default="", max_length=40)
    department: str = Field(default="", max_length=100)
    phone: str = Field(default="", max_length=30)
    emergency_contact: str = Field(default="", max_length=40)
    emergency_phone: str = Field(default="", max_length=30)


class AssessmentRequest(BaseModel):
    answers: dict[str, int]


class AssessmentQuestionUpload(BaseModel):
    id: str = Field(min_length=1, max_length=60, pattern=r"^[A-Za-z0-9_-]+$")
    text: str = Field(min_length=2, max_length=300)
    factor: Literal["anxiety", "obsessive", "somatization", "safety"]


class AssessmentTemplateRequest(BaseModel):
    title: str = Field(min_length=2, max_length=80)
    description: str = Field(default="", max_length=300)
    questions: list[AssessmentQuestionUpload] = Field(min_length=4, max_length=40)


class DiaryRequest(BaseModel):
    mood: int = Field(ge=1, le=5)
    note: str = Field(default="", max_length=4000)
    entry_date: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")


class ArticleRequest(BaseModel):
    title: str = Field(min_length=2, max_length=120)
    summary: str = Field(default="", max_length=300)
    content: str = Field(min_length=10, max_length=30_000)
    cover_data: str = Field(default="", max_length=900_000)


class KnowledgeUploadRequest(BaseModel):
    filename: str = Field(min_length=1, max_length=120)
    content: str = Field(min_length=1, max_length=1_000_000)
