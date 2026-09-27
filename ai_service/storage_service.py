from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .config import Settings, settings


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class StorageService:
    """本地 SQLite 持久化：账号登录、个人资料、测评、预警与管理端查询。"""

    def __init__(self, app_settings: Settings = settings, database_path: Path | None = None) -> None:
        self.path = database_path or app_settings.database_path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=20)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _init_database(self) -> None:
        """建表：users/sessions/profiles 管登录与资料，alerts 管管理端预警。"""
        with self._connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    password_hash TEXT NOT NULL,
                    salt TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'user',
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS sessions (
                    token TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    expires_at TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS profiles (
                    user_id INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
                    nickname TEXT NOT NULL DEFAULT '',
                    avatar_data TEXT NOT NULL DEFAULT '',
                    chat_style TEXT NOT NULL DEFAULT '',
                    background_notes TEXT NOT NULL DEFAULT '',
                    real_name TEXT NOT NULL DEFAULT '',
                    student_id TEXT NOT NULL DEFAULT '',
                    department TEXT NOT NULL DEFAULT '',
                    phone TEXT NOT NULL DEFAULT '',
                    emergency_contact TEXT NOT NULL DEFAULT '',
                    emergency_phone TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    safety_level TEXT NOT NULL DEFAULT 'normal',
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_messages_user_time
                    ON messages(user_id, created_at DESC);
                CREATE TABLE IF NOT EXISTS assessments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    score INTEGER NOT NULL,
                    risk_level TEXT NOT NULL,
                    answers_json TEXT NOT NULL,
                    factor_scores_json TEXT NOT NULL DEFAULT '{}',
                    template_id INTEGER,
                    summary TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    assessment_id INTEGER REFERENCES assessments(id) ON DELETE SET NULL,
                    content TEXT NOT NULL,
                    risk_level TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    source TEXT NOT NULL,
                    level TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    status TEXT NOT NULL DEFAULT 'pending',
                    created_at TEXT NOT NULL,
                    acknowledged_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_alerts_status_time
                    ON alerts(status, created_at DESC);
                CREATE TABLE IF NOT EXISTS knowledge_files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    filename TEXT NOT NULL,
                    relative_path TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(user_id, filename)
                );
                CREATE TABLE IF NOT EXISTS diary_entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    entry_date TEXT NOT NULL,
                    mood INTEGER NOT NULL,
                    mood_label TEXT NOT NULL,
                    note TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(user_id, entry_date)
                );
                CREATE TABLE IF NOT EXISTS articles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    summary TEXT NOT NULL DEFAULT '',
                    content TEXT NOT NULL,
                    cover_data TEXT NOT NULL DEFAULT '',
                    published INTEGER NOT NULL DEFAULT 1,
                    author_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS assessment_templates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL DEFAULT '',
                    questions_json TEXT NOT NULL,
                    active INTEGER NOT NULL DEFAULT 1,
                    created_by INTEGER REFERENCES users(id) ON DELETE SET NULL,
                    created_at TEXT NOT NULL
                );
                """
            )
            existing_columns = {
                row["name"] for row in db.execute("PRAGMA table_info(profiles)").fetchall()
            }
            profile_columns = (
                "nickname",
                "avatar_data",
                "real_name",
                "student_id",
                "department",
                "phone",
                "emergency_contact",
                "emergency_phone",
            )
            for column in profile_columns:
                if column not in existing_columns:
                    db.execute(
                        f"ALTER TABLE profiles ADD COLUMN {column} TEXT NOT NULL DEFAULT ''"
                    )
            assessment_columns = {
                row["name"] for row in db.execute("PRAGMA table_info(assessments)").fetchall()
            }
            if "factor_scores_json" not in assessment_columns:
                db.execute(
                    "ALTER TABLE assessments ADD COLUMN factor_scores_json TEXT NOT NULL DEFAULT '{}'"
                )
            if "template_id" not in assessment_columns:
                db.execute("ALTER TABLE assessments ADD COLUMN template_id INTEGER")
            if "safety_signal" not in assessment_columns:
                db.execute("ALTER TABLE assessments ADD COLUMN safety_signal INTEGER NOT NULL DEFAULT 0")
            alert_columns = {
                row["name"] for row in db.execute("PRAGMA table_info(alerts)").fetchall()
            }
            if "metadata_json" not in alert_columns:
                db.execute(
                    "ALTER TABLE alerts ADD COLUMN metadata_json TEXT NOT NULL DEFAULT '{}'"
                )

    @staticmethod
    def _hash_password(password: str, salt_hex: str) -> str:
        """口令加盐哈希（PBKDF2-SHA256），不明文存密码。"""
        digest = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), 310_000
        )
        return digest.hex()

    @staticmethod
    def _public_user(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "username": row["username"],
            "role": row["role"],
            "created_at": row["created_at"],
        }

    def register(self, username: str, password: str) -> tuple[dict[str, Any], str]:
        """注册用户；users 为空时第一人 role=admin，并创建空 profiles 行。"""
        username = username.strip()
        salt = secrets.token_hex(16)
        now = _now()
        with self._connect() as db:
            role = "admin" if db.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0 else "user"
            try:
                cursor = db.execute(
                    "INSERT INTO users(username, password_hash, salt, role, created_at) VALUES(?,?,?,?,?)",
                    (username, self._hash_password(password, salt), salt, role, now),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError("用户名已存在") from exc
            user_id = int(cursor.lastrowid)
            db.execute(
                "INSERT INTO profiles(user_id, updated_at) VALUES(?,?)", (user_id, now)
            )
            row = db.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return self._public_user(row), self.create_session(user_id)

    def authenticate(self, username: str, password: str) -> tuple[dict[str, Any], str]:
        """登录校验：重算哈希后恒定时间比较，成功则发 session token。"""
        with self._connect() as db:
            row = db.execute(
                "SELECT * FROM users WHERE username = ? COLLATE NOCASE", (username.strip(),)
            ).fetchone()
        if not row:
            raise ValueError("用户名或密码不正确")
        actual = self._hash_password(password, row["salt"])
        if not hmac.compare_digest(actual, row["password_hash"]):
            raise ValueError("用户名或密码不正确")
        return self._public_user(row), self.create_session(int(row["id"]))

    def create_session(self, user_id: int) -> str:
        """生成随机 token，写入 sessions，默认 30 天有效。"""
        token = secrets.token_urlsafe(32)
        expires = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()
        with self._connect() as db:
            db.execute(
                "INSERT INTO sessions(token, user_id, expires_at, created_at) VALUES(?,?,?,?)",
                (token, user_id, expires, _now()),
            )
        return token

    def user_from_token(self, token: str) -> dict[str, Any] | None:
        """用 token 联查 users；过期或不存在返回 None。"""
        with self._connect() as db:
            row = db.execute(
                """
                SELECT u.* FROM sessions s JOIN users u ON u.id = s.user_id
                WHERE s.token = ? AND s.expires_at > ?
                """,
                (token, _now()),
            ).fetchone()
        return self._public_user(row) if row else None

    def revoke_session(self, token: str) -> None:
        with self._connect() as db:
            db.execute("DELETE FROM sessions WHERE token = ?", (token,))

    def get_profile(self, user_id: int) -> dict[str, str]:
        """读取个人资料（昵称、头像、联系方式、陪伴偏好）。"""
        with self._connect() as db:
            row = db.execute("SELECT * FROM profiles WHERE user_id = ?", (user_id,)).fetchone()
        return {
            "nickname": row["nickname"] if row else "",
            "avatar_data": row["avatar_data"] if row else "",
            "chat_style": row["chat_style"] if row else "",
            "background_notes": row["background_notes"] if row else "",
            "real_name": row["real_name"] if row else "",
            "student_id": row["student_id"] if row else "",
            "department": row["department"] if row else "",
            "phone": row["phone"] if row else "",
            "emergency_contact": row["emergency_contact"] if row else "",
            "emergency_phone": row["emergency_phone"] if row else "",
        }

    def update_profile(
        self,
        user_id: int,
        nickname: str,
        avatar_data: str,
        chat_style: str,
        background_notes: str,
        real_name: str = "",
        student_id: str = "",
        department: str = "",
        phone: str = "",
        emergency_contact: str = "",
        emergency_phone: str = "",
    ) -> dict[str, str]:
        """按 user_id  upsert 个人资料，供用户端和聊天风格使用。"""
        with self._connect() as db:
            db.execute(
                """
                INSERT INTO profiles(
                    user_id, nickname, avatar_data, chat_style, background_notes, real_name, student_id,
                    department, phone, emergency_contact, emergency_phone, updated_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(user_id) DO UPDATE SET
                    nickname=excluded.nickname,
                    avatar_data=excluded.avatar_data,
                    chat_style=excluded.chat_style,
                    background_notes=excluded.background_notes,
                    real_name=excluded.real_name,
                    student_id=excluded.student_id,
                    department=excluded.department,
                    phone=excluded.phone,
                    emergency_contact=excluded.emergency_contact,
                    emergency_phone=excluded.emergency_phone,
                    updated_at=excluded.updated_at
                """,
                (
                    user_id,
                    nickname.strip(),
                    avatar_data.strip(),
                    chat_style.strip(),
                    background_notes.strip(),
                    real_name.strip(),
                    student_id.strip(),
                    department.strip(),
                    phone.strip(),
                    emergency_contact.strip(),
                    emergency_phone.strip(),
                    _now(),
                ),
            )
        return self.get_profile(user_id)

    def save_message(
        self, user_id: int, session_id: str, role: str, content: str, safety_level: str = "normal"
    ) -> None:
        with self._connect() as db:
            db.execute(
                "INSERT INTO messages(user_id, session_id, role, content, safety_level, created_at) VALUES(?,?,?,?,?,?)",
                (user_id, session_id, role, content, safety_level, _now()),
            )

    def list_messages(self, user_id: int, limit: int = 40) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT role, content, safety_level, created_at FROM messages WHERE user_id = ? ORDER BY id DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        return [dict(row) for row in reversed(rows)]

    def save_assessment(
        self,
        user_id: int,
        score: int,
        risk_level: str,
        answers: dict[str, int],
        factor_scores: dict[str, Any],
        summary: str,
        template_id: int | None = None,
        safety_signal: bool = False,
    ) -> dict[str, Any]:
        now = _now()
        with self._connect() as db:
            cursor = db.execute(
                """
                INSERT INTO assessments(
                    user_id, score, risk_level, answers_json, factor_scores_json,
                    template_id, summary, created_at, safety_signal
                ) VALUES(?,?,?,?,?,?,?,?,?)
                """,
                (
                    user_id,
                    score,
                    risk_level,
                    json.dumps(answers, ensure_ascii=False),
                    json.dumps(factor_scores, ensure_ascii=False),
                    template_id,
                    summary,
                    now,
                    int(safety_signal),
                ),
            )
            assessment_id = int(cursor.lastrowid)
        return {
            "id": assessment_id,
            "score": score,
            "risk_level": risk_level,
            "answers": answers,
            "factor_scores": factor_scores,
            "template_id": template_id,
            "safety_signal": safety_signal,
            "summary": summary,
            "created_at": now,
        }

    @staticmethod
    def _split_factor_payload(payload: Any) -> tuple[dict[str, Any], list[Any], dict[str, Any]]:
        if not isinstance(payload, dict):
            return {}, [], {}
        if isinstance(payload.get("factors"), dict):
            classification = payload.get("classification")
            return (
                payload["factors"],
                list(payload.get("elevated_factors") or []),
                classification if isinstance(classification, dict) else {},
            )
        factors = {
            key: value
            for key, value in payload.items()
            if isinstance(value, dict) and value.get("name")
        }
        elevated = payload.get("_elevated") or payload.get("elevated_factors") or []
        classification = payload.get("_classification") or payload.get("classification") or {}
        return factors, list(elevated), classification if isinstance(classification, dict) else {}

    def latest_assessment(self, user_id: int) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute(
                "SELECT * FROM assessments WHERE user_id = ? ORDER BY id DESC LIMIT 1", (user_id,)
            ).fetchone()
        if not row:
            return None
        result = dict(row)
        packed = json.loads(result.pop("factor_scores_json") or "{}")
        factors, elevated, classification = StorageService._split_factor_payload(packed)
        result["answers"] = json.loads(result.pop("answers_json"))
        result["factor_scores"] = factors
        result["elevated_factors"] = elevated
        result["classification"] = classification
        result["safety_signal"] = bool(result.get("safety_signal"))
        return result

    def save_report(
        self, user_id: int, assessment_id: int | None, content: str, risk_level: str
    ) -> dict[str, Any]:
        now = _now()
        with self._connect() as db:
            cursor = db.execute(
                "INSERT INTO reports(user_id, assessment_id, content, risk_level, created_at) VALUES(?,?,?,?,?)",
                (user_id, assessment_id, content, risk_level, now),
            )
            report_id = int(cursor.lastrowid)
        return {
            "id": report_id,
            "content": content,
            "risk_level": risk_level,
            "created_at": now,
        }

    def create_alert(
        self,
        user_id: int,
        source: str,
        level: str,
        reason: str,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """写入管理端预警（危机词、高分测评等），默认 status=pending。"""
        with self._connect() as db:
            cursor = db.execute(
                """
                INSERT INTO alerts(
                    user_id, source, level, reason, metadata_json, status, created_at
                ) VALUES(?,?,?,?,?,?,?)
                """,
                (
                    user_id,
                    source,
                    level,
                    reason,
                    json.dumps(metadata or {}, ensure_ascii=False),
                    "pending",
                    _now(),
                ),
            )
            alert_id = int(cursor.lastrowid)
        return alert_id

    def acknowledge_alert(self, alert_id: int) -> bool:
        """管理员标记已核实。"""
        with self._connect() as db:
            cursor = db.execute(
                "UPDATE alerts SET status='acknowledged', acknowledged_at=? WHERE id=?",
                (_now(), alert_id),
            )
        return cursor.rowcount > 0

    @staticmethod
    def _mask_username(username: str) -> str:
        if len(username) <= 1:
            return "*"
        return username[0] + "*" * min(2, len(username) - 1)

    def admin_overview(self) -> dict[str, Any]:
        """统计待处理预警、高风险用户、今日新增等，供顶部指标卡。"""
        today = datetime.now(timezone.utc).date().isoformat()
        seven_days_ago = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
        with self._connect() as db:
            active = db.execute(
                "SELECT COUNT(DISTINCT user_id) FROM messages WHERE created_at >= ?",
                (seven_days_ago,),
            ).fetchone()[0]
            pending = db.execute("SELECT COUNT(*) FROM alerts WHERE status='pending'").fetchone()[0]
            alerts_today = db.execute(
                "SELECT COUNT(*) FROM alerts WHERE substr(created_at,1,10)=?", (today,)
            ).fetchone()[0]
            needs_follow_up = db.execute(
                "SELECT COUNT(DISTINCT user_id) FROM alerts WHERE status='pending'"
            ).fetchone()[0]
            assessments = db.execute(
                "SELECT COUNT(*) FROM assessments WHERE substr(created_at,1,10)=?", (today,)
            ).fetchone()[0]
            users = db.execute("SELECT COUNT(*) FROM users").fetchone()[0]
            risk_rows = db.execute(
                """
                SELECT risk_level, COUNT(*) AS count FROM (
                    SELECT CASE WHEN EXISTS(SELECT 1 FROM alerts al WHERE al.user_id=u.id AND al.status='pending' AND al.level='high')
                        THEN 'high' ELSE COALESCE((SELECT risk_level FROM assessments a WHERE a.user_id=u.id ORDER BY id DESC LIMIT 1), 'unknown') END AS risk_level
                    FROM users u WHERE u.role='user'
                )
                GROUP BY risk_level
                """
            ).fetchall()
        distribution = {"high": 0, "medium": 0, "low": 0}
        for row in risk_rows:
            if row["risk_level"] in distribution:
                distribution[row["risk_level"]] = int(row["count"])
        return {
            "active_users": int(active),
            "pending_alerts": int(pending),
            "high_risk_users": int(distribution["high"]),
            "alerts_today": int(alerts_today),
            "needs_follow_up": int(needs_follow_up),
            "assessments_today": int(assessments),
            "total_users": int(users),
            "risk_distribution": distribution,
        }

    def list_alerts(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute(
                """
                SELECT a.*, u.username, p.real_name, p.nickname
                FROM alerts a
                JOIN users u ON u.id=a.user_id
                LEFT JOIN profiles p ON p.user_id=a.user_id
                ORDER BY CASE a.status WHEN 'pending' THEN 0 ELSE 1 END,
                         CASE a.level WHEN 'high' THEN 0 WHEN 'medium' THEN 1 ELSE 2 END,
                         a.id DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["metadata"] = json.loads(item.pop("metadata_json") or "{}")
            item["username"] = self._mask_username(item["username"])
            item["display_name"] = item["nickname"] or item["real_name"] or item["username"]
            result.append(item)
        return result

    def list_users(self, limit: int = 100) -> list[dict[str, Any]]:
        """管理端队列：联查资料、最近测评因子、待处理预警数。"""
        with self._connect() as db:
            rows = db.execute(
                """
                SELECT u.id, u.username, u.role, u.created_at,
                       p.nickname, p.avatar_data, p.real_name, p.student_id, p.department, p.phone,
                       p.emergency_contact, p.emergency_phone,
                       MAX(m.created_at) AS last_active,
                       CASE WHEN EXISTS(SELECT 1 FROM alerts al WHERE al.user_id=u.id AND al.status='pending' AND al.level='high') THEN 'high'
                            ELSE COALESCE((SELECT risk_level FROM assessments a WHERE a.user_id=u.id ORDER BY a.id DESC LIMIT 1), 'unknown') END AS risk_level,
                       COALESCE((SELECT factor_scores_json FROM assessments a WHERE a.user_id=u.id ORDER BY a.id DESC LIMIT 1), '{}') AS factor_scores_json,
                       COALESCE((SELECT metadata_json FROM alerts al WHERE al.user_id=u.id AND al.source IN ('chat_keyword','diary_keyword') ORDER BY al.id DESC LIMIT 1), '{}') AS keyword_metadata_json,
                       (SELECT COUNT(*) FROM alerts al WHERE al.user_id=u.id AND al.status='pending') AS pending_alerts,
                       CASE WHEN EXISTS(SELECT 1 FROM alerts al WHERE al.user_id=u.id AND al.status='pending' AND al.level='high' AND al.source IN ('chat_keyword','diary_keyword')) THEN 3
                            WHEN EXISTS(SELECT 1 FROM alerts al WHERE al.user_id=u.id AND al.status='pending' AND al.level='high') THEN 2
                            WHEN EXISTS(SELECT 1 FROM alerts al WHERE al.user_id=u.id AND al.status='pending') THEN 1 ELSE 0 END AS attention_priority
                FROM users u
                LEFT JOIN profiles p ON p.user_id=u.id
                LEFT JOIN messages m ON m.user_id=u.id
                WHERE u.role='user'
                GROUP BY u.id ORDER BY attention_priority DESC, pending_alerts DESC, COALESCE(last_active, u.created_at) DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["factor_scores"], item["elevated_factors"], item["classification"] = StorageService._split_factor_payload(
                json.loads(item.pop("factor_scores_json") or "{}")
            )
            keyword_metadata = json.loads(item.pop("keyword_metadata_json") or "{}")
            item["keyword_hits"] = keyword_metadata.get("matched_keywords", [])
            item["username"] = self._mask_username(item["username"])
            item["display_name"] = item["nickname"] or item["real_name"] or item["username"]
            result.append(item)
        return result

    def user_detail(self, user_id: int) -> dict[str, Any] | None:
        """单人核实页：profile + 最近测评 + 消息 + 报告 + 预警。"""
        with self._connect() as db:
            user = db.execute("SELECT id, username, role, created_at FROM users WHERE id=?", (user_id,)).fetchone()
            if not user:
                return None
            reports = db.execute(
                "SELECT id, content, risk_level, created_at FROM reports WHERE user_id=? ORDER BY id DESC LIMIT 5",
                (user_id,),
            ).fetchall()
            alerts = db.execute(
                "SELECT * FROM alerts WHERE user_id=? ORDER BY id DESC LIMIT 20",
                (user_id,),
            ).fetchall()
        result = dict(user)
        result["username"] = self._mask_username(result["username"])
        result["profile"] = self.get_profile(user_id)
        result["assessment"] = self.latest_assessment(user_id)
        result["messages"] = self.list_messages(user_id, 20)
        result["reports"] = [dict(row) for row in reports]
        result["alerts"] = []
        for row in alerts:
            alert = dict(row)
            alert["metadata"] = json.loads(alert.pop("metadata_json") or "{}")
            result["alerts"].append(alert)
        return result

    def save_diary(
        self, user_id: int, mood: int, note: str, entry_date: str | None = None
    ) -> dict[str, Any]:
        labels = {1: "低落", 2: "有点累", 3: "平静", 4: "不错", 5: "开心"}
        date = entry_date or datetime.now(timezone(timedelta(hours=8))).date().isoformat()
        now = _now()
        with self._connect() as db:
            db.execute(
                """
                INSERT INTO diary_entries(
                    user_id, entry_date, mood, mood_label, note, created_at, updated_at
                ) VALUES(?,?,?,?,?,?,?)
                ON CONFLICT(user_id, entry_date) DO UPDATE SET
                    mood=excluded.mood,
                    mood_label=excluded.mood_label,
                    note=excluded.note,
                    updated_at=excluded.updated_at
                """,
                (user_id, date, mood, labels[mood], note.strip(), now, now),
            )
            row = db.execute(
                "SELECT * FROM diary_entries WHERE user_id=? AND entry_date=?",
                (user_id, date),
            ).fetchone()
        return dict(row)

    def list_diary(self, user_id: int, limit: int = 30) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute(
                """
                SELECT id, entry_date, mood, mood_label, note, created_at, updated_at
                FROM diary_entries WHERE user_id=? ORDER BY entry_date DESC LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def save_article(
        self,
        author_id: int,
        title: str,
        summary: str,
        content: str,
        cover_data: str = "",
    ) -> dict[str, Any]:
        now = _now()
        with self._connect() as db:
            cursor = db.execute(
                """
                INSERT INTO articles(
                    title, summary, content, cover_data, published, author_id, created_at
                ) VALUES(?,?,?,?,1,?,?)
                """,
                (title.strip(), summary.strip(), content.strip(), cover_data.strip(), author_id, now),
            )
            article_id = int(cursor.lastrowid)
        return {
            "id": article_id,
            "title": title.strip(),
            "summary": summary.strip(),
            "content": content.strip(),
            "cover_data": cover_data.strip(),
            "created_at": now,
        }

    def list_articles(self, limit: int = 40) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute(
                """
                SELECT id, title, summary, content, cover_data, created_at
                FROM articles WHERE published=1 ORDER BY id DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def save_assessment_template(
        self,
        created_by: int,
        title: str,
        description: str,
        questions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        now = _now()
        with self._connect() as db:
            db.execute("UPDATE assessment_templates SET active=0")
            cursor = db.execute(
                """
                INSERT INTO assessment_templates(
                    title, description, questions_json, active, created_by, created_at
                ) VALUES(?,?,?,1,?,?)
                """,
                (
                    title.strip(),
                    description.strip(),
                    json.dumps(questions, ensure_ascii=False),
                    created_by,
                    now,
                ),
            )
            template_id = int(cursor.lastrowid)
        return {
            "id": template_id,
            "title": title.strip(),
            "description": description.strip(),
            "questions": questions,
            "active": True,
            "created_at": now,
        }

    def active_assessment_template(self) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute(
                "SELECT * FROM assessment_templates WHERE active=1 ORDER BY id DESC LIMIT 1"
            ).fetchone()
        if not row:
            return None
        result = dict(row)
        result["questions"] = json.loads(result.pop("questions_json"))
        result["active"] = bool(result["active"])
        return result

    def upsert_knowledge_file(self, user_id: int, filename: str, relative_path: str) -> None:
        with self._connect() as db:
            db.execute(
                """
                INSERT INTO knowledge_files(user_id, filename, relative_path, created_at)
                VALUES(?,?,?,?)
                ON CONFLICT(user_id, filename) DO UPDATE SET
                    relative_path=excluded.relative_path, created_at=excluded.created_at
                """,
                (user_id, filename, relative_path, _now()),
            )

    def list_knowledge_files(self, user_id: int) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT id, filename, relative_path, created_at FROM knowledge_files WHERE user_id=? ORDER BY id DESC",
                (user_id,),
            ).fetchall()
        return [dict(row) for row in rows]
