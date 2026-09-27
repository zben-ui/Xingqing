import unittest
import uuid
from dataclasses import replace
from pathlib import Path
import tempfile
from unittest.mock import patch

from fastapi.testclient import TestClient

from ai_service.main import app
from ai_service import main
from ai_service.config import settings
from ai_service.storage_service import StorageService


class SafetyApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        isolated = StorageService(replace(settings, database_path=Path(cls.temporary.name) / "api-test.db"))
        cls.storage_patch = patch.object(main, "storage", isolated)
        cls.report_patch = patch.object(main.reports, "storage", isolated)
        cls.storage_patch.start()
        cls.report_patch.start()

    @classmethod
    def tearDownClass(cls):
        cls.report_patch.stop()
        cls.storage_patch.stop()
        cls.temporary.cleanup()

    def test_admin_login_and_contact_profile(self) -> None:
        with TestClient(app) as client:
            admin_name = f"admin_{uuid.uuid4().hex[:8]}"
            admin = client.post(
                "/api/auth/register",
                json={"username": admin_name, "password": "test-password-123"},
            ).json()
            self.assertEqual(admin["user"]["role"], "admin")
            admin_login = client.post(
                "/api/auth/admin-login",
                json={"username": admin_name, "password": "test-password-123"},
            )
            self.assertEqual(admin_login.status_code, 200)
            admin_headers = {
                "Authorization": f"Bearer {admin_login.json()['token']}"
            }

            user_name = f"student_{uuid.uuid4().hex[:8]}"
            student = client.post(
                "/api/auth/register",
                json={"username": user_name, "password": "test-password-123"},
            ).json()
            self.assertEqual(student["user"]["role"], "user")
            student_headers = {"Authorization": f"Bearer {student['token']}"}
            saved = client.put(
                "/api/profile",
                headers=student_headers,
                json={
                    "nickname": "小晴",
                    "avatar_data": "",
                    "chat_style": "简短温柔",
                    "background_notes": "近期备考",
                    "real_name": "测试学生",
                    "student_id": "QA-001",
                    "department": "测试学院",
                    "phone": "13800000000",
                    "emergency_contact": "家长",
                    "emergency_phone": "13900000000",
                },
            )
            self.assertEqual(saved.status_code, 200)
            self.assertEqual(saved.json()["student_id"], "QA-001")
            self.assertEqual(saved.json()["nickname"], "小晴")

            diary = client.post(
                "/api/diary",
                headers=student_headers,
                json={"mood": 4, "note": "今天完成了一个小目标"},
            )
            self.assertEqual(diary.status_code, 200)
            diary_list = client.get("/api/diary", headers=student_headers).json()
            self.assertEqual(diary_list["entries"][0]["mood_label"], "不错")

            questions = client.get(
                "/api/assessments/questions", headers=student_headers
            ).json()
            answers = {item["id"]: (0 if item.get("scale") == "upi" else 1) for item in questions["questions"]}
            if "anxiety_1" in answers:
                answers["anxiety_1"] = 2
            elif "scl_anx_1" in answers:
                answers["scl_anx_1"] = 3
            assessment = client.post(
                "/api/assessments",
                headers=student_headers,
                json={"answers": answers},
            )
            self.assertEqual(assessment.status_code, 200)
            scores = assessment.json()["factor_scores"]
            self.assertTrue("anxiety" in scores or "somatization" in scores)

            article = client.post(
                "/api/admin/articles",
                headers=admin_headers,
                json={
                    "title": "给忙碌大脑的三分钟",
                    "summary": "一个温和的停顿练习",
                    "content": "先把注意力放回呼吸，再看看此刻身体有哪些感受。",
                    "cover_data": "",
                },
            )
            self.assertEqual(article.status_code, 200)
            articles = client.get("/api/articles", headers=student_headers).json()
            self.assertEqual(articles["articles"][0]["title"], "给忙碌大脑的三分钟")

            template = client.post(
                "/api/admin/assessment-templates",
                headers=admin_headers,
                json={
                    "title": "四因子快速自测",
                    "description": "用于功能测试",
                    "questions": [
                        {"id": "a1", "text": "我感到紧张。", "factor": "anxiety"},
                        {"id": "o1", "text": "我会反复确认。", "factor": "obsessive"},
                        {"id": "s1", "text": "压力时身体不舒服。", "factor": "somatization"},
                        {"id": "safe1", "text": "我有伤害自己的念头。", "factor": "safety"},
                    ],
                },
            )
            self.assertEqual(template.status_code, 200)
            active = client.get(
                "/api/assessments/questions", headers=student_headers
            ).json()
            self.assertEqual(active["instrument"], "scl90_upi")
            self.assertTrue(any(item["id"] == "upi_1" for item in active["questions"]))

            crisis = client.post(
                "/api/agent/chat",
                headers=student_headers,
                json={"user_message": "我想自杀", "session_id": "keyword-test"},
            )
            self.assertEqual(crisis.status_code, 200)
            alerts = client.get("/api/admin/alerts", headers=admin_headers).json()
            keyword_alert = next(
                item for item in alerts["alerts"] if item["source"] == "chat_keyword"
            )
            self.assertIn("自杀", keyword_alert["metadata"]["matched_keywords"])

            admin_users = client.get(
                "/api/admin/users", headers=admin_headers
            ).json()
            student_row = next(
                item for item in admin_users["users"] if item["display_name"] == "小晴"
            )
            self.assertIn("anxiety", student_row["factor_scores"])
            self.assertEqual(student_row["risk_level"], "high")
            self.assertTrue(all(item["role"] == "user" for item in admin_users["users"]))
            overview = client.get("/api/admin/overview", headers=admin_headers).json()
            self.assertEqual(overview["high_risk_users"], 1)
            denied = client.post(
                "/api/auth/admin-login",
                json={"username": user_name, "password": "test-password-123"},
            )
            self.assertEqual(denied.status_code, 403)

    def test_crisis_short_circuits_regular_chat(self) -> None:
        with TestClient(app) as client:
            auth = client.post(
                "/api/auth/register",
                json={
                    "username": f"safety_{uuid.uuid4().hex[:10]}",
                    "password": "test-password-123",
                },
            )
            self.assertEqual(auth.status_code, 200)
            token = auth.json()["token"]
            response = client.post(
                "/api/agent/chat",
                json={"user_message": "我不想活了", "session_id": "safety-test"},
                headers={"Authorization": f"Bearer {token}"},
            )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["safety_level"], "crisis")
        self.assertEqual(payload["action"], "comfort")
        self.assertIn("信任", payload["answer"])


if __name__ == "__main__":
    unittest.main()
