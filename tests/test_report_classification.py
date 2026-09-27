from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from ai_service import main
from ai_service.assessment_service import AssessmentService
from ai_service.config import settings
from ai_service.ollama_client import OllamaError
from ai_service.rag_service import RagService
from ai_service.report_service import ReportService
from ai_service.screening_classifier import ScreeningClassifier
from ai_service.storage_service import StorageService


class Model:
    def __init__(self, offline=False): self.offline = offline
    async def embed(self, text): raise OllamaError("test keyword fallback")
    async def chat(self, messages, **kwargs):
        self.messages, self.kwargs = messages, kwargs
        if self.offline: raise OllamaError("test offline")
        return "## 状态概览\n分类仅供支持参考。\n## 建议的下一步\n联系现实中的支持。"


class ReportClassificationTests(unittest.IsolatedAsyncioTestCase):
    async def test_only_compact_classification_reaches_model_and_local_measures_survive(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            isolated_settings = replace(settings, database_path=folder / "test.db",
                knowledge_base_dir=folder / "kb", vector_store_path=folder / "vectors.json")
            storage = StorageService(isolated_settings)
            user, _ = storage.register("report-test", "test-password-123")
            service = AssessmentService()
            answers = {q.id: 0 if q.scale == "upi" else 4 for q in service.QUESTIONS}
            result = service.score(answers)
            result["elevated_factors"][0]["measures"].append("LOCAL_MEASURES_SENTINEL")
            storage.save_assessment(user["id"], result["score"], result["risk_level"], answers,
                {"factors": result["factor_scores"], "classification": result["classification"],
                 "elevated_factors": result["elevated_factors"]}, result["summary"])
            storage.save_message(user["id"], "test", "user", "最近压力大")
            model = Model()
            report = await ReportService(model, storage, RagService(model, isolated_settings)).generate(user["id"])
            prompt = model.messages[-1]["content"]
            compact = json.loads(prompt.split("本地分类结果：", 1)[1].split("\n", 1)[0])
            self.assertEqual(set(compact), {"cluster_id", "cluster_name", "algorithm", "attention_level", "safety_signal"})
            self.assertEqual(compact["cluster_id"], 1)
            self.assertNotIn("LOCAL_MEASURES_SENTINEL", prompt)
            for forbidden in ("factor_scores", "feature_vector", "positive_flag_count", "scl90_anxiety_flag", "score_text"):
                self.assertNotIn(forbidden, prompt)
            self.assertIn("LOCAL_MEASURES_SENTINEL", report["content"])
            self.assertEqual(model.kwargs["max_tokens"], 640)
            self.assertEqual(report["performance"]["factor_input_mode"], "local-classification-only")
            with patch.object(main, "storage", storage):
                chat_context = main._chat_profile(user["id"])["assessment_context"]
            self.assertNotIn("factor_scores", chat_context)
            self.assertNotIn("scl90_anxiety_flag", chat_context)
            offline = Model(offline=True)
            fallback = await ReportService(offline, storage, RagService(offline, isolated_settings)).generate(user["id"])
            self.assertFalse(fallback["model_available"])
            self.assertIn("LOCAL_MEASURES_SENTINEL", fallback["content"])

    def test_short_assessments_not_mislabeled_as_standard_screening(self):
        result = ScreeningClassifier().model_context({"factor_scores": {"anxiety": {"percent": 80}},
                                                     "risk_level": "high"})
        self.assertIsNone(result["cluster_id"])
        self.assertEqual(result["algorithm"], "custom-assessment-priority-v1")


class ImmediateAlertTests(unittest.TestCase):
    def test_alert_is_visible_before_reply_and_keyword_priority_beats_count(self):
        with tempfile.TemporaryDirectory() as temporary:
            storage = StorageService(database_path=Path(temporary) / "test.db")
            admin, _ = storage.register("test-admin", "test-password-123")
            student, token = storage.register("test-student", "test-password-123")
            other, _ = storage.register("test-other", "test-password-123")
            for _ in range(5): storage.create_alert(other["id"], "assessment", "medium", "测试")
            async def interrupted(*args, **kwargs):
                self.assertEqual(storage.admin_overview()["high_risk_users"], 1)
                self.assertEqual(storage.list_users()[0]["id"], student["id"])
                self.assertEqual(storage.list_users()[0]["attention_priority"], 3)
                self.assertEqual(len(storage.list_messages(student["id"])), 1)
                raise RuntimeError("simulate disconnect before first token")
                yield {}
            with patch.object(main, "storage", storage), patch.object(main.agent, "chat_stream", interrupted):
                with TestClient(main.app, raise_server_exceptions=False) as client:
                    client.post("/api/agent/chat/stream", headers={"Authorization": f"Bearer {token}"},
                                json={"session_id": "aborted", "user_message": "这是测试，自 杀等词需要关注"})
                    self.assertEqual(client.get("/api/admin/users", headers={"Authorization": f"Bearer {token}"}).status_code, 403)
            alerts = storage.user_detail(student["id"])["alerts"]
            self.assertEqual(len(alerts), 1)
            self.assertEqual(alerts[0]["metadata"]["attention_priority"], "urgent")
            storage.acknowledge_alert(alerts[0]["id"])
            row = next(item for item in storage.list_users() if item["id"] == student["id"])
            self.assertEqual(row["attention_priority"], 0)
            self.assertEqual(row["risk_level"], "unknown")
