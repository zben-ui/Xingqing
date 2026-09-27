import asyncio
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ai_service.agent_service import AgentService
from ai_service.config import settings
from ai_service.content_service import ContentService
from ai_service.memory_service import MemoryService
from ai_service.ollama_client import OllamaError
from ai_service.rag_service import RagService
from ai_service.report_service import ReportService
from ai_service.safety_service import SafetyService
from ai_service.storage_service import StorageService
from ai_service.tts_service import TTSService, TTSError


class _LocalModel:
    async def embed(self, text):
        raise OllamaError("offline test")

    async def chat(self, messages, **kwargs):
        self.messages = messages
        return "现有资料提示压力，需要进一步了解。"


class AgentIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_chat_and_report_share_private_safe_corpus(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            model = _LocalModel()
            rag = RagService(model, replace(settings, knowledge_base_dir=folder / "kb",
                                           vector_store_path=folder / "vectors.json"))
            self.assertEqual(len(rag.corpus.records), 46)
            self.assertTrue(any(item.file.startswith("Agent/") for item in rag.chunks))
            self.assertFalse(any("students_eval" in item.file for item in rag.chunks))
            cached = rag.chunks[0]
            from dataclasses import asdict
            cached.embedding = [1., 0.]
            rag.settings.vector_store_path.write_text(json.dumps({"mode": "embedding", "chunks": [asdict(cached)]}), encoding="utf-8")
            rag.load_store()
            self.assertTrue(all(item.embedding is None for item in rag.chunks))
            first = folder / "kb/users/7/secret.txt"
            first.parent.mkdir(parents=True)
            first.write_text("焦虑 PRIVATE_ONLY_USER_7", encoding="utf-8")
            rag.chunks = rag.read_documents()
            sources = await rag.search("焦虑", user_id=8)
            self.assertFalse(any(item["file"].startswith("users/7/") for item in sources))
            agent = AgentService(model, rag, MemoryService(), SafetyService())
            result = await agent.chat("我最近焦虑紧张，怎么照顾自己", "integration", user_id=8)
            self.assertTrue(result["sources"])
            self.assertIn("不得写成诊断", model.messages[0]["content"])
            storage = StorageService(replace(settings, database_path=folder / "test.db"))
            user, _ = storage.register("integration", "test-password-123")
            storage.save_assessment(user["id"], 2, "high", {"safety": 2}, {}, "建议现实支持", safety_signal=True)
            report = await ReportService(model, storage, rag).generate(user["id"])
            self.assertTrue(report["sources"])
            self.assertTrue(any(item["file"].startswith("Agent/") for item in report["sources"]))
            self.assertIn("12356", report["content"])
            self.assertTrue(report["agent"]["cluster_model_enabled"])
            self.assertNotIn("聚类结果反映", rag.corpus.rules_for_report(False))
            self.assertTrue(storage.latest_assessment(user["id"])["safety_signal"])

    async def test_four_supplied_articles_are_text_not_executable_html(self):
        articles = ContentService(settings.agent_data_dir.parent / "image").articles()
        self.assertEqual(len(articles), 4)
        self.assertEqual(articles[0]["title"], "《晨光里的褶皱》")
        self.assertNotIn("<script", articles[0]["content"])
        self.assertTrue(articles[0]["cover_url"].endswith("1.png"))

    async def test_tts_stream_output_cache_and_errors(self):
        class Socket:
            async def __aenter__(self): return self
            async def __aexit__(self, *args): pass
            async def send(self, payload):
                task = json.loads(payload)
                assert task["payload"]["model"] == "sambert-zhiwei-v1"
                assert task["payload"]["input"]["text"] == "你好"
            def __aiter__(self):
                async def events():
                    yield b"ID3-test-audio"
                    yield json.dumps({"header": {"event": "task-finished"}})
                return events()
        tts = TTSService(replace(settings, enable_tts=True))
        with patch.object(tts, "_key", return_value="test-only"), patch("ai_service.tts_service.connect", return_value=Socket()) as connect:
            self.assertEqual(await tts.synthesize("你好"), b"ID3-test-audio")
            await tts.synthesize("你好")
            self.assertEqual(connect.call_count, 1)
        with patch.object(tts, "_key", return_value=""):
            with self.assertRaises(TTSError): await tts.synthesize("未配置")
