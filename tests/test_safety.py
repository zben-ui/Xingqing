import unittest

from ai_service.safety_service import SafetyService


class SafetyServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = SafetyService()

    def test_normal(self) -> None:
        self.assertEqual(self.service.detect("今天想复习一下").level, "normal")

    def test_emotional_support(self) -> None:
        self.assertEqual(self.service.detect("我最近压力大，很焦虑").level, "emotional_support")

    def test_crisis(self) -> None:
        self.assertEqual(self.service.detect("我不想活了").level, "crisis")

    def test_spaced_keyword_is_a_manual_review_signal(self) -> None:
        result = self.service.detect("聊天里提到自\u200b 杀这个词")
        self.assertEqual(result.level, "crisis")
        self.assertIn("自杀", result.matched_keywords)


if __name__ == "__main__":
    unittest.main()
