import unittest

from ai_service.assessment_service import AssessmentService


class AssessmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = AssessmentService()

    def _baseline(self) -> dict[str, int]:
        answers = {}
        for item in self.service.QUESTIONS:
            answers[item.id] = 0 if item.scale == "upi" else 1
        return answers

    def test_low_risk_complete_answers(self) -> None:
        result = self.service.score(self._baseline())
        self.assertEqual(result["score"], 0)
        self.assertEqual(result["risk_level"], "low")
        self.assertFalse(result["factor_scores"]["anxiety"]["positive"])
        self.assertEqual(result["factor_scores"]["anxiety"]["mean"], 1)
        self.assertEqual(result["classification"]["cluster_id"], 0)

    def test_safety_answer_is_high_priority(self) -> None:
        answers = self._baseline()
        answers["upi_25"] = 1
        result = self.service.score(answers)
        self.assertEqual(result["risk_level"], "high")
        self.assertTrue(result["safety_signal"])
        self.assertEqual(result["classification"]["flags"]["suicidal_ideation"], 1)

    def test_incomplete_answers_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.service.score({"mood": 0})

    def test_scl_factor_cutoff(self) -> None:
        answers = self._baseline()
        for item in self.service.QUESTIONS:
            if item.factor == "anxiety":
                answers[item.id] = 4
        result = self.service.score(answers)
        self.assertGreater(result["factor_scores"]["anxiety"]["mean"], 2.5)
        self.assertTrue(result["factor_scores"]["anxiety"]["positive"])
        self.assertFalse(result["factor_scores"]["obsessive"]["positive"])
        self.assertTrue(result["elevated_factors"])
        self.assertEqual(result["elevated_factors"][0]["key"], "anxiety")
        self.assertTrue(result["elevated_factors"][0]["measures"])

    def test_upi_cutoff_and_cluster(self) -> None:
        answers = self._baseline()
        scored = [item for item in self.service.QUESTIONS if item.scale == "upi" and item.scored]
        for item in scored[:26]:
            answers[item.id] = 1
        result = self.service.score(answers)
        self.assertGreater(result["factor_scores"]["upi"]["raw"], 25)
        self.assertTrue(result["factor_scores"]["upi"]["positive"])
        self.assertEqual(result["classification"]["cluster_id"], 1)

    def test_legacy_template_still_works(self) -> None:
        questions = [
            {"id": "anxiety_1", "text": "紧张", "factor": "anxiety"},
            {"id": "anxiety_2", "text": "担心", "factor": "anxiety"},
            {"id": "obsessive_1", "text": "反复", "factor": "obsessive"},
            {"id": "somatization_1", "text": "头痛", "factor": "somatization"},
            {"id": "safety_1", "text": "伤害自己", "factor": "safety"},
        ]
        answers = {item["id"]: 0 for item in questions}
        for item in questions:
            if item["factor"] == "anxiety":
                answers[item["id"]] = 3
        result = self.service.score(answers, questions)
        self.assertEqual(result["factor_scores"]["anxiety"]["percent"], 100)
        self.assertEqual(result["risk_level"], "high")


if __name__ == "__main__":
    unittest.main()
