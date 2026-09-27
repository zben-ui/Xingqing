import unittest

from ai_service.memory_service import MemoryService


class MemoryServiceTests(unittest.TestCase):
    def test_keeps_max_turns_and_clears(self) -> None:
        memory = MemoryService(max_turns=2)
        for index in range(3):
            memory.add_turn("demo", f"u{index}", f"a{index}")
        messages = memory.get_messages("demo")
        self.assertEqual(len(messages), 4)
        self.assertEqual(messages[0]["content"], "u1")
        self.assertTrue(memory.clear("demo"))
        self.assertEqual(memory.get_messages("demo"), [])


if __name__ == "__main__":
    unittest.main()

