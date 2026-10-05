import unittest
from pathlib import Path

from src.chatbot import FAQChatbot


BASE_DIR = Path(__file__).resolve().parents[1]


class FAQChatbotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bot = FAQChatbot(BASE_DIR / "data" / "faqs.json")

    def test_preprocess_returns_text(self):
        result = self.bot.preprocess("How many tasks do I need to complete?")
        self.assertIsInstance(result, str)
        self.assertTrue(result)

    def test_task_count_question(self):
        result = self.bot.get_response("How many projects must I finish?")
        self.assertTrue(result["is_match"])
        self.assertIn("2 or 3", result["answer"])

    def test_github_question(self):
        result = self.bot.get_response("Where do I upload my source code?")
        self.assertTrue(result["is_match"])
        self.assertIn("GitHub", result["answer"])

    def test_task_four_question(self):
        result = self.bot.get_response("Can I use YOLO for task four?")
        self.assertTrue(result["is_match"])
        self.assertIn("YOLO", result["answer"])

    def test_unknown_question_falls_back(self):
        result = self.bot.get_response(
            "What is the distance between Earth and Neptune?",
            threshold=0.30,
        )
        self.assertFalse(result["is_match"])

    def test_empty_question(self):
        result = self.bot.get_response("   ")
        self.assertFalse(result["is_match"])
        self.assertEqual(result["confidence"], 0.0)


if __name__ == "__main__":
    unittest.main()
