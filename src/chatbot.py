from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from nltk.stem import PorterStemmer
from nltk.tokenize import wordpunct_tokenize
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class FAQChatbot:
    """FAQ chatbot using NLP preprocessing, TF-IDF and cosine similarity."""

    def __init__(self, faq_path: str | Path) -> None:
        self.faq_path = Path(faq_path)
        self.stemmer = PorterStemmer()
        self.stop_words = set(ENGLISH_STOP_WORDS)
        # Keep question-intent words because they help distinguish FAQ meanings.
        self.intent_words = {"how", "many", "where", "what", "which", "when", "why", "who"}

        self.faqs = self._load_faqs()
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=1,
        )

        training_corpus = [
            self.preprocess(
                " ".join(
                    [
                        faq["question"],
                        *faq.get("keywords", []),
                    ]
                )
            )
            for faq in self.faqs
        ]
        self.faq_matrix = self.vectorizer.fit_transform(training_corpus)

    def _load_faqs(self) -> list[dict[str, Any]]:
        if not self.faq_path.exists():
            raise FileNotFoundError(f"FAQ file not found: {self.faq_path}")

        with self.faq_path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        if not isinstance(data, list) or not data:
            raise ValueError("FAQ file must contain a non-empty JSON list.")

        for index, item in enumerate(data):
            if not isinstance(item, dict):
                raise ValueError(f"FAQ item {index} must be a JSON object.")
            if not item.get("question") or not item.get("answer"):
                raise ValueError(
                    f"FAQ item {index} must contain non-empty 'question' and 'answer'."
                )

        return data

    def preprocess(self, text: str) -> str:
        """
        Clean, tokenize, remove stop words, and stem text.

        wordpunct_tokenize is used because it does not require downloading
        external NLTK corpora such as punkt.
        """
        text = text.lower().strip()
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)
        text = re.sub(r"[^a-z0-9\s'-]", " ", text)

        tokens = wordpunct_tokenize(text)
        cleaned_tokens = []

        for token in tokens:
            token = token.strip("'-")
            if not token:
                continue
            if token in self.stop_words and token not in self.intent_words:
                continue
            if len(token) == 1 and not token.isdigit():
                continue
            cleaned_tokens.append(self.stemmer.stem(token))

        return " ".join(cleaned_tokens)

    def get_response(
        self,
        user_question: str,
        threshold: float = 0.24,
        suggestion_count: int = 3,
    ) -> dict[str, Any]:
        """Return the best FAQ answer and similarity information."""
        if not isinstance(user_question, str) or not user_question.strip():
            return {
                "answer": "Please type a question so I can help you.",
                "confidence": 0.0,
                "matched_question": None,
                "is_match": False,
                "suggestions": self._default_suggestions(suggestion_count),
            }

        normalized = user_question.strip().lower()

        greetings = {
            "hi",
            "hello",
            "hey",
            "good morning",
            "good afternoon",
            "good evening",
        }
        if normalized in greetings:
            return {
                "answer": (
                    "Hello! Ask me anything about the CodeAlpha AI internship "
                    "tasks, GitHub submission, certificate requirements, or project rules."
                ),
                "confidence": 1.0,
                "matched_question": "Greeting",
                "is_match": True,
                "suggestions": self._default_suggestions(suggestion_count),
            }

        processed_question = self.preprocess(user_question)
        if not processed_question:
            return {
                "answer": (
                    "I could not identify enough meaningful words in that message. "
                    "Please ask a more specific internship-related question."
                ),
                "confidence": 0.0,
                "matched_question": None,
                "is_match": False,
                "suggestions": self._default_suggestions(suggestion_count),
            }

        question_vector = self.vectorizer.transform([processed_question])
        similarity_scores = cosine_similarity(
            question_vector, self.faq_matrix
        ).flatten()

        ranked_indices = np.argsort(similarity_scores)[::-1]
        best_index = int(ranked_indices[0])
        best_score = float(similarity_scores[best_index])
        best_faq = self.faqs[best_index]

        suggestions = [
            self.faqs[int(index)]["question"]
            for index in ranked_indices[:suggestion_count]
        ]

        if best_score < threshold:
            return {
                "answer": (
                    "I’m not confident enough to give a definite answer to that. "
                    "Try rephrasing your question or choose one of the suggested FAQs."
                ),
                "confidence": best_score,
                "matched_question": best_faq["question"],
                "is_match": False,
                "suggestions": suggestions,
            }

        return {
            "answer": best_faq["answer"],
            "confidence": best_score,
            "matched_question": best_faq["question"],
            "is_match": True,
            "suggestions": suggestions,
        }

    def _default_suggestions(self, count: int) -> list[str]:
        return [faq["question"] for faq in self.faqs[:count]]
