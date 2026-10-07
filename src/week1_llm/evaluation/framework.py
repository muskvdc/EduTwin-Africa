from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Any


@dataclass(frozen=True)
class TestCase:
    input: str
    expected: str
    criteria: str = "contains"
    category: str = "functional"

    def evaluate(self, actual: str) -> dict[str, Any]:
        actual_text = actual or ""
        if self.criteria == "exact_match":
            passed = actual_text.strip() == self.expected.strip()
        elif self.criteria == "contains":
            passed = self.expected.lower() in actual_text.lower()
        elif self.criteria == "keyword":
            passed = all(word.lower() in actual_text.lower() for word in self.expected.split())
        else:
            raise ValueError(f"Unknown criteria: {self.criteria}")
        return {
            "passed": passed,
            "input": self.input,
            "expected": self.expected,
            "actual": actual_text,
            "criteria": self.criteria,
            "category": self.category,
        }


class EvaluationMetrics:
    @staticmethod
    def accuracy(results: list[dict[str, Any]]) -> float:
        return (
            sum(1 for result in results if result["passed"]) / len(results)
            if results else 0.0
        )


class AITestSuite:
    def __init__(self, name: str):
        self.name = name
        self.test_cases: list[TestCase] = []
        self.results: list[dict[str, Any]] = []

    def add_test(self, test_case: TestCase) -> None:
        self.test_cases.append(test_case)

    def run_tests(self, ai_function: Callable[[str], str]) -> dict[str, Any]:
        self.results = [test.evaluate(ai_function(test.input)) for test in self.test_cases]
        return {
            "total_tests": len(self.results),
            "passed": sum(1 for result in self.results if result["passed"]),
            "failed": sum(1 for result in self.results if not result["passed"]),
            "accuracy": EvaluationMetrics.accuracy(self.results),
        }

    def report(self) -> dict[str, Any]:
        by_category: dict[str, dict[str, int]] = {}
        for result in self.results:
            category = result["category"]
            bucket = by_category.setdefault(category, {"passed": 0, "failed": 0})
            bucket["passed" if result["passed"] else "failed"] += 1
        return {
            "suite": self.name,
            "accuracy": EvaluationMetrics.accuracy(self.results),
            "categories": by_category,
        }
