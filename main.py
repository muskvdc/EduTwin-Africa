from __future__ import annotations

from pathlib import Path
from week1_llm.pipeline import AssistantPipeline


def main() -> None:
    project_root = Path(__file__).resolve().parent
    pipeline = AssistantPipeline(knowledge_dir=project_root / "data" / "knowledge")

    print("=" * 60)
    print("WEEK 1 LLM FOUNDATION")
    print("=" * 60)
    print("Type 'exit' to quit.")

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break
        if not user_input:
            continue

        try:
            answer = pipeline.chat(user_input)
            print(f"\nAI: {answer}")
        except Exception as exc:
            print(f"\nError: {exc}")


if __name__ == "__main__":
    main()
