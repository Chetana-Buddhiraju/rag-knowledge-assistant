"""Zero-dependency terminal chat, useful for quick smoke-testing without
starting Streamlit or FastAPI.

Usage:
    python cli.py --role sales --profile improved --backend local
"""
from __future__ import annotations

import argparse
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.access_control import get_allowed_departments
from src.config import PROFILES, ROLE_DEPARTMENT_ACCESS, Settings
from src.rag_pipeline import RAGPipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=sorted(ROLE_DEPARTMENT_ACCESS.keys()), default="admin")
    parser.add_argument("--profile", choices=list(PROFILES.keys()), default="improved")
    parser.add_argument("--backend", choices=["local", "github", "azure"], default="local")
    args = parser.parse_args()

    settings = Settings()
    settings.backend = args.backend
    settings.profile_name = args.profile
    pipeline = RAGPipeline(settings)
    allowed = get_allowed_departments(args.role)

    print(f"Role={args.role} (access: {allowed or 'none'}) | backend={args.backend} | profile={args.profile}")
    print("Type 'exit' to quit, 'reset' to clear conversation history.\n")

    history: list[dict[str, str]] = []
    while True:
        try:
            query = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not query:
            continue
        if query.lower() == "exit":
            break
        if query.lower() == "reset":
            history = []
            print("(conversation cleared)")
            continue

        result = pipeline.ask(query, allowed, history)
        if result.ambiguous:
            print("bot> That's ambiguous. Did you mean one of:")
            for opt in result.clarification_options:
                print(f"      - {opt}")
            continue

        answer = result.answer
        print(f"bot> {answer.text}")
        if answer.citations:
            for c in answer.citations:
                print(f"      [{c.index}] {c.title} — {c.section} ({c.source_path})")
        print(f"      (confidence={answer.confidence:.2f}, total_ms={result.timings_ms.get('total_ms', 0):.1f})")

        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": answer.text})


if __name__ == "__main__":
    main()
