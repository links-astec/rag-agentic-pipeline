#!/usr/bin/env python3
"""
main.py — CLI for RAG Iteration 2 (Agentic RAG via LangGraph).

Commands:
    python main.py ingest <source>         Load & index a file, folder, or URL
    python main.py query  "<question>"     One-shot Q&A (verbose agent trace)
    python main.py chat                    Interactive chat loop
    python main.py steps  "<question>"     Show each graph node as it executes
    python main.py clear                   Drop and recreate the vector collection
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from pipeline import RAGPipeline


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1].lower()
    rag = RAGPipeline()

    if command == "ingest":
        if len(sys.argv) < 3:
            print("Usage: python main.py ingest <source>")
            sys.exit(1)
        rag.ingest(sys.argv[2])

    elif command == "query":
        if len(sys.argv) < 3:
            print('Usage: python main.py query "<question>"')
            sys.exit(1)
        question = " ".join(sys.argv[2:])
        answer = rag.query(question, verbose=True)
        print(f"\nAnswer:\n{answer}\n")

    elif command == "steps":
        if len(sys.argv) < 3:
            print('Usage: python main.py steps "<question>"')
            sys.exit(1)
        question = " ".join(sys.argv[2:])
        print(f"\nRunning agent graph for: '{question}'\n")
        for node_name, state in rag.stream_steps(question):
            print(f"  ✓ [{node_name}]")
            if node_name == "rewrite_query":
                print(f"    Rewritten: {state.get('rewritten_query', '—')}")
            elif node_name == "retrieve":
                print(f"    Retrieved: {len(state.get('documents', []))} chunks")
            elif node_name == "grade_documents":
                print(f"    Kept:      {len(state.get('documents', []))} relevant chunks")
            elif node_name == "generate":
                answer = state.get("generation", "")
                print(f"    Answer:    {answer[:120]}{'...' if len(answer) > 120 else ''}")
        print()

    elif command == "chat":
        print("RAG Agent Chat — type 'quit' to exit.\n")
        while True:
            try:
                question = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break
            if not question or question.lower() in ("quit", "exit"):
                print("Goodbye!")
                break
            answer = rag.query(question)
            print(f"\nAssistant: {answer}\n")

    elif command == "clear":
        rag.clear()
        print("Vector store cleared.")

    else:
        print(f"Unknown command: '{command}'")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
