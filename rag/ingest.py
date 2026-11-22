"""
Build (or rebuild) the knowledge index, and optionally try queries.

    python -m rag.ingest
    python -m rag.ingest --query "I can't sleep before my exam"
    python -m rag.ingest --chunks          # list every chunk that will be embedded
"""
import argparse

from dotenv import load_dotenv

load_dotenv()

from . import store  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", action="append", help="run a test retrieval after indexing (repeatable)")
    ap.add_argument("--chunks", action="store_true", help="print the chunks and exit (no embedding calls)")
    a = ap.parse_args()

    if a.chunks:
        for c in store.build_chunks():
            print(f"[{c['source']}] {c['section']}  ({len(c['text'].split())} words)")
        return

    n = store.build_index()
    print(f"Indexed {n} chunks from {len(store.load_documents())} documents  [{store.info()}]")
    for q in a.query or []:
        print(f"\n> {q}")
        hits = store.retrieve(q)
        if not hits:
            print("  No passage above the relevance threshold.")
        for h in hits:
            print(f"  score={h['score']:.3f} kw={h['lexical']:<5} {h['title']} / {h['section']}")


if __name__ == "__main__":
    main()
