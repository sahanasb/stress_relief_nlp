"""
Offline tests for the name understanding and RAG code (no API keys, no models):

    python tests/test_core.py
"""
import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")     # Arabic etc. in the output on Windows consoles
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ["RAG_FAKE_EMBED"] = "1"          # offline keyword-hash embeddings

from name_utils import (extract_name, format_name, is_no, is_yes, name_candidates,  # noqa: E402
                        spell_out, spelled_name)
from rag import store  # noqa: E402

failed = []


def check(label, got, want):
    ok = got == want
    print(("ok   " if ok else "FAIL ") + f"{label}: {got!r}" + ("" if ok else f"  (wanted {want!r})"))
    if not ok:
        failed.append(label)


# ------------------------------------------------------------------ names
for text, want in [
    ("my name is saja", "Saja"),
    ("Hi, my name's Saja and I'm a bit stressed", "Saja"),
    ("you can call me Mo", "Mo"),
    ("everyone calls me Jay", "Jay"),
    ("I'm Priya", "Priya"),
    ("I'm stressed", None),
    ("I am feeling tired", None),
    ("Saja", "Saja"),
    ("uh, Saja", "Saja"),
    ("o'brien", "O'Brien"),
    ("mary-jane", "Mary-Jane"),
    ("it's Mary Anne", "Mary Anne"),
    ("S A J A", "Saja"),
    ("s-a-j-a", "Saja"),
    ("Saja, that's S-A-J-A", "Saja"),
    ("my name is Sasha, that's S. A. S. H. A.", "Sasha"),
    ("اسمي سجى", "سجى"),
    ("", None),
    ("yeah that's right", None),
]:
    check(f"extract_name({text!r})", extract_name(text), want)

check("spelled_name ignores 'I a m'", spelled_name("I a m tired"), None)
check("candidates", name_candidates(["my name is Sasha", "my name is Saja", "my name is sasha"]), ["Sasha", "Saja"])
check("format DeShawn", format_name("DeShawn"), "DeShawn")
check("spell_out", spell_out("Saja"), "S, A, J, A")
for t in ("yes", "Yeah that's right", "yep!", "correct", "that's right"):
    check(f"is_yes({t!r})", is_yes(t), True)
for t in ("no", "nope", "No, it's Sasha", "that's wrong", "not quite"):
    check(f"is_no({t!r})", is_no(t), True)
check("is_yes('no')", is_yes("no"), False)

# ------------------------------------------------------------------ RAG
docs = store.load_documents()
chunks = store.build_chunks(docs)
print(f"\n{len(docs)} documents -> {len(chunks)} chunks")
check("has chunks", len(chunks) > 15, True)
check("unique chunk ids", len({c["id"] for c in chunks}), len(chunks))
longest = max(len(c["text"].split()) for c in chunks)
check("no oversized chunk", longest <= store.CHUNK_WORDS + 25, True)
check("chunks end on a sentence", all(c["text"].rstrip()[-1] in ".!?)\"'" for c in chunks), True)

long_body = " ".join(f"This is sentence number {i} about calm breathing and slow relaxed steady exhale." for i in range(60))
wins = store._windows(long_body)
check("long text becomes several windows", len(wins) > 3, True)
check("windows are sentence aligned", all(w.endswith(".") for w in wins), True)
check("windows overlap", wins[0].split(". ")[-1] in wins[1], True)

store.STORE_DIR = ROOT / "rag_store_test"
store.build_index()


def top(q, **kw):
    return [h["source"] for h in store.retrieve(q, **kw)]


for q, want in [
    ("I can't sleep, my mind won't switch off at night", "sleep-and-winding-down.md"),
    ("my exam is tomorrow and I'm panicking", "exam-and-academic-stress.md"),
    ("bills and rent are piling up, money is tight", "financial-stress.md"),
    ("my heart is racing, how can I breathe slower", "breathing-exercises.md"),
]:
    got = top(q)
    check(f"top hit for {q!r}", got[0] if got else None, want)

check("speaking nerves query finds the speaking doc",
      "public-speaking-and-performance-nerves.md" in top("I have to give a presentation and my hands shake"), True)
check("irrelevant query returns nothing", top("what is the capital of france and who won the football"), [])
check("per-document cap", max(top("stress anxiety worry breathing sleep exam work").count(s)
                              for s in set(top("stress anxiety worry breathing sleep exam work")) or {None}) <= store.MAX_PER_DOC, True)

hist = [{"role": "user", "content": "I have an exam tomorrow"}, {"role": "assistant", "content": "That sounds stressful."}]
check("short reply uses history", store.build_query("yes please", hist), "I have an exam tomorrow yes please")
check("long message ignores history", store.build_query("I have been feeling very overwhelmed with everything lately", hist),
      "I have been feeling very overwhelmed with everything lately")
got = [h["source"] for h in store.retrieve_knowledge("it's the tests", hist, "academic")]
check("history + topic finds exam doc", "exam-and-academic-stress.md" in got, True)

# retrieval must survive a broken embedding backend (keyword fallback)
os.environ["RAG_FAKE_EMBED"] = "0"
store._dense_down_until = 0.0
os.environ["OPENAI_API_KEY"] = ""
orig = store.embed_texts
store.embed_texts = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("api down"))
store._qcache.clear()
got = [h["source"] for h in store.retrieve_knowledge("I can't sleep at night, my mind keeps racing", [], "general")]
check("keyword fallback works with the API down", "sleep-and-winding-down.md" in got, True)
store.embed_texts = orig

import shutil  # noqa: E402
shutil.rmtree(store.STORE_DIR, ignore_errors=True)
print("\nFAILED: " + ", ".join(failed) if failed else "\nall passed")
sys.exit(1 if failed else 0)
