"""
Retrieval-Augmented Generation for SerenAI.

Pipeline
  1. Load documents from knowledge/  (.md, .txt, optionally .pdf)
  2. Chunk by section; long sections are packed sentence by sentence into overlapping windows
     (no chunk is ever cut in the middle of a sentence)
  3. Embed with OpenAI `text-embedding-3-small`
  4. Store in a local file index (default) or Pinecone (RAG_BACKEND=pinecone)
  5. At chat time (hybrid search):
       - build a query from the message + recent context (+ the detected topic as a soft hint)
       - dense search (cosine) AND keyword search (BM25), merged with Reciprocal Rank Fusion
       - relevance gate, small topic boost, then MMR re-ranking so the passages are not near-duplicates
       - if embeddings are unavailable (no key, API down) it silently falls back to keyword search

Environment variables (all optional):
  RAG_BACKEND=local|pinecone   RAG_TOP_K=3   RAG_MIN_SCORE=0.25   RAG_LEX_MIN=3.0   RAG_KB_DIR=...
  PINECONE_API_KEY, PINECONE_INDEX=serenai-kb, PINECONE_NAMESPACE=stress-kb
  RAG_FAKE_EMBED=1  -> offline keyword-hash embeddings (used by the demo mode and tests only)
"""
import hashlib
import json
import logging
import math
import os
import re
import threading
import time
import zlib
from collections import OrderedDict, Counter
from pathlib import Path

import numpy as np

log = logging.getLogger("serenai.rag")

ROOT = Path(__file__).resolve().parent.parent
KB_DIR = Path(os.getenv("RAG_KB_DIR", ROOT / "knowledge"))
STORE_DIR = ROOT / "rag_store"
EMBED_MODEL = "text-embedding-3-small"      # 1536 dims
CHUNK_WORDS = 160                            # target chunk size (words)
OVERLAP_WORDS = 30
CHUNKER_VERSION = 2                          # bump when chunking changes, so cached indexes rebuild
TOP_K = int(os.getenv("RAG_TOP_K", "3"))
BACKEND = os.getenv("RAG_BACKEND", "local").lower()
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "serenai-kb")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "stress-kb")
MMR_LAMBDA = 0.7                             # 1.0 = pure relevance, 0.0 = pure diversity
MAX_PER_DOC = 2
RRF_K = 60

# Soft hint: the classifier's topic nudges documents whose file name contains the fragment.
TOPIC_HINTS = {
    "academic": "academic", "work or career": "work", "relationships": "relationship",
    "family": "family", "financial": "financial", "health or wellness": "body",
}
TOPIC_BOOST = 0.25                           # fraction of the best fused score added to a matching document

_lock = threading.RLock()


def fake_mode():
    return os.getenv("RAG_FAKE_EMBED") == "1"


def min_score():
    """Minimum dense cosine similarity for a passage to count as relevant."""
    return float(os.getenv("RAG_MIN_SCORE", "0.08" if fake_mode() else "0.25"))


def lex_min():
    """Minimum BM25 score for a keyword-only match to count as relevant."""
    return float(os.getenv("RAG_LEX_MIN", "3.0"))


def info():
    return {"backend": BACKEND, "embedding": "offline keyword demo" if fake_mode() else EMBED_MODEL,
            "top_k": TOP_K, "search": "hybrid: dense + BM25, MMR re-rank"}


# ---------------------------------------------------------------- documents & chunks
def _pdf_text(path):
    import fitz  # PyMuPDF (optional dependency)
    with fitz.open(path) as doc:
        text = " ".join(p.get_text() for p in doc)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    return re.sub(r"\s+", " ", text).strip()


def load_documents(kb_dir=None):
    docs = []
    for p in sorted(Path(kb_dir or KB_DIR).glob("**/*")):
        ext = p.suffix.lower()
        try:
            if ext in (".md", ".txt"):
                text = p.read_text(encoding="utf-8-sig")
            elif ext == ".pdf":
                text = _pdf_text(p)
            else:
                continue
        except Exception as e:                      # one unreadable file must not take the whole index down
            log.warning("skipping %s: %s", p.name, e)
            continue
        title = p.stem.replace("-", " ").replace("_", " ").title()
        m = re.search(r"(?m)^#\s+(.+)$", text)
        if m:
            title = m.group(1).strip()
        docs.append({"source": p.name, "title": title, "text": text})
    return docs


def _sentences(body):
    body = " ".join(body.split())
    return [s for s in re.split(r"(?<=[.!?])\s+", body) if s]


def _windows(body, size=CHUNK_WORDS, overlap=OVERLAP_WORDS):
    """Pack whole sentences into windows of ~`size` words; each window repeats the tail of the previous one."""
    sents = []
    for s in _sentences(body):                      # a single huge "sentence" (e.g. a PDF) is cut by words
        w = s.split()
        sents.extend([" ".join(w[i:i + size]) for i in range(0, len(w), size)] if len(w) > size else [s])
    if not sents:
        return []
    out, i = [], 0
    while i < len(sents):
        j, n = i, 0
        while j < len(sents) and (n == 0 or n + len(sents[j].split()) <= size):
            n += len(sents[j].split())
            j += 1
        out.append(" ".join(sents[i:j]))
        if j >= len(sents):
            break
        back, k = 0, j                              # step back over up to `overlap` words
        while k - 1 > i and back + len(sents[k - 1].split()) <= overlap:
            k -= 1
            back += len(sents[k].split())
        i = k if k > i else j
    return out


def chunk_document(doc):
    """One chunk per '## ' section; long sections become sentence-aligned overlapping windows."""
    text = re.sub(r"(?m)^#\s+.+$", "", doc["text"], count=1)          # drop the H1 (it is the title)
    parts = re.split(r"(?m)^##\s+", text)
    sections = [("Overview", parts[0])] + [
        (p.split("\n", 1)[0].strip(), p.split("\n", 1)[1] if "\n" in p else "") for p in parts[1:]]
    chunks = []
    for heading, body in sections:
        if len(body.split()) < 8:
            continue
        for win in _windows(body):
            body_text = f"{doc['title']} - {heading}. {win}"
            chunks.append({
                "id": hashlib.sha1(f"{doc['source']}|{body_text}".encode()).hexdigest()[:20],
                "source": doc["source"], "title": doc["title"], "section": heading, "text": body_text,
            })
    return chunks


def build_chunks(docs=None):
    chunks = []
    for d in (docs if docs is not None else load_documents()):
        chunks.extend(chunk_document(d))
    return chunks


# ---------------------------------------------------------------- text analysis (shared by BM25 and the demo embedder)
_STOP = set("""the and for that this with you your have are was were not but can how what when who why from they them then
there their been being into out about just like very really its it's i'm i've don't can't dont cant would could should
some more most much many any all has had does did doing get got feel feeling feels want need
""".split())


def _stem(w):
    """Very light stemmer: breathing/breathe/breathes -> breath, exercises -> exercis, stressed/stress -> stress."""
    w = w.replace("'s", "")
    for suf, rep in (("ies", "y"), ("ing", ""), ("ed", ""), ("es", ""), ("s", "")):
        if len(w) > len(suf) + 3 and w.endswith(suf) and not (suf == "s" and w.endswith("ss")):
            w = w[:-len(suf)] + rep
            break
    return w[:-1] if len(w) > 4 and w.endswith("e") else w


def tokens(text):
    return [_stem(w) for w in re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.lower()) if len(w) > 2 and w not in _STOP]


class _BM25:
    """Small, dependency-free BM25 (Okapi) over the chunk texts."""

    def __init__(self, chunks, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(tokens(c["text"])) for c in chunks]
        self.len = np.array([sum(t.values()) for t in self.tf], dtype=np.float32)
        self.avg = float(self.len.mean()) if len(self.len) else 1.0
        df = Counter(w for t in self.tf for w in t)
        n = len(chunks)
        self.idf = {w: math.log(1 + (n - d + 0.5) / (d + 0.5)) for w, d in df.items()}

    def scores(self, query):
        q = tokens(query)
        out = np.zeros(len(self.tf), dtype=np.float32)
        for w in set(q):
            idf = self.idf.get(w)
            if idf is None:
                continue
            for i, t in enumerate(self.tf):
                f = t.get(w)
                if f:
                    out[i] += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
        return out


# ---------------------------------------------------------------- embeddings
_client = None
_qcache = OrderedDict()          # query text -> embedding, so repeated / retried queries cost no API call
_QCACHE_MAX = 256


def _openai():
    global _client
    if _client is None:
        from openai import OpenAI
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client


def _fake_embed(text, dims=2048):
    v = np.zeros(dims, dtype=np.float32)
    for w in tokens(text):
        v[zlib.crc32(w.encode()) % dims] += 1
    n = np.linalg.norm(v)
    return v / n if n else v


def _normalize(m):
    m = np.asarray(m, dtype=np.float32)
    n = np.linalg.norm(m, axis=-1, keepdims=True)
    return m / np.where(n == 0, 1, n)


def embed_texts(texts, retries=4, batch=64):
    """Return an (n, d) unit-length float32 array. Retries with backoff on API errors."""
    if fake_mode():
        return np.vstack([_fake_embed(t) for t in texts])
    out = []
    for b in range(0, len(texts), batch):
        for attempt in range(retries):
            try:
                res = _openai().embeddings.create(input=texts[b:b + batch], model=EMBED_MODEL)
                out.extend(d.embedding for d in res.data)
                break
            except Exception as e:
                log.warning("embedding error (%s), retrying", e)
                if attempt == retries - 1:
                    raise
                time.sleep(2 ** attempt)
    return _normalize(out)


def embed_query(text):
    with _lock:
        if text in _qcache:
            _qcache.move_to_end(text)
            return _qcache[text]
    v = embed_texts([text])[0]
    with _lock:
        _qcache[text] = v
        while len(_qcache) > _QCACHE_MAX:
            _qcache.popitem(last=False)
    return v


# ---------------------------------------------------------------- local index
def _kb_hash(docs):
    h = hashlib.sha1((("fake" if fake_mode() else EMBED_MODEL) + f"|v{CHUNKER_VERSION}|{CHUNK_WORDS}|{OVERLAP_WORDS}").encode())
    for d in docs:
        h.update(d["source"].encode()); h.update(d["text"].encode())
    return h.hexdigest()


def _save_local(chunks, vectors, kb_hash):
    STORE_DIR.mkdir(exist_ok=True)
    np.save(STORE_DIR / "vectors.npy", vectors)
    (STORE_DIR / "chunks.json").write_text(json.dumps({"hash": kb_hash, "chunks": chunks}), encoding="utf-8")


_local = None      # (vectors | None, chunks)
_bm25 = None


def _lexical_index():
    """Chunks + BM25 are cheap and need no API, so they are built in memory for every backend."""
    global _bm25
    with _lock:
        if _bm25 is None:
            chunks = build_chunks()
            _bm25 = (_BM25(chunks), chunks)
        return _bm25


def _load_local():
    global _local
    with _lock:
        if _local is not None:
            return _local
        docs = load_documents()
        h = _kb_hash(docs)
        try:
            meta = json.loads((STORE_DIR / "chunks.json").read_text(encoding="utf-8"))
            if meta["hash"] == h:
                _local = (np.load(STORE_DIR / "vectors.npy"), meta["chunks"])
                return _local
        except (OSError, ValueError, KeyError):
            pass
        build_index(docs)           # first run, or the knowledge files changed
        return _local


# ---------------------------------------------------------------- Pinecone (optional)
_pc_index = None


def _pinecone_index():
    global _pc_index
    with _lock:
        if _pc_index is None:
            from pinecone import Pinecone, ServerlessSpec
            pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
            if PINECONE_INDEX not in pc.list_indexes().names():
                pc.create_index(name=PINECONE_INDEX, dimension=1536, metric="cosine",
                                spec=ServerlessSpec(cloud="aws", region="us-east-1"))
            while not pc.describe_index(PINECONE_INDEX).status["ready"]:
                time.sleep(1)
            _pc_index = pc.Index(PINECONE_INDEX)
            _ensure_pinecone_filled(_pc_index)
        return _pc_index


def _ensure_pinecone_filled(index):
    """Index is empty or out of date (different chunk count) -> rebuild once, instead of silently returning nothing."""
    try:
        want = len(_lexical_index()[1])
        stats = index.describe_index_stats()
        have = (stats.get("namespaces") or {}).get(PINECONE_NAMESPACE, {}).get("vector_count", 0)
        if have != want:
            log.info("Pinecone namespace has %s vectors, knowledge has %s chunks - rebuilding", have, want)
            build_index(_index=index)
    except Exception as e:
        log.warning("could not verify the Pinecone index: %s", e)


# ---------------------------------------------------------------- public API
def build_index(docs=None, _index=None):
    """(Re)build the index from the knowledge folder. Returns the number of chunks stored."""
    global _local, _bm25
    docs = docs if docs is not None else load_documents()
    chunks = build_chunks(docs)
    if not chunks:
        raise RuntimeError(f"No knowledge found in {KB_DIR}")
    vectors = embed_texts([c["text"] for c in chunks])
    with _lock:
        _bm25 = (_BM25(chunks), chunks)
        if BACKEND == "pinecone":
            index = _index or _pinecone_index()
            try:
                index.delete(delete_all=True, namespace=PINECONE_NAMESPACE)
            except Exception as e:
                if "not found" not in str(e).lower() and "404" not in str(e):
                    raise
            for b in range(0, len(chunks), 100):
                index.upsert(namespace=PINECONE_NAMESPACE, vectors=[
                    {"id": c["id"], "values": vectors[b + j].tolist(),
                     "metadata": {k: c[k] for k in ("text", "title", "section", "source")}}
                    for j, c in enumerate(chunks[b:b + 100])])
        else:
            _save_local(chunks, vectors, _kb_hash(docs))
            _local = (vectors, chunks)
    return len(chunks)


def _dense_search(qvec, pool):
    """[(chunk, cosine, vector | None), ...] best first."""
    if BACKEND == "pinecone":
        res = _pinecone_index().query(vector=qvec.tolist(), top_k=pool, include_metadata=True, include_values=True,
                                      namespace=PINECONE_NAMESPACE)
        by_id = {c["id"]: c for c in _lexical_index()[1]}
        out = []
        for m in res["matches"]:
            md = m["metadata"]
            chunk = by_id.get(m["id"]) or {"id": m["id"], **{k: md[k] for k in ("text", "title", "section", "source")}}
            vec = np.asarray(m["values"], dtype=np.float32) if m.get("values") else None
            out.append((chunk, float(m["score"]), vec))
        return out
    vectors, chunks = _load_local()
    sims = vectors @ qvec
    return [(chunks[i], float(sims[i]), vectors[i]) for i in np.argsort(-sims)[:pool]]


_dense_down_until = 0.0


def _rrf(*rankings):
    """Reciprocal Rank Fusion over lists of ids (best first) -> {id: fused score}."""
    fused = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)
    return fused


def _mmr(cands, top_k):
    """Pick `top_k` of `cands` (dicts with 'rel' and optional 'vec') balancing relevance against redundancy."""
    picked, per_doc, rest = [], {}, list(cands)
    while rest and len(picked) < top_k:
        best, best_val = None, -1e9
        for c in rest:
            if per_doc.get(c["source"], 0) >= MAX_PER_DOC:
                continue
            red = 0.0
            if c.get("vec") is not None:
                red = max((float(c["vec"] @ p["vec"]) for p in picked if p.get("vec") is not None), default=0.0)
            val = MMR_LAMBDA * c["rel"] - (1 - MMR_LAMBDA) * red
            if val > best_val:
                best, best_val = c, val
        if best is None:
            break
        picked.append(best)
        per_doc[best["source"]] = per_doc.get(best["source"], 0) + 1
        rest.remove(best)
    return picked


def retrieve(query, top_k=None, threshold=None, topic=None, dense_query=None):
    """
    Top-k knowledge chunks for `query` (hybrid dense + keyword search).
    Each result has the chunk fields plus `score` (dense cosine similarity; for keyword-only matches a 0-1 squash of BM25).
    """
    top_k = top_k or TOP_K
    threshold = min_score() if threshold is None else threshold
    pool = max(top_k * 4, 12)

    bm25, lex_chunks = _lexical_index()
    lex_scores = bm25.scores(query)
    lex_rank = [int(i) for i in np.argsort(-lex_scores)[:pool] if lex_scores[i] > 0]
    lex_by_id = {lex_chunks[i]["id"]: float(lex_scores[i]) for i in lex_rank}

    global _dense_down_until
    dense = []
    if time.time() >= _dense_down_until:
        try:
            dense = _dense_search(embed_query(dense_query or query), pool)
        except Exception as e:
            _dense_down_until = time.time() + 60        # don't hammer a failing API (and its retry sleeps) every turn
            log.warning("dense search unavailable (%s) - using keyword search for the next minute", e)
    dense_by_id = {c["id"]: (c, cos, vec) for c, cos, vec in dense}

    fused = _rrf([c["id"] for c, _, _ in dense], [lex_chunks[i]["id"] for i in lex_rank])
    chunk_by_id = {c["id"]: c for c in lex_chunks}
    chunk_by_id.update({cid: v[0] for cid, v in dense_by_id.items()})
    if not fused:
        return []
    best = max(fused.values())
    hint = TOPIC_HINTS.get(topic or "")

    cands = []
    for cid, f in fused.items():
        c = chunk_by_id[cid]
        cos = dense_by_id[cid][1] if cid in dense_by_id else None
        lex = lex_by_id.get(cid, 0.0)
        relevant = (cos is not None and cos >= threshold) or lex >= lex_min()
        if not relevant:
            continue
        rel = f / best
        if hint and hint in c["source"].lower():
            rel += TOPIC_BOOST
        score = cos if cos is not None else lex / (lex + 6.0)
        vec = dense_by_id[cid][2] if cid in dense_by_id else None
        cands.append({**c, "rel": rel, "score": float(score), "lexical": round(lex, 2), "vec": vec})
    cands.sort(key=lambda c: -c["rel"])
    picked = _mmr(cands, top_k)
    return [{k: v for k, v in c.items() if k not in ("rel", "vec")} for c in picked]


def build_query(text, history=None):
    """
    The search query for a chat turn. Very short replies ("yes", "it's my exam") are meaningless alone,
    so they are expanded with the user's earlier messages; long messages stand on their own.
    """
    text = (text or "").strip()
    if history and len(text.split()) < 8:
        prev = [m["content"] for m in history if m.get("role") == "user"][-2:]
        return " ".join(prev + [text])
    return text


def retrieve_knowledge(text, history=None, topic=None):
    """
    Retrieval step used by the chat pipeline. Never raises: if retrieval is unavailable
    the conversation simply continues without extra knowledge.
    """
    try:
        query = build_query(text, history)
        dense_q = f"{query} (topic: {topic})" if topic and topic != "general" else query
        return retrieve(query, topic=topic, dense_query=dense_q)
    except Exception as e:
        if not getattr(retrieve_knowledge, "_warned", False):
            retrieve_knowledge._warned = True
            log.warning("RAG retrieval unavailable, continuing without it: %s", e)
        return []
