"""
Minimal local RAG over a single large document (e.g. the VW PEP document, ~55k tokens)
using Ollama: qwen3:4b for answering, bge-m3 for embeddings (multilingual: DE/EN).

Setup:
    ollama pull qwen3:4b
    ollama pull bge-m3
    pip install ollama numpy pypdf

Usage:
    python rag.py path/to/pep.pdf            # interactive Q&A
    python rag.py path/to/pep.pdf --rebuild  # force re-indexing
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import ollama

# ----------------------------- config ---------------------------------------
LLM_MODEL = "qwen3-4b:latest"
EMBED_MODEL = "bge-m3"       # alternative: "nomic-embed-text" (English-centric, faster)
CHARS_PER_TOKEN = 3.5        # German text with the Qwen / bge-m3 tokenizers (len/4 would underestimate)
CHUNK_TOKENS = 1800          # max chunk size: chunks end up at ~1500-1800 tokens (~35 chunks for 55k tokens)
OVERLAP_TOKENS = 225         # ~12.5% of the chunk size: the next chunk repeats the last ~225 tokens
OVERLAP_MAX_TOKENS = 300     # ...never more than this
TOP_K = 3                    # chunks sent to the LLM (4 x ~1700 = ~7k tokens)
NUM_CTX = 12288              # Ollama default is small (2-4k) -> raise it explicitly
EMBED_NUM_CTX = 8192         # otherwise Ollama may silently truncate big chunks before embedding
THINKING = False             # True = qwen3 reasoning mode (slower, sometimes better)
CACHE_DIR = Path(".rag_cache")

SYSTEM_PROMPT = (
    "You are an assistant that answers questions about the provided document "
    "(Volkswagen Produktentstehungsprozess / PEP). Use ONLY the context excerpts. "
    "If the answer is not in the context, say you could not find it in the document. "
    "Answer in the same language as the question. Cite the excerpt numbers you used, "
    "like [2] or [1][4]."
)


# ----------------------------- loading --------------------------------------
def load_text(path: Path) -> list[tuple[int, str]]:
    """Return a list of (page_number, text). Non-PDF files are treated as one page."""
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return [(i + 1, (p.extract_text() or "")) for i, p in enumerate(reader.pages)]
    return [(1, path.read_text(encoding="utf-8", errors="ignore"))]


def approx_tokens(text: str) -> int:
    # Rough estimate, good enough for chunk sizing (~3.5 chars per token for German)
    return max(1, int(len(text) / CHARS_PER_TOKEN))


# ----------------------------- chunking -------------------------------------
def chunk_pages(pages: list[tuple[int, str]]) -> list[dict]:
    """Sentence/paragraph-aware chunking with overlap. Keeps page numbers."""
    units: list[tuple[int, str]] = []  # (page, sentence-ish unit)
    for page_no, text in pages:
        text = re.sub(r"[ \t]+", " ", text)
        for para in re.split(r"\n\s*\n", text):
            para = para.strip()
            if not para:
                continue
            for s in re.split(r"(?<=[.!?;:])\s+|\n", para):
                s = s.strip()
                if s:
                    units.append((page_no, s))

    chunks, cur, cur_tok = [], [], 0
    for page_no, s in units:
        t = approx_tokens(s)
        if cur and cur_tok + t > CHUNK_TOKENS:
            chunks.append(_make_chunk(cur))
            # build overlap from the tail of the previous chunk
            tail, tail_tok = [], 0
            for u in reversed(cur):
                t_u = approx_tokens(u[1])
                if tail and tail_tok + t_u > OVERLAP_MAX_TOKENS:
                    break
                tail_tok += t_u
                tail.insert(0, u)
                if tail_tok >= OVERLAP_TOKENS:
                    break
            cur, cur_tok = tail, sum(approx_tokens(u[1]) for u in tail)
        cur.append((page_no, s))
        cur_tok += t
    if cur:
        chunks.append(_make_chunk(cur))
    return chunks


def _make_chunk(units: list[tuple[int, str]]) -> dict:
    pages = sorted({p for p, _ in units})
    return {
        "text": " ".join(s for _, s in units),
        "page_start": pages[0],
        "page_end": pages[-1],
    }


# ----------------------------- embeddings -----------------------------------
def embed(texts: list[str], batch: int = 4) -> np.ndarray:
    vecs = []
    for i in range(0, len(texts), batch):
        resp = ollama.embed(model=EMBED_MODEL, input=texts[i : i + batch], options={"num_ctx": EMBED_NUM_CTX})
        vecs.extend(resp["embeddings"])
    arr = np.array(vecs, dtype=np.float32)
    arr /= np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12  # normalize -> dot = cosine
    return arr


def build_or_load_index(path: Path, rebuild: bool = False):
    CACHE_DIR.mkdir(exist_ok=True)
    key = hashlib.md5(
        (path.name + str(path.stat().st_mtime) + EMBED_MODEL
         + str(CHUNK_TOKENS) + str(OVERLAP_TOKENS)).encode()
    ).hexdigest()[:12]
    chunks_f, vecs_f = CACHE_DIR / f"{key}.json", CACHE_DIR / f"{key}.npy"

    if not rebuild and chunks_f.exists() and vecs_f.exists():
        print("Loaded cached index.")
        return json.loads(chunks_f.read_text(encoding="utf-8")), np.load(vecs_f)

    print("Reading and chunking document ...")
    chunks = chunk_pages(load_text(path))
    print(f"{len(chunks)} chunks (~{sum(approx_tokens(c['text']) for c in chunks)} tokens). Embedding ...")
    vecs = embed([c["text"] for c in chunks])
    chunks_f.write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
    np.save(vecs_f, vecs)
    return chunks, vecs


# ----------------------------- retrieval ------------------------------------
def keyword_score(query: str, chunks: list[dict]) -> np.ndarray:
    """Cheap lexical score so exact terms (e.g. 'Meilenstein', 'SOP', 'BF', 'VFF') still match."""
    terms = {t for t in re.findall(r"\w+", query.lower()) if len(t) > 2}
    if not terms:
        return np.zeros(len(chunks), dtype=np.float32)
    scores = []
    for c in chunks:
        words = set(re.findall(r"\w+", c["text"].lower()))
        scores.append(len(terms & words) / len(terms))
    return np.array(scores, dtype=np.float32)


def retrieve(query: str, chunks, vecs, k: int = TOP_K, alpha: float = 0.8):
    q = embed([query])[0]
    sem = vecs @ q
    hybrid = alpha * sem + (1 - alpha) * keyword_score(query, chunks)
    idx = np.argsort(-hybrid)[:k]
    return [(int(i), float(hybrid[i])) for i in idx]


# ----------------------------- generation -----------------------------------
def answer(query: str, chunks, vecs):
    hits = retrieve(query, chunks, vecs)
    context = "\n\n".join(
        f"[{n}] (pages {chunks[i]['page_start']}-{chunks[i]['page_end']})\n{chunks[i]['text']}"
        for n, (i, _) in enumerate(hits, 1)
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
    ]
    stream = ollama.chat(
        model=LLM_MODEL,
        messages=messages,
        stream=True,
        think=THINKING,
        options={"num_ctx": NUM_CTX, "temperature": 0.2},
    )
    print("\nAnswer:\n")
    for part in stream:
        print(part["message"]["content"], end="", flush=True)
    print("\n\nSources:")
    for n, (i, score) in enumerate(hits, 1):
        c = chunks[i]
        print(f"  [{n}] pages {c['page_start']}-{c['page_end']}  (score {score:.2f})")


# ----------------------------- main -----------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("document", type=Path)
    ap.add_argument("--rebuild", action="store_true", help="re-chunk and re-embed")
    args = ap.parse_args()
    if not args.document.exists():
        sys.exit(f"File not found: {args.document}")

    chunks, vecs = build_or_load_index(args.document, args.rebuild)
    print("\nAsk questions about the document (empty line or Ctrl+C to quit).")
    while True:
        try:
            q = input("\n> ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not q:
            break
        answer(q, chunks, vecs)


if __name__ == "__main__":
    main()