"""
rag_vw_pep_flask.py - your plain, fast RAG as an engine for chat_flask.py, now with a vector database
(ChromaDB) as a persistent "personal data pool".

What it does
  indexing : pypdf text -> sentence/paragraph chunks (~1800 tokens, ~225 overlap) -> bge-m3 embeddings
             -> stored in ChromaDB (.vector_db/) with document name and page numbers as metadata
  search   : one document  -> exactly the scoring of rag_vw_pep.py (0.8 * cosine + 0.2 * keyword overlap)
             all documents -> the "pool": the vector index finds candidates across every file, then the
                              same 0.8 / 0.2 scoring picks the top chunks (answers name their document)
  answer   : local Ollama LLM, conversation memory (short follow-ups are rewritten into a standalone query)

Data pool
  - add files any time (add_document), remove them (delete_document); everything survives restarts
  - old indexes of the previous version (.rag_cache/) are imported automatically, without re-embedding
  - ChromaDB telemetry is switched off; nothing leaves your computer

Setup:  ollama pull bge-m3   (+ your chat model, see `ollama list`)
        pip install ollama numpy pypdf chromadb
"""

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Iterator

import numpy as np
import ollama

# ----------------------------- config ---------------------------------------
LLM_MODEL = os.getenv("LLM_MODEL", "gemma4-custom:latest")   # the tag shown by `ollama list`
EMBED_MODEL = "bge-m3"       # alternative: "nomic-embed-text" (English-centric, faster)
CHARS_PER_TOKEN = 3.5        # German text with the Qwen / bge-m3 tokenizers (len/4 would underestimate)
CHUNK_TOKENS = 1800          # max chunk size: chunks end up at ~1500-1800 tokens (~35 chunks for 55k tokens)
OVERLAP_TOKENS = 225         # ~12.5% of the chunk size: the next chunk repeats the last ~225 tokens
OVERLAP_MAX_TOKENS = 300     # ...never more than this
TOP_K = int(os.getenv("RAG_TOP_K", "3"))           # chunks sent to the LLM (each ~1700 tokens; fewer = faster)
ALPHA = 0.8                  # weight of the embedding similarity (the rest is keyword overlap)
NUM_CTX = int(os.getenv("RAG_NUM_CTX", "12288"))   # Ollama default is small (2-4k) -> raise it explicitly
EMBED_NUM_CTX = 8192         # otherwise Ollama may silently truncate big chunks before embedding
KEEP_ALIVE = "30m"           # keep the models loaded between questions (reloading takes many seconds)
THINKING = False             # True = qwen3 reasoning mode (slower, sometimes better)
HISTORY_MESSAGES = 6         # last N chat messages (3 Q/A pairs) sent to the LLM; 0 = no memory
HISTORY_MAX_CHARS = 1200     # each history message is truncated to this length
REWRITE_MAX_WORDS = 12       # follow-ups up to this many words are rewritten into a standalone query;
                             # longer questions are searched as they are (saves one LLM call). 999 = always rewrite
INDEX_VERSION = 3            # part of the document id; also the version of the old .rag_cache indexes we can import

# vector database
VECTOR_DB_DIR = Path(os.getenv("RAG_DB_DIR", ".vector_db"))
COLLECTION = f"data_pool_{EMBED_MODEL}"          # one collection per embedding model (vector sizes differ)
LEGACY_CACHE_DIR = Path(".rag_cache")            # indexes written by the previous (numpy) version
EXACT_BELOW = 300            # a search scope (one document / the whole pool) with at most this many chunks is
                             # scored exactly, chunk by chunk; bigger scopes use the vector index (ANN)
N_CANDIDATES = 50            # candidates the vector index returns for bigger scopes, before the 0.8/0.2 scoring
POOL_ID = "__all__"          # doc_id that means "search all documents"

SYSTEM_PROMPT = (
    "You are an assistant that answers questions about the provided document "
    "(Volkswagen Produktentstehungsprozess / PEP). Use ONLY the context excerpts. "
    "If the answer is not in the context, say you could not find it in the document. "
    "Answer in the same language as the question. Cite the excerpt numbers you used, "
    "like [2] or [1][4]."
)
# used when searching the whole pool
POOL_PROMPT = (
    "You are an assistant that answers questions about the user's own documents (a personal data pool; it "
    "contains e.g. Volkswagen Produktentstehungsprozess / PEP guidelines). Use ONLY the context excerpts. "
    "If the answer is not in the context, say you could not find it in the documents. "
    "Answer in the same language as the question. Cite the excerpt numbers you used, like [2] or [1][4]. "
    "Every excerpt names its document: say which document a fact comes from when the excerpts come "
    "from different documents."
)
# only appended when the chat already has earlier turns
MEMORY_NOTE = (
    " Earlier conversation turns are given for continuity (e.g. to resolve 'it' or 'the next phase'), "
    "but facts must come from the current context excerpts."
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
        resp = ollama.embed(model=EMBED_MODEL, input=texts[i : i + batch], options={"num_ctx": EMBED_NUM_CTX},
                            keep_alive=KEEP_ALIVE)
        vecs.extend(resp["embeddings"])
    arr = np.array(vecs, dtype=np.float32)
    arr /= np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12  # normalize -> dot = cosine
    return arr


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


def get_reranker():
    """No reranker in this plain engine (kept so eval_retrieval.py works with both engine versions)."""
    return None


# ----------------------------- vector database ------------------------------
def _open_collection(db_dir: Path):
    try:
        import chromadb
    except ImportError as e:
        raise ImportError("ChromaDB is not installed. Run:  pip install chromadb") from e
    db_dir.mkdir(parents=True, exist_ok=True)
    try:
        from chromadb.config import Settings

        client = chromadb.PersistentClient(path=str(db_dir), settings=Settings(anonymized_telemetry=False))
    except ImportError:
        client = chromadb.PersistentClient(path=str(db_dir))
    # we always pass our own bge-m3 embeddings, so no embedding function (it would try to download a model)
    return client.get_or_create_collection(COLLECTION, embedding_function=None)


# =================================== engine =================================
class RagEngine:
    """The data pool: documents are chunked, embedded and stored in ChromaDB; questions search one document
    or all of them (doc_id = POOL_ID)."""

    def __init__(self, db_dir: Path = VECTOR_DB_DIR, legacy_dir: Path | None = LEGACY_CACHE_DIR):
        self.db_dir = Path(db_dir)
        self.col = _open_collection(self.db_dir)
        # doc_id -> {"name", "chunks" (number of chunks), "file_hash"}
        self.docs: dict[str, dict] = {}
        self._migrated: set[str] = set()
        self._load_registry()
        if legacy_dir:
            self._migrate_legacy(Path(legacy_dir))

    # ---- registry of the documents in the pool (small json next to the database) ----
    @property
    def _reg_path(self) -> Path:
        return self.db_dir / f"registry_{COLLECTION}.json"

    def _load_registry(self) -> None:
        if self._reg_path.exists():
            try:
                data = json.loads(self._reg_path.read_text(encoding="utf-8"))
                self.docs = data.get("docs", {})
                self._migrated = set(data.get("migrated", []))
                return
            except Exception:  # noqa: BLE001
                pass
        if self.col.count() > 0:  # registry lost: rebuild it from the chunk metadata
            for meta in self.col.get(include=["metadatas"])["metadatas"]:
                d = self.docs.setdefault(meta["doc_id"], {"name": meta["doc_name"], "chunks": 0, "file_hash": ""})
                d["chunks"] += 1
            self._save_registry()

    def _save_registry(self) -> None:
        tmp = self._reg_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"docs": self.docs, "migrated": sorted(self._migrated)}, ensure_ascii=False),
                       encoding="utf-8")
        tmp.replace(self._reg_path)

    # ---- storing chunks ----
    def _store(self, doc_id: str, name: str, chunks: list[dict], vecs: np.ndarray) -> None:
        step = 200  # ChromaDB limits the size of one write
        for s in range(0, len(chunks), step):
            idx = range(s, min(s + step, len(chunks)))
            self.col.upsert(
                ids=[f"{doc_id}:{i}" for i in idx],
                embeddings=[[float(x) for x in vecs[i]] for i in idx],
                documents=[chunks[i]["text"] for i in idx],
                metadatas=[{"doc_id": doc_id, "doc_name": name, "chunk_idx": i,
                            "page_start": int(chunks[i]["page_start"]), "page_end": int(chunks[i]["page_end"])}
                           for i in idx],
            )

    def _migrate_legacy(self, legacy_dir: Path) -> None:
        """Import indexes of the previous version (.rag_cache/*.json + *.npy) without re-embedding."""
        if not legacy_dir.exists():
            return
        changed = False
        for meta_f in sorted(legacy_dir.glob("*.json")):
            stem, vecs_f = meta_f.stem, meta_f.with_suffix(".npy")
            if stem in self._migrated or stem in self.docs or not vecs_f.exists():
                continue
            try:
                meta = json.loads(meta_f.read_text(encoding="utf-8"))
                if not isinstance(meta, dict) or meta.get("version") != INDEX_VERSION or "chunks" not in meta:
                    continue  # not an index of the previous version (e.g. the cache of rag_vw_pep.py)
                fh = meta.get("file_hash") or ""
                vecs = np.load(vecs_f)
                if len(vecs) != len(meta["chunks"]):
                    continue
                if fh and any(d.get("file_hash") == fh for d in self.docs.values()):
                    self._migrated.add(stem)  # this file is already in the pool
                    changed = True
                    continue
                self._store(stem, meta["name"], meta["chunks"], vecs)
                self.docs[stem] = {"name": meta["name"], "chunks": len(meta["chunks"]), "file_hash": fh}
                self._migrated.add(stem)
                changed = True
                print(f"Imported '{meta['name']}' from the old cache ({len(meta['chunks'])} chunks, no re-embedding).")
            except Exception as e:  # noqa: BLE001
                print(f"Could not import {meta_f.name}: {e}")
        if changed:
            self._save_registry()

    # ---- adding / removing documents ----
    @staticmethod
    def _hashes(path: Path) -> tuple[str, str]:
        """(hash of the file, id of this index = file + chunking/embedding settings)."""
        h = hashlib.md5()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        file_hash = h.hexdigest()
        cfg = (EMBED_MODEL, CHUNK_TOKENS, OVERLAP_TOKENS, OVERLAP_MAX_TOKENS, CHARS_PER_TOKEN, INDEX_VERSION)
        return file_hash[:12], hashlib.md5((file_hash + repr(cfg)).encode()).hexdigest()[:12]

    def add_document(self, path: str | Path, name: str | None = None) -> dict:
        path = Path(path)
        file_hash, doc_id = self._hashes(path)
        if doc_id in self.docs:
            d = self.docs[doc_id]
            return {"doc_id": doc_id, "name": d["name"], "chunks": d["chunks"], "cached": True}

        print(f"Reading and chunking {path.name} ...")
        chunks = chunk_pages(load_text(path))
        if not chunks:
            raise ValueError("No text could be extracted (scanned PDF? OCR needed).")
        print(f"{len(chunks)} chunks (~{sum(approx_tokens(c['text']) for c in chunks)} tokens). Embedding ...")
        vecs = embed([c["text"] for c in chunks])
        name = name or path.name
        try:
            self._store(doc_id, name, chunks, vecs)
        except Exception:
            self.col.delete(where={"doc_id": doc_id})  # no half-written document
            raise
        self.docs[doc_id] = {"name": name, "chunks": len(chunks), "file_hash": file_hash}
        self._save_registry()
        for old in [i for i, d in self.docs.items() if i != doc_id and d.get("file_hash") == file_hash]:
            print(f"Removing the older index of the same file ({old}).")
            self.delete_document(old)
        return {"doc_id": doc_id, "name": name, "chunks": len(chunks), "cached": False}

    def delete_document(self, doc_id: str) -> bool:
        if doc_id not in self.docs:
            return False
        self.col.delete(where={"doc_id": doc_id})
        self.docs.pop(doc_id)
        self._save_registry()
        return True

    def list_documents(self) -> list[dict]:
        return [{"doc_id": i, "name": d["name"], "chunks": d["chunks"]} for i, d in self.docs.items()]

    # ---- retrieval ----
    def _scope(self, doc_id: str):
        """(where-filter, number of chunks) for one document or the whole pool."""
        if doc_id == POOL_ID:
            return None, sum(d["chunks"] for d in self.docs.values())
        if doc_id not in self.docs:
            raise KeyError(f"Unknown document: {doc_id}")
        return {"doc_id": doc_id}, self.docs[doc_id]["chunks"]

    @staticmethod
    def _hit(text: str, meta: dict, sem: float | None = None) -> dict:
        return {"text": text, "doc_id": meta["doc_id"], "doc_name": meta["doc_name"],
                "chunk_idx": int(meta["chunk_idx"]), "page_start": int(meta["page_start"]),
                "page_end": int(meta["page_end"]), "sem": sem}

    def _all_chunks(self, doc_id: str) -> list[dict]:
        """Every chunk of the scope in document order."""
        where, n = self._scope(doc_id)
        if n == 0:
            return []
        res = self.col.get(where=where, include=["documents", "metadatas"])
        hits = [self._hit(t, m) for t, m in zip(res["documents"], res["metadatas"])]
        return sorted(hits, key=lambda h: (h["doc_name"], h["doc_id"], h["chunk_idx"]))

    def _candidates(self, doc_id: str, q: np.ndarray) -> list[dict]:
        """Chunks with their embedding similarity: all of the scope if it is small, else the ANN top N."""
        where, n = self._scope(doc_id)
        if n == 0:
            return []
        if n <= EXACT_BELOW:  # exact: same numbers as the numpy version
            res = self.col.get(where=where, include=["documents", "metadatas", "embeddings"])
            order = sorted(range(len(res["ids"])), key=lambda i: (res["metadatas"][i]["doc_name"],
                           res["metadatas"][i]["doc_id"], res["metadatas"][i]["chunk_idx"]))
            vecs = np.asarray(res["embeddings"], dtype=np.float32)[order]
            sims = vecs @ q
            return [self._hit(res["documents"][i], res["metadatas"][i], float(s)) for i, s in zip(order, sims)]
        res = self.col.query(query_embeddings=[[float(x) for x in q]], n_results=min(N_CANDIDATES, n),
                             where=where, include=["documents", "metadatas", "distances"])
        # vectors are normalized and ChromaDB's default distance is the squared L2 distance: cos = 1 - d / 2
        return [self._hit(t, m, 1.0 - float(d) / 2.0)
                for t, m, d in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])]

    def retrieve(self, doc_id: str, query: str, k: int = TOP_K, mode: str = "hybrid") -> list[dict]:
        """Top chunks as dicts (text, doc_name, pages, score).
        mode: hybrid (default, as rag_vw_pep.py) | dense | bm25 (= keyword overlap only) | rerank (= hybrid)."""
        if mode == "bm25":
            hits = self._all_chunks(doc_id)
            kw = keyword_score(query, hits)
            return [dict(hits[i], score=float(kw[i])) for i in np.argsort(-kw)[:k] if kw[i] > 0]
        hits = self._candidates(doc_id, embed([query])[0])
        if not hits:
            return []
        sem = np.array([h["sem"] for h in hits], dtype=np.float32)
        score = sem if mode == "dense" else ALPHA * sem + (1 - ALPHA) * keyword_score(query, hits)
        return [dict(hits[i], score=float(score[i])) for i in np.argsort(-score)[:k]]

    def export_chunks(self) -> list[dict]:
        """All chunks of all documents (used by make_pep_dataset.py)."""
        out = []
        for doc_id, d in self.docs.items():
            for h in self._all_chunks(doc_id):
                out.append({"doc": doc_id, "doc_name": d["name"], "idx": h["chunk_idx"], "text": h["text"],
                            "page_start": h["page_start"], "page_end": h["page_end"]})
        return out

    # ---- conversation memory ----
    @staticmethod
    def _trim_history(history: list[dict] | None) -> list[dict]:
        """Keep the last N user/assistant messages, each truncated."""
        if HISTORY_MESSAGES <= 0:
            return []
        out = []
        for m in (history or [])[-HISTORY_MESSAGES:]:
            if m.get("role") in ("user", "assistant") and m.get("content"):
                out.append({"role": m["role"], "content": str(m["content"])[:HISTORY_MAX_CHARS]})
        return out

    def condense_question(self, history: list[dict], question: str) -> str:
        """Rewrite a short follow-up ("and the next phase?") into a standalone search query."""
        if not history or len(question.split()) > REWRITE_MAX_WORDS:
            return question
        convo = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in history)
        prompt = (
            "Given the conversation and a follow-up question, rewrite the follow-up as a "
            "standalone question that can be understood without the conversation. Keep the "
            "same language. Output ONLY the rewritten question.\n\n"
            f"Conversation:\n{convo}\n\nFollow-up question: {question}\n\nStandalone question:"
        )
        try:
            r = ollama.chat(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                think=THINKING,
                options={"num_ctx": NUM_CTX, "temperature": 0.0, "num_predict": 400},
                keep_alive=KEEP_ALIVE,
            )
            text = re.sub(r"<think>.*?</think>", "", r["message"]["content"], flags=re.S)
            return text.strip().strip('"') or question
        except Exception:  # noqa: BLE001
            return question

    # ---- generation ----
    def ask_stream(self, doc_id: str, question: str, history: list[dict] | None = None) -> Iterator[dict]:
        """Yields {"search_query"}, then {"token"} events, then one {"sources": [...]} event."""
        if doc_id != POOL_ID and doc_id not in self.docs:
            raise KeyError(f"Unknown document: {doc_id}")
        if not self.docs:
            raise ValueError("The data pool is empty - upload a document first.")
        pool = doc_id == POOL_ID
        history = self._trim_history(history)
        search_q = self.condense_question(history, question)
        yield {"search_query": search_q}
        t_start = time.time()
        hits = self.retrieve(doc_id, search_q)
        t_retrieval = time.time() - t_start
        context = "\n\n".join(
            f"[{n}] ({'document: ' + h['doc_name'] + ', ' if pool else ''}pages {h['page_start']}-{h['page_end']})\n{h['text']}"
            for n, h in enumerate(hits, 1)
        )
        messages = [
            {"role": "system", "content": (POOL_PROMPT if pool else SYSTEM_PROMPT) + (MEMORY_NOTE if history else "")},
            *history,
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ]
        stream = ollama.chat(
            model=LLM_MODEL,
            messages=messages,
            stream=True,
            think=THINKING,
            options={"num_ctx": NUM_CTX, "temperature": 0.2},
            keep_alive=KEEP_ALIVE,
        )
        t_first, stats = None, {}
        for part in stream:
            tok = part["message"]["content"]
            if tok:
                if t_first is None:
                    t_first = time.time() - t_start
                yield {"token": tok}
            if part.get("done"):
                stats = part
        # where did the time go? (printed in the terminal that runs chat_flask.py)
        sec = lambda k: (stats.get(k) or 0) / 1e9  # noqa: E731  (Ollama reports nanoseconds)
        gen_s = sec("eval_duration")
        print(f"[timing] search {t_retrieval:.1f}s | model load {sec('load_duration'):.1f}s | "
              f"prompt {stats.get('prompt_eval_count') or '?'} tokens read in {sec('prompt_eval_duration'):.1f}s | "
              f"first word after {t_first or 0:.1f}s | answer {stats.get('eval_count') or '?'} tokens"
              + (f" at {(stats.get('eval_count') or 0) / gen_s:.1f} tok/s" if gen_s else ""), flush=True)

        yield {
            "sources": [
                {
                    "n": n,
                    "pages": f"{h['page_start']}-{h['page_end']}",
                    "heading": "",
                    "doc": h["doc_name"],
                    "score": round(h["score"], 2),
                    "text": h["text"],
                }
                for n, h in enumerate(hits, 1)
            ]
        }