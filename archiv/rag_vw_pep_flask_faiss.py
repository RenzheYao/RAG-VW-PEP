"""
rag_vw_pep_flask.py - your plain, fast RAG as an engine for chat_flask.py, with a FAISS vector index and
SQLite as a persistent "personal data pool" (scales to hundreds of thousands of chunks).

What it does
  indexing : pypdf text -> sentence/paragraph chunks (~1800 tokens, ~225 overlap) -> bge-m3 embeddings
             -> text/pages/vectors stored in SQLite (.faiss_db/), vectors also in a FAISS index
  search   : one document  -> exactly the scoring of rag_vw_pep.py (0.8 * cosine + 0.2 * keyword overlap)
             all documents -> the "pool": FAISS finds candidates across every file, then the
                              same 0.8 / 0.2 scoring picks the top chunks (answers name their document)
  answer   : local Ollama LLM, conversation memory (short follow-ups are rewritten into a standalone query)

Data pool
  - add files any time (add_document), remove them (delete_document); everything survives restarts
  - old indexes (.rag_cache/ and the ChromaDB folder .vector_db/) are imported automatically, without
    re-embedding (the ChromaDB import needs `pip install chromadb` once)
  - RAG_INDEX=hnsw switches FAISS from exact search (default) to the approximate HNSW graph for very large pools

Setup:  ollama pull bge-m3   (+ your chat model, see `ollama list`)
        pip install ollama numpy pypdf faiss-cpu
"""

import hashlib
import json
import os
import re
import sqlite3
import threading
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

# vector database (FAISS index + SQLite)
VECTOR_DB_DIR = Path(os.getenv("RAG_DB_DIR", ".faiss_db"))
INDEX_TYPE = os.getenv("RAG_INDEX", "flat").lower()   # flat = exact search | hnsw = approximate graph, for 500k+ chunks
HNSW_M, HNSW_EF_CONSTRUCTION, HNSW_EF_SEARCH = 32, 200, 128
LEGACY_CACHE_DIR = Path(".rag_cache")            # indexes written by the first (numpy) version
CHROMA_DIR = Path(".vector_db")                  # the ChromaDB version's database (imported once)
COLLECTION = f"data_pool_{EMBED_MODEL}"
EXACT_BELOW = 300            # a pool with at most this many chunks (and any single document) is scored exactly
                             # chunk by chunk; bigger pools use the FAISS index
N_CANDIDATES = 50            # candidates FAISS returns for big pools, before the 0.8 / 0.2 scoring
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


# =================================== engine =================================
class RagEngine:
    """The data pool. Text, pages and vectors live in SQLite (the source of truth); a FAISS index over the
    vectors does the fast similarity search. Questions search one document or all of them (doc_id = POOL_ID)."""

    def __init__(self, db_dir: Path = VECTOR_DB_DIR, legacy_dir: Path | None = LEGACY_CACHE_DIR,
                 chroma_dir: Path | None = CHROMA_DIR):
        try:
            import faiss
        except ImportError as e:
            raise ImportError("FAISS is not installed. Run:  pip install faiss-cpu") from e
        self.faiss = faiss
        self.db_dir = Path(db_dir)
        self.db_dir.mkdir(parents=True, exist_ok=True)
        tag = re.sub(r"[^\w.-]", "_", EMBED_MODEL)          # one database per embedding model (vector sizes differ)
        self._index_path = self.db_dir / f"index_{tag}_{INDEX_TYPE}.faiss"
        self._lock = threading.RLock()
        self.db = sqlite3.connect(str(self.db_dir / f"pool_{tag}.sqlite"), check_same_thread=False)
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS docs(doc_id TEXT PRIMARY KEY, name TEXT, chunks INTEGER, file_hash TEXT);
            CREATE TABLE IF NOT EXISTS chunks(id INTEGER PRIMARY KEY AUTOINCREMENT, doc_id TEXT, doc_name TEXT,
                chunk_idx INTEGER, page_start INTEGER, page_end INTEGER, text TEXT, vec BLOB);
            CREATE INDEX IF NOT EXISTS ix_chunks_doc ON chunks(doc_id);
            CREATE TABLE IF NOT EXISTS migrated(src TEXT PRIMARY KEY);
        """)
        self.db.commit()
        # doc_id -> {"name", "chunks" (number of chunks), "file_hash"}
        self.docs: dict[str, dict] = {
            r[0]: {"name": r[1], "chunks": r[2], "file_hash": r[3] or ""}
            for r in self.db.execute("SELECT doc_id, name, chunks, file_hash FROM docs")
        }
        self.index = None
        self._load_index()
        if legacy_dir:
            self._migrate_legacy(Path(legacy_dir))
        if chroma_dir:
            self._migrate_chroma(Path(chroma_dir))

    # ---- the FAISS index (a cache: it can always be rebuilt from the vectors in SQLite) ----
    def _new_index(self, dim: int):
        faiss = self.faiss
        if INDEX_TYPE == "hnsw":
            base = faiss.IndexHNSWFlat(dim, HNSW_M, faiss.METRIC_INNER_PRODUCT)
            base.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
            return faiss.IndexIDMap(base)           # IndexIDMap: our own ids (SQLite row ids); no in-place removal
        return faiss.IndexIDMap2(faiss.IndexFlatIP(dim))  # exact inner product (= cosine, vectors are normalized)

    def _tune(self) -> None:
        if INDEX_TYPE == "hnsw" and self.index is not None:
            self.faiss.downcast_index(self.index.index).hnsw.efSearch = HNSW_EF_SEARCH

    def _save_index(self) -> None:
        if self.index is None:
            self._index_path.unlink(missing_ok=True)
            return
        tmp = self._index_path.with_suffix(".tmp")
        self.faiss.write_index(self.index, str(tmp))
        tmp.replace(self._index_path)

    def _load_index(self) -> None:
        n = self.db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        if n == 0:
            self.index = None
            return
        if self._index_path.exists():
            try:
                idx = self.faiss.read_index(str(self._index_path))
                if idx.ntotal == n:
                    self.index = idx
                    self._tune()
                    return
            except Exception:  # noqa: BLE001
                pass
        print(f"Rebuilding the FAISS index ({n} chunks) ...")
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        self.index = None
        ids, vecs = [], []

        def flush():
            if not ids:
                return
            if self.index is None:
                self.index = self._new_index(len(vecs[0]))
            self.index.add_with_ids(np.ascontiguousarray(np.vstack(vecs), dtype=np.float32),
                                    np.array(ids, dtype=np.int64))
            ids.clear()
            vecs.clear()

        for cid, blob in self.db.execute("SELECT id, vec FROM chunks ORDER BY id"):
            ids.append(cid)
            vecs.append(np.frombuffer(blob, dtype=np.float32))
            if len(ids) >= 20000:
                flush()
        flush()
        self._tune()
        self._save_index()

    # ---- storing chunks ----
    def _store(self, doc_id: str, name: str, chunks: list[dict], vecs: np.ndarray, file_hash: str = "") -> None:
        vecs = np.ascontiguousarray(vecs, dtype=np.float32)
        with self._lock:
            if self.index is not None and self.index.d != vecs.shape[1]:
                raise ValueError(f"Vector size {vecs.shape[1]} does not match the index ({self.index.d}); "
                                 f"delete {self.db_dir}/ if you changed the embedding model.")
            ids = []
            try:
                cur = self.db.cursor()
                for i, c in enumerate(chunks):
                    cur.execute(
                        "INSERT INTO chunks(doc_id, doc_name, chunk_idx, page_start, page_end, text, vec) "
                        "VALUES (?,?,?,?,?,?,?)",
                        (doc_id, name, i, int(c["page_start"]), int(c["page_end"]), c["text"], vecs[i].tobytes()))
                    ids.append(cur.lastrowid)
                cur.execute("INSERT OR REPLACE INTO docs VALUES (?,?,?,?)", (doc_id, name, len(chunks), file_hash))
                self.db.commit()
            except Exception:
                self.db.rollback()
                raise
            if self.index is None:
                self.index = self._new_index(vecs.shape[1])
                self._tune()
            self.index.add_with_ids(vecs, np.array(ids, dtype=np.int64))
            self._save_index()
            self.docs[doc_id] = {"name": name, "chunks": len(chunks), "file_hash": file_hash}

    def _mark_migrated(self, src: str) -> None:
        self.db.execute("INSERT OR IGNORE INTO migrated VALUES (?)", (src,))
        self.db.commit()

    def _is_migrated(self, src: str) -> bool:
        return self.db.execute("SELECT 1 FROM migrated WHERE src=?", (src,)).fetchone() is not None

    def _migrate_legacy(self, legacy_dir: Path) -> None:
        """Import indexes of the first (numpy) version: .rag_cache/*.json + *.npy, without re-embedding."""
        if not legacy_dir.exists():
            return
        for meta_f in sorted(legacy_dir.glob("*.json")):
            stem, vecs_f = meta_f.stem, meta_f.with_suffix(".npy")
            if self._is_migrated("legacy:" + stem) or stem in self.docs or not vecs_f.exists():
                continue
            try:
                meta = json.loads(meta_f.read_text(encoding="utf-8"))
                if not isinstance(meta, dict) or meta.get("version") != INDEX_VERSION or "chunks" not in meta:
                    continue  # not an index of that version (e.g. the cache of rag_vw_pep.py)
                fh = meta.get("file_hash") or ""
                vecs = np.load(vecs_f)
                if len(vecs) != len(meta["chunks"]):
                    continue
                if not (fh and any(d.get("file_hash") == fh for d in self.docs.values())):
                    self._store(stem, meta["name"], meta["chunks"], vecs, fh)
                    print(f"Imported '{meta['name']}' from the old cache ({len(meta['chunks'])} chunks, no re-embedding).")
                self._mark_migrated("legacy:" + stem)
            except Exception as e:  # noqa: BLE001
                print(f"Could not import {meta_f.name}: {e}")

    def _migrate_chroma(self, chroma_dir: Path) -> None:
        """Import the documents of the ChromaDB version (.vector_db/) once, without re-embedding.
        Needs chromadb installed (pip install chromadb); silently skipped otherwise."""
        src = "chroma:" + COLLECTION
        if not chroma_dir.exists() or self._is_migrated(src):
            return
        try:
            import chromadb
            col = chromadb.PersistentClient(path=str(chroma_dir)).get_collection(f"data_pool_{EMBED_MODEL}")
            res = col.get(include=["documents", "metadatas", "embeddings"])
        except Exception:  # noqa: BLE001  (chromadb missing, or no such collection)
            return
        by_doc: dict[str, list] = {}
        for text, meta, vec in zip(res["documents"], res["metadatas"], res["embeddings"]):
            by_doc.setdefault(meta["doc_id"], []).append((meta["chunk_idx"], meta, text, vec))
        hashes = {}
        try:
            reg = json.loads((chroma_dir / f"registry_data_pool_{EMBED_MODEL}.json").read_text(encoding="utf-8"))
            hashes = {k: v.get("file_hash", "") for k, v in reg.get("docs", {}).items()}
        except Exception:  # noqa: BLE001
            pass
        for doc_id, rows in by_doc.items():
            if doc_id in self.docs:
                continue
            rows.sort(key=lambda r: r[0])
            chunks = [{"text": t, "page_start": m["page_start"], "page_end": m["page_end"]} for _, m, t, _ in rows]
            vecs = np.array([r[3] for r in rows], dtype=np.float32)
            self._store(doc_id, rows[0][1]["doc_name"], chunks, vecs, hashes.get(doc_id, ""))
            print(f"Imported '{rows[0][1]['doc_name']}' from ChromaDB ({len(chunks)} chunks, no re-embedding).")
        self._mark_migrated(src)

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
        self._store(doc_id, name, chunks, vecs, file_hash)
        for old in [i for i, d in self.docs.items() if i != doc_id and d.get("file_hash") == file_hash]:
            print(f"Removing the older index of the same file ({old}).")
            self.delete_document(old)
        return {"doc_id": doc_id, "name": name, "chunks": len(chunks), "cached": False}

    def delete_document(self, doc_id: str) -> bool:
        with self._lock:
            if doc_id not in self.docs:
                return False
            ids = [r[0] for r in self.db.execute("SELECT id FROM chunks WHERE doc_id=?", (doc_id,))]
            self.db.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
            self.db.execute("DELETE FROM docs WHERE doc_id=?", (doc_id,))
            self.db.commit()
            self.docs.pop(doc_id)
            if not self.docs:
                self.index = None
                self._save_index()
            elif INDEX_TYPE == "hnsw":  # HNSW cannot remove vectors: rebuild from the stored ones
                self._rebuild_index()
            else:
                self.index.remove_ids(np.array(ids, dtype=np.int64))
                self._save_index()
            return True

    def list_documents(self) -> list[dict]:
        return [{"doc_id": i, "name": d["name"], "chunks": d["chunks"]} for i, d in self.docs.items()]

    # ---- retrieval ----
    def _scope_size(self, doc_id: str) -> int:
        if doc_id == POOL_ID:
            return sum(d["chunks"] for d in self.docs.values())
        if doc_id not in self.docs:
            raise KeyError(f"Unknown document: {doc_id}")
        return self.docs[doc_id]["chunks"]

    _COLS = "id, doc_id, doc_name, chunk_idx, page_start, page_end, text"

    @staticmethod
    def _hit(row, sem: float | None = None) -> dict:
        return {"text": row[6], "doc_id": row[1], "doc_name": row[2], "chunk_idx": int(row[3]),
                "page_start": int(row[4]), "page_end": int(row[5]), "sem": sem}

    def _all_chunks(self, doc_id: str) -> list[dict]:
        """Every chunk of the scope in document order."""
        sql, args = f"SELECT {self._COLS} FROM chunks", ()
        if doc_id != POOL_ID:
            sql, args = sql + " WHERE doc_id=?", (doc_id,)
        with self._lock:
            rows = self.db.execute(sql + " ORDER BY doc_name, doc_id, chunk_idx", args).fetchall()
        return [self._hit(r) for r in rows]

    def _candidates(self, doc_id: str, q: np.ndarray) -> list[dict]:
        """Chunks with their embedding similarity: all of the scope when it is one document or a small pool
        (exact), else the top N of the FAISS index."""
        n = self._scope_size(doc_id)
        if n == 0:
            return []
        with self._lock:
            if doc_id != POOL_ID or n <= EXACT_BELOW:  # exact: numpy over the stored vectors
                sql, args = f"SELECT {self._COLS}, vec FROM chunks", ()
                if doc_id != POOL_ID:
                    sql, args = sql + " WHERE doc_id=?", (doc_id,)
                rows = self.db.execute(sql + " ORDER BY doc_name, doc_id, chunk_idx", args).fetchall()
                vecs = np.vstack([np.frombuffer(r[7], dtype=np.float32) for r in rows])
                return [self._hit(r, float(s)) for r, s in zip(rows, vecs @ q)]
            scores, ids = self.index.search(np.ascontiguousarray(q[None, :], dtype=np.float32), min(N_CANDIDATES, n))
            found = [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]
            if not found:
                return []
            marks = ",".join("?" * len(found))
            rows = {r[0]: r for r in self.db.execute(
                f"SELECT {self._COLS} FROM chunks WHERE id IN ({marks})", [i for i, _ in found])}
        return [self._hit(rows[i], s) for i, s in found if i in rows]

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
        with self._lock:
            rows = self.db.execute(f"SELECT {self._COLS} FROM chunks ORDER BY doc_name, doc_id, chunk_idx").fetchall()
        return [{"doc": r[1], "doc_name": r[2], "idx": int(r[3]), "text": r[6],
                 "page_start": int(r[4]), "page_end": int(r[5])} for r in rows]

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