"""
main_autoindex.py - Single-file RAG API (Ollama Embeddings + Chroma)
- Chat endpoint (/chat)
- Automatic indexing (local folders + optional S3) on startup + polling
- Prepared for per-space isolation via metadata filters (space_id)
"""

from __future__ import annotations

import os
import re
import json
import time
import math
import uuid
import asyncio
import hashlib
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores.utils import filter_complex_metadata

# Optional: BM25 (Korean tokenization is weak with default whitespace; enable only if you accept that)
try:
    from langchain_community.retrievers import BM25Retriever
except Exception:
    BM25Retriever = None  # type: ignore

# Optional: S3
try:
    import boto3
except Exception:
    boto3 = None  # type: ignore


# ----------------------------
# 1) Configuration
# ----------------------------
DB_DIR = os.getenv("DB_DIR", "chroma_db")

# Automatic indexing sources
LOCAL_SOURCES = [p for p in os.getenv("LOCAL_SOURCES", "data_storage").split(";") if p.strip()]
S3_BUCKET = os.getenv("S3_BUCKET", "").strip()
S3_PREFIXES = [p for p in os.getenv("S3_PREFIXES", "").split(";") if p.strip()]
S3_REGION = os.getenv("S3_REGION", "").strip()

INDEX_POLL_SECONDS = int(os.getenv("INDEX_POLL_SECONDS", "120"))
INDEX_STATE_PATH = os.getenv("INDEX_STATE_PATH", os.path.join(DB_DIR, "index_state.json"))
TMP_DIR = os.getenv("TMP_DIR", "tmp_ingest")

# Retrieval parameters (dense)
K_DENSE = int(os.getenv("K_DENSE", "20"))
FETCH_K = int(os.getenv("FETCH_K", "40"))
LAMBDA_MULT = float(os.getenv("LAMBDA_MULT", "0.35"))

# Optional sparse
ENABLE_BM25 = os.getenv("ENABLE_BM25", "0").strip() == "1"
K_SPARSE = int(os.getenv("K_SPARSE", "20"))

# Final selection
K_FINAL = int(os.getenv("K_FINAL", "8"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "6500"))

# Space isolation (v1: default single space)
DEFAULT_SPACE_ID = os.getenv("DEFAULT_SPACE_ID", "default")


# ----------------------------
# 2) Prompt
# ----------------------------
BASE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant.\n"
     "Use ONLY the information provided in [Context] for factual claims.\n"
     "If the answer is not in the context, say you don't have that information.\n\n"
     "CRITICAL LANGUAGE RULE:\n"
     "- You MUST answer ONLY in {answer_language}.\n"
     "- Even if the context is in a different language, translate it into {answer_language}.\n\n"
     "STYLE RULES:\n"
     "- Do NOT mention documents, files, pages, sources, or citations.\n"
     "- Keep it practical and direct.\n"
    ),
    ("human",
     "[Context]\n{context}\n\n"
     "[Question]\n{question}\n\n"
     "[Answer]\n")
])

# ----------------------------
# 3) LLM / Embeddings (Ollama)
# ----------------------------
LLM_MODEL = os.getenv("LLM_MODEL", "gemma2:2b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3")  # recommended for EN/KR in MVP

llm = ChatOllama(
    model=LLM_MODEL,
    temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
    num_predict=int(os.getenv("LLM_NUM_PREDICT", "240")),
    timeout=int(os.getenv("LLM_TIMEOUT", "25")),
)

embedding_model = OllamaEmbeddings(model=EMBED_MODEL)


# ----------------------------
# 4) Vector Store
# ----------------------------
def get_vectorstore() -> Chroma:
    return Chroma(
        persist_directory=DB_DIR,
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"},
    )


# ----------------------------
# 5) API Models
# ----------------------------
class ChatRequest(BaseModel):
    question: str
    space_id: Optional[str] = None  # 준비용. MVP에선 기본값 사용.

class SourceInfo(BaseModel):
    source: str
    snippet: str

class ChatResponse(BaseModel):
    answer: str
    time_taken: float
    sources: List[SourceInfo]
    
class IngestRequest(BaseModel):
    space_id: str
    file_path: str


# ----------------------------
# 6) Utilities
# ----------------------------
def preprocess_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()

def is_korean(text: str) -> bool:
    return any(("\uAC00" <= c <= "\uD7A3") or ("\u1100" <= c <= "\u11FF") or ("\u3130" <= c <= "\u318F") for c in (text or ""))

def sanitize_metadata(meta: dict) -> dict:
    clean = {}
    for k, v in (meta or {}).items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            clean[k] = v
        else:
            clean[k] = str(v)
    return clean

def split_semantic_then_fallback(docs: List[Document]) -> List[Document]:
    try:
        return SemanticChunker(
            embedding=embedding_model,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=90,
        ).split_documents(docs)
    except Exception:
        splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=120)
        return splitter.split_documents(docs)

def build_context(docs: List[Document], max_chars: int = MAX_CONTEXT_CHARS) -> str:
    parts: List[str] = []
    total = 0
    for d in docs:
        text = (d.page_content or "").strip()
        if not text:
            continue
        if total + len(text) > max_chars:
            remain = max_chars - total
            if remain > 200:
                parts.append(text[:remain])
            break
        parts.append(text)
        total += len(text)
    return "\n\n---\n\n".join(parts)

def cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb + 1e-12)

# def rerank_by_embedding(query: str, docs: List[Document], top_k: int) -> List[Document]:
#     if not docs:
#         return []
#     q = embedding_model.embed_query(query)
#     doc_vecs = embedding_model.embed_documents([(d.page_content or "") for d in docs])
#     scored = [(cosine(q, v), d) for v, d in zip(doc_vecs, docs)]
#     scored.sort(key=lambda x: x[0], reverse=True)
#     return [d for _, d in scored[:top_k]]

def rerank_by_embedding(query: str, docs: List[Document], top_k: int) -> List[Document]:
    if not docs:
        return []
    
    q = embedding_model.embed_query(query)
    
    doc_vecs = []
    texts = [(d.page_content or "") for d in docs]
    batch_size = 10
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_vecs = embedding_model.embed_documents(batch)
        doc_vecs.extend(batch_vecs)

    scored = [(cosine(q, v), d) for v, d in zip(doc_vecs, docs)]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in scored[:top_k]]

def stable_doc_id(source_uri: str, space_id: str) -> str:
    h = hashlib.sha1(f"{space_id}::{source_uri}".encode("utf-8")).hexdigest()
    return h

def file_sha256(path: str) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def load_state() -> dict:
    try:
        with open(INDEX_STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"files": {}}

def save_state(state: dict) -> None:
    ensure_dir(os.path.dirname(INDEX_STATE_PATH) or ".")
    tmp = INDEX_STATE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    os.replace(tmp, INDEX_STATE_PATH)

def supported_ext(name: str) -> bool:
    nl = (name or "").lower()
    return nl.endswith(".pdf") or nl.endswith(".txt") or nl.endswith(".md")

def load_and_chunk_from_path(path: str, source_label: str, space_id: str, doc_id: str) -> List[Document]:
    ext = os.path.splitext(path)[1].lower()

    if ext == ".pdf":
        loader = PyMuPDFLoader(path)
        docs = loader.load()
        for d in docs:
            d.page_content = preprocess_text(d.page_content)
            d.metadata = d.metadata or {}
            d.metadata["source"] = source_label
        chunks = split_semantic_then_fallback(docs)

    elif ext in [".txt", ".md"]:
        raw = TextLoader(path, encoding="utf-8").load()[0].page_content
        raw = preprocess_text(raw)
        docs = [Document(page_content=raw, metadata={"source": source_label})]
        chunks = split_semantic_then_fallback(docs)

    else:
        return []

    cleaned: List[Document] = []
    for i, d in enumerate(chunks):
        d.metadata = d.metadata or {}
        d.metadata.update({
            "doc_id": doc_id,
            "chunk_index": i,
            "source": source_label,
            "space_id": space_id,
        })
        d.metadata = sanitize_metadata(d.metadata)
        cleaned.append(d)

    cleaned = filter_complex_metadata(cleaned)
    return cleaned

async def rebuild_bm25(app: FastAPI, space_id: str = DEFAULT_SPACE_ID) -> None:
    if not ENABLE_BM25 or BM25Retriever is None:
        app.state.bm25 = None
        return

    vectordb: Chroma = app.state.vectordb
    raw = vectordb._collection.get(where={"space_id": space_id}, include=["documents", "metadatas"])
    docs = [
        Document(page_content=t, metadata=(m or {}))
        for t, m in zip(raw.get("documents", []), raw.get("metadatas", []))
        if t and t.strip()
    ]
    if docs:
        bm25 = BM25Retriever.from_documents(docs)
        bm25.k = K_SPARSE
        app.state.bm25 = bm25
    else:
        app.state.bm25 = None


# ----------------------------
# 7) Indexing sources
# ----------------------------
@dataclass(frozen=True)
class IndexItem:
    uri: str
    label: str
    local_path: str
    fingerprint: str

def list_local_items(space_id: str) -> List[IndexItem]:
    items: List[IndexItem] = []
    for root in LOCAL_SOURCES:
        if not root or not os.path.exists(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if not supported_ext(fn):
                    continue
                path = os.path.join(dirpath, fn)
                uri = f"file://{os.path.abspath(path)}"
                fp = f"sha256:{file_sha256(path)}"
                items.append(IndexItem(uri=uri, label=path, local_path=path, fingerprint=fp))
    return items

def list_s3_items(space_id: str) -> List[IndexItem]:
    if not (S3_BUCKET and S3_PREFIXES):
        return []
    if boto3 is None:
        print("!! boto3 not installed; S3 indexing disabled.")
        return []

    ensure_dir(TMP_DIR)
    session = boto3.session.Session(region_name=S3_REGION or None)
    s3 = session.client("s3")

    out: List[IndexItem] = []
    for prefix in S3_PREFIXES:
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if not supported_ext(key):
                    continue
                etag = (obj.get("ETag") or "").strip('"')
                lm = obj.get("LastModified")
                fp = f"etag:{etag};lm:{lm}"
                uri = f"s3://{S3_BUCKET}/{key}"
                local_path = os.path.join(TMP_DIR, uuid.uuid4().hex + os.path.splitext(key)[1].lower())
                s3.download_file(S3_BUCKET, key, local_path)
                out.append(IndexItem(uri=uri, label=uri, local_path=local_path, fingerprint=fp))
    return out

async def sync_once(app: FastAPI, space_id: str = DEFAULT_SPACE_ID) -> Tuple[int, int, int]:
    vectordb: Chroma = app.state.vectordb
    state = load_state()
    known: Dict[str, dict] = state.get("files", {})

    items = list_local_items(space_id) + list_s3_items(space_id)
    seen_uris = {it.uri for it in items}

    added_or_updated = 0
    removed = 0

    # remove missing
    missing = [uri for uri in list(known.keys()) if uri not in seen_uris]
    if missing:
        async with app.state.write_lock:
            for uri in missing:
                doc_id = known[uri].get("doc_id")
                if doc_id:
                    data = vectordb._collection.get(where={"doc_id": doc_id})
                    ids = data.get("ids", [])
                    if ids:
                        vectordb._collection.delete(ids=ids)
                known.pop(uri, None)
                removed += 1

    # add/update
    for it in items:
        prev = known.get(it.uri)
        if prev and prev.get("fingerprint") == it.fingerprint:
            continue

        doc_id = stable_doc_id(it.uri, space_id)

        try:
            chunks = load_and_chunk_from_path(it.local_path, it.label, space_id, doc_id)
            if not chunks:
                continue

            async with app.state.write_lock:
                old = vectordb._collection.get(where={"doc_id": doc_id})
                old_ids = old.get("ids", [])
                if old_ids:
                    vectordb._collection.delete(ids=old_ids)

                vectordb.add_documents(chunks)

            known[it.uri] = {
                "fingerprint": it.fingerprint,
                "doc_id": doc_id,
                "label": it.label,
                "space_id": space_id,
                "indexed_at": time.time(),
            }
            added_or_updated += 1

        finally:
            if it.uri.startswith("s3://"):
                try:
                    os.remove(it.local_path)
                except Exception:
                    pass

    state["files"] = known
    save_state(state)

    async with app.state.rebuild_lock:
        await rebuild_bm25(app, space_id=space_id)

    return added_or_updated, removed, len(items)

async def periodic_sync(app: FastAPI) -> None:
    while not app.state.stop_event.is_set():
        try:
            await sync_once(app, space_id=DEFAULT_SPACE_ID)
        except Exception as e:
            print(f"!! indexer error: {e}")
        await asyncio.sleep(INDEX_POLL_SECONDS)


# ----------------------------
# 8) FastAPI lifespan
# ----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_dir(DB_DIR)
    ensure_dir(TMP_DIR)

    app.state.vectordb = get_vectorstore()
    app.state.bm25 = None

    app.state.write_lock = asyncio.Lock()
    app.state.rebuild_lock = asyncio.Lock()
    app.state.stop_event = asyncio.Event()

    # await sync_once(app, space_id=DEFAULT_SPACE_ID)
    # app.state.sync_task = asyncio.create_task(periodic_sync(app))

    print(">> Server Started (auto-index enabled)")
    yield

    app.state.stop_event.set()
    # try:
    #     app.state.sync_task.cancel()
    # except Exception:
    #     pass
    print(">> Server Shutdown")


app = FastAPI(title="RAG API (Auto-Index)", lifespan=lifespan)


# ----------------------------
# 9) Chat API
# ----------------------------
@app.post("/chatbot", response_model=ChatResponse)
async def chat(req: ChatRequest):
    start = time.time()
    vectordb: Chroma = app.state.vectordb
    bm25 = getattr(app.state, "bm25", None)

    q = (req.question or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="question is required")

    space_id = (req.space_id or DEFAULT_SPACE_ID).strip() or DEFAULT_SPACE_ID
    answer_language = "Korean" if is_korean(q) else "English"

    dense_search_type = "mmr" if len(q) >= 12 else "similarity"
    dense_kwargs = {"k": K_DENSE, "fetch_k": FETCH_K, "lambda_mult": LAMBDA_MULT} if dense_search_type == "mmr" else {"k": K_DENSE}
    dense_kwargs["filter"] = {"space_id": space_id}

    dense_docs = vectordb.as_retriever(
        search_type=dense_search_type,
        search_kwargs=dense_kwargs,
    ).invoke(q)

    candidates = list(dense_docs)

    if ENABLE_BM25 and bm25 is not None:
        try:
            bm25.k = K_SPARSE
            sparse_docs = bm25.invoke(q)
            seen = {d.metadata.get("doc_id", "") + ":" + str(d.metadata.get("chunk_index", "")) for d in candidates}
            for d in sparse_docs:
                if (d.metadata or {}).get("space_id") != space_id:
                    continue
                key = d.metadata.get("doc_id", "") + ":" + str(d.metadata.get("chunk_index", ""))
                if key not in seen:
                    candidates.append(d)
                    seen.add(key)
        except Exception:
            pass

    final_docs = rerank_by_embedding(q, candidates, top_k=K_FINAL)

    if not final_docs:
        answer = "관련 자료에서 답을 찾지 못했습니다." if answer_language == "Korean" else "I don't have information about that."
        return ChatResponse(answer=answer, time_taken=time.time() - start, sources=[])

    context = build_context(final_docs, max_chars=MAX_CONTEXT_CHARS)
    chain = BASE_PROMPT | llm | StrOutputParser()
    answer = (chain.invoke({"context": context, "question": q, "answer_language": answer_language}) or "").strip()

    sources = [
        SourceInfo(
            source=(d.metadata or {}).get("source", "unknown"),
            snippet=(d.page_content or ""),
        )
        for d in final_docs
    ]

    return ChatResponse(answer=answer, time_taken=time.time() - start, sources=sources)

@app.post("/ingest")
async def ingest_file(req: IngestRequest):
    if not os.path.exists(req.file_path):
        raise HTTPException(status_code=404, detail=f"File not found: {req.file_path}")

    vectordb: Chroma = app.state.vectordb

    uri = f"file://{os.path.abspath(req.file_path)}"
    doc_id = stable_doc_id(uri, req.space_id)
    source_label = os.path.basename(req.file_path)

    chunks = load_and_chunk_from_path(req.file_path, source_label, req.space_id, doc_id)

    if not chunks:
        return {"status": "skipped", "message": "지원하지 않는 확장자이거나 추출할 텍스트가 없습니다."}

    async with app.state.write_lock:
        old_data = vectordb._collection.get(where={"doc_id": doc_id})
        old_ids = old_data.get("ids", [])
        if old_ids:
            vectordb._collection.delete(ids=old_ids)

        batch_size = 10
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            vectordb.add_documents(batch)
            print(f"임베딩 진행 중... ({i + len(batch)} / {len(chunks)})")

    async with app.state.rebuild_lock:
        await rebuild_bm25(app, space_id=req.space_id)

    return {
        "status": "success", 
        "message": f"성공적으로 {len(chunks)}개의 청크를 DB에 추가했습니다.",
        "space_id": req.space_id
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main_autoindex:app", host="0.0.0.0", port=8000, reload=True)
