from datetime import datetime
import traceback
import os, re, json, time, math, uuid, asyncio, hashlib
from dataclasses import dataclass
from typing import List, Dict, Tuple
from fastapi import FastAPI, HTTPException

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.utils import filter_complex_metadata

from core.config import *
from schemas.dto import ChatRequest, ChatResponse, SourceInfo, IngestRequest

try:
    from langchain_community.retrievers import BM25Retriever
except Exception:
    BM25Retriever = None
try:
    import boto3
except Exception:
    boto3 = None

# --- Utilities ---
def preprocess_text(text: str) -> str: return re.sub(r"\s+", " ", (text or "")).strip()
def is_korean(text: str) -> bool: return any(("\uAC00" <= c <= "\uD7A3") or ("\u1100" <= c <= "\u11FF") or ("\u3130" <= c <= "\u318F") for c in (text or ""))
def sanitize_metadata(meta: dict) -> dict:
    clean = {}
    for k, v in (meta or {}).items():
        if v is None: continue
        if isinstance(v, (str, int, float, bool)): clean[k] = v
        else: clean[k] = str(v)
    return clean

def split_semantic_then_fallback(docs: List[Document]) -> List[Document]:
    try:
        return SemanticChunker(embedding=embedding_model, breakpoint_threshold_type="percentile", breakpoint_threshold_amount=90).split_documents(docs)
    except Exception:
        splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=120)
        return splitter.split_documents(docs)

def build_context(docs: List[Document], max_chars: int = MAX_CONTEXT_CHARS) -> str:
    parts, total = [], 0
    for d in docs:
        text = (d.page_content or "").strip()
        if not text: continue
        if total + len(text) > max_chars:
            remain = max_chars - total
            if remain > 200: parts.append(text[:remain])
            break
        parts.append(text)
        total += len(text)
    return "\n\n---\n\n".join(parts)

def cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return dot / (na * nb + 1e-12)

def rerank_by_embedding(query: str, docs: List[Document], top_k: int) -> List[Document]:
    if not docs: return []
    q = embedding_model.embed_query(query)
    doc_vecs, texts, batch_size = [], [(d.page_content or "") for d in docs], 10
    for i in range(0, len(texts), batch_size):
        doc_vecs.extend(embedding_model.embed_documents(texts[i : i + batch_size]))
    scored = [(cosine(q, v), d) for v, d in zip(doc_vecs, docs)]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in scored[:top_k]]

def stable_doc_id(source_uri: str, space_id: str) -> str: return hashlib.sha1(f"{space_id}::{source_uri}".encode("utf-8")).hexdigest()
def file_sha256(path: str) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""): sha.update(chunk)
    return sha.hexdigest()

def load_state() -> dict:
    try:
        with open(INDEX_STATE_PATH, "r", encoding="utf-8") as f: return json.load(f)
    except Exception: return {"files": {}}

def save_state(state: dict) -> None:
    ensure_dir(os.path.dirname(INDEX_STATE_PATH) or ".")
    tmp = INDEX_STATE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f: json.dump(state, f, ensure_ascii=False, indent=2)
    os.replace(tmp, INDEX_STATE_PATH)

def supported_ext(name: str) -> bool:
    nl = (name or "").lower()
    return nl.endswith(".pdf") or nl.endswith(".txt") or nl.endswith(".md")

def load_and_chunk_from_path(path: str, source_label: str, space_id: str, doc_id: str) -> List[Document]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        docs = PyMuPDFLoader(path).load()
        for d in docs:
            d.page_content = preprocess_text(d.page_content)
            d.metadata = d.metadata or {}
            d.metadata["source"] = source_label
        chunks = split_semantic_then_fallback(docs)
    elif ext in [".txt", ".md"]:
        raw = preprocess_text(TextLoader(path, encoding="utf-8").load()[0].page_content)
        chunks = split_semantic_then_fallback([Document(page_content=raw, metadata={"source": source_label})])
    else: return []

    cleaned = []
    for i, d in enumerate(chunks):
        d.metadata = d.metadata or {}
        d.metadata.update({"doc_id": doc_id, "chunk_index": i, "source": source_label, "space_id": space_id})
        d.metadata = sanitize_metadata(d.metadata)
        cleaned.append(d)
    return filter_complex_metadata(cleaned)

async def rebuild_bm25(app: FastAPI, space_id: str = DEFAULT_SPACE_ID) -> None:
    if not ENABLE_BM25 or BM25Retriever is None:
        app.state.bm25 = None
        return
    raw = app.state.vectordb._collection.get(where={"space_id": space_id}, include=["documents", "metadatas"])
    docs = [Document(page_content=t, metadata=(m or {})) for t, m in zip(raw.get("documents", []), raw.get("metadatas", [])) if t and t.strip()]
    if docs:
        bm25 = BM25Retriever.from_documents(docs)
        bm25.k = K_SPARSE
        app.state.bm25 = bm25
    else: app.state.bm25 = None

# --- Auto Indexing Logic ---
@dataclass(frozen=True)
class IndexItem:
    uri: str
    label: str
    local_path: str
    fingerprint: str

def list_local_items(space_id: str) -> List[IndexItem]:
    items = []
    for root in LOCAL_SOURCES:
        if not root or not os.path.exists(root): continue
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if not supported_ext(fn): continue
                path = os.path.join(dirpath, fn)
                items.append(IndexItem(uri=f"file://{os.path.abspath(path)}", label=path, local_path=path, fingerprint=f"sha256:{file_sha256(path)}"))
    return items

def list_s3_items(space_id: str) -> List[IndexItem]:
    if not (S3_BUCKET and S3_PREFIXES) or boto3 is None: return []
    ensure_dir(TMP_DIR)
    s3, out = boto3.session.Session(region_name=S3_REGION or None).client("s3"), []
    for prefix in S3_PREFIXES:
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=S3_BUCKET, Prefix=prefix):
            for obj in page.get("Contents", []):
                if not supported_ext(obj["Key"]): continue
                
                # 💡 f-string 에러를 피하기 위해 ETag를 밖에서 먼저 깔끔하게 처리합니다!
                raw_etag = (obj.get('ETag') or '').strip('"')
                fp = f"etag:{raw_etag};lm:{obj.get('LastModified')}"
                local_path = os.path.join(TMP_DIR, uuid.uuid4().hex + os.path.splitext(obj["Key"])[1].lower())
                
                s3.download_file(S3_BUCKET, obj["Key"], local_path)
                out.append(IndexItem(uri=f"s3://{S3_BUCKET}/{obj['Key']}", label=f"s3://{S3_BUCKET}/{obj['Key']}", local_path=local_path, fingerprint=fp))
    return out

async def sync_once(app: FastAPI, space_id: str = DEFAULT_SPACE_ID) -> Tuple[int, int, int]:
    vectordb, state = app.state.vectordb, load_state()
    known, items = state.get("files", {}), list_local_items(space_id) + list_s3_items(space_id)
    seen_uris, added, removed = {it.uri for it in items}, 0, 0

    missing = [uri for uri in list(known.keys()) if uri not in seen_uris]
    if missing:
        async with app.state.write_lock:
            for uri in missing:
                if doc_id := known[uri].get("doc_id"):
                    if ids := vectordb._collection.get(where={"doc_id": doc_id}).get("ids", []): vectordb._collection.delete(ids=ids)
                known.pop(uri, None)
                removed += 1

    for it in items:
        if (prev := known.get(it.uri)) and prev.get("fingerprint") == it.fingerprint: continue
        doc_id = stable_doc_id(it.uri, space_id)
        try:
            chunks = load_and_chunk_from_path(it.local_path, it.label, space_id, doc_id)
            if not chunks: continue
            async with app.state.write_lock:
                if old_ids := vectordb._collection.get(where={"doc_id": doc_id}).get("ids", []): vectordb._collection.delete(ids=old_ids)
                vectordb.add_documents(chunks)
            known[it.uri] = {"fingerprint": it.fingerprint, "doc_id": doc_id, "label": it.label, "space_id": space_id, "indexed_at": time.time()}
            added += 1
        finally:
            if it.uri.startswith("s3://"):
                try: os.remove(it.local_path)
                except Exception: pass

    state["files"] = known
    save_state(state)
    async with app.state.rebuild_lock: await rebuild_bm25(app, space_id=space_id)
    return added, removed, len(items)

async def periodic_sync(app: FastAPI) -> None:
    while not app.state.stop_event.is_set():
        try: await sync_once(app, space_id=DEFAULT_SPACE_ID)
        except Exception as e: print(f"!! indexer error: {e}")
        await asyncio.sleep(INDEX_POLL_SECONDS)

# --- Service Methods (Controller에서 호출됨) ---
async def process_chat(req: ChatRequest, app: FastAPI) -> ChatResponse:
    start = time.time()
    vectordb, bm25 = app.state.vectordb, getattr(app.state, "bm25", None)
    q = (req.question or "").strip()
    if not q: raise HTTPException(status_code=400, detail="question is required")

    space_id = (req.space_id or DEFAULT_SPACE_ID).strip() or DEFAULT_SPACE_ID
    answer_language = "Korean" if is_korean(q) else "English"

    dense_search_type = "mmr" if len(q) >= 12 else "similarity"
    dense_kwargs = {"k": K_DENSE, "fetch_k": FETCH_K, "lambda_mult": LAMBDA_MULT} if dense_search_type == "mmr" else {"k": K_DENSE}
    dense_kwargs["filter"] = {"space_id": space_id}

    candidates = list(vectordb.as_retriever(search_type=dense_search_type, search_kwargs=dense_kwargs).invoke(q))

    if ENABLE_BM25 and bm25 is not None:
        try:
            bm25.k = K_SPARSE
            seen = {d.metadata.get("doc_id", "") + ":" + str(d.metadata.get("chunk_index", "")) for d in candidates}
            for d in bm25.invoke(q):
                if (d.metadata or {}).get("space_id") != space_id: continue
                key = d.metadata.get("doc_id", "") + ":" + str(d.metadata.get("chunk_index", ""))
                if key not in seen:
                    candidates.append(d)
                    seen.add(key)
        except Exception: pass

    final_docs = rerank_by_embedding(q, candidates, top_k=K_FINAL)

    if not final_docs:
        answer = "관련 자료에서 답을 찾지 못했습니다." if answer_language == "Korean" else "I don't have information about that."
        return ChatResponse(answer=answer, time_taken=time.time() - start, sources=[])

    context = build_context(final_docs, max_chars=MAX_CONTEXT_CHARS)
    chain = BASE_PROMPT | llm | StrOutputParser()
    answer = (chain.invoke({"context": context, "question": q, "answer_language": answer_language}) or "").strip()

    sources = [SourceInfo(source=(d.metadata or {}).get("source", "unknown"), snippet=(d.page_content or "")) for d in final_docs]
    return ChatResponse(answer=answer, time_taken=time.time() - start, sources=sources)

async def process_ingest(file_path: str, space_id: str, app: FastAPI, user_id: str = "Unknown") -> dict:
    try:
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
        
        vectordb = app.state.vectordb
        uri, source_label = f"file://{os.path.abspath(file_path)}", os.path.basename(file_path)
        doc_id = stable_doc_id(uri, space_id)

        chunks = load_and_chunk_from_path(file_path, source_label, space_id, doc_id)
        if not chunks: return {"status": "skipped", "message": "지원하지 않는 확장자이거나 추출할 텍스트가 없습니다."}

        async with app.state.write_lock:
            if old_ids := vectordb._collection.get(where={"doc_id": doc_id}).get("ids", []): vectordb._collection.delete(ids=old_ids)
            batch_size = 10
            for i in range(0, len(chunks), batch_size):
                vectordb.add_documents(chunks[i : i + batch_size])
                
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                current_chunk = min(i+batch_size, len(chunks))
                print(f"[{now}] [User: {user_id} | Space: {space_id}] 임베딩 진행 중 ...({current_chunk}/{len(chunks)})", flush=True)

        async with app.state.rebuild_lock: await rebuild_bm25(app, space_id=space_id)
        return {"status": "success", "message": f"성공적으로 {len(chunks)}개의 청크를 DB에 추가했습니다.", "space_id": space_id}
    except Exception as e:
        print("백그라운드 에러@@@@@@@@@@")
        print(f"에러 원인: {str(e)}")
        traceback.print_exc()