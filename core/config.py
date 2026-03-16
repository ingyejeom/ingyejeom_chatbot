import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

# 1) Configuration
DB_DIR = os.getenv("DB_DIR", "/content/chroma_db")
LOCAL_SOURCES = [p for p in os.getenv("LOCAL_SOURCES", "/content/data_storage").split(";") if p.strip()]
S3_BUCKET = os.getenv("S3_BUCKET", "").strip()
S3_PREFIXES = [p for p in os.getenv("S3_PREFIXES", "").split(";") if p.strip()]
S3_REGION = os.getenv("S3_REGION", "").strip()

INDEX_POLL_SECONDS = int(os.getenv("INDEX_POLL_SECONDS", "120"))
INDEX_STATE_PATH = os.getenv("INDEX_STATE_PATH", os.path.join(DB_DIR, "index_state.json"))
# TMP_DIR = os.getenv("TMP_DIR", "tmp_ingest")

K_DENSE = int(os.getenv("K_DENSE", "20"))
FETCH_K = int(os.getenv("FETCH_K", "40"))
LAMBDA_MULT = float(os.getenv("LAMBDA_MULT", "0.35"))

ENABLE_BM25 = os.getenv("ENABLE_BM25", "0").strip() == "1"
K_SPARSE = int(os.getenv("K_SPARSE", "20"))
K_FINAL = int(os.getenv("K_FINAL", "8"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "6500"))
DEFAULT_SPACE_ID = os.getenv("DEFAULT_SPACE_ID", "default")

# 2) Prompt
BASE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant.\n"
     "Never use your internal knowledge.\n"
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

# 3) LLM / Embeddings
LLM_MODEL = os.getenv("LLM_MODEL", "gemma2:9b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "mxbai-embed-large")

llm = ChatOllama(
    model=LLM_MODEL,
    temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
    num_predict=int(os.getenv("LLM_NUM_PREDICT", "240")),
    timeout=int(os.getenv("LLM_TIMEOUT", "25")),
    base_url="http://127.0.0.1:11434"
)
embedding_model = OllamaEmbeddings(
    model=EMBED_MODEL,
    base_url="http://127.0.0.1:11434"
)

# 4) Vector Store Builder
def get_vectorstore() -> Chroma:
    return Chroma(
        persist_directory=DB_DIR,
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"},
    )