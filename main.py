import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from core.config import DB_DIR, ensure_dir, get_vectorstore
# from services.chatbot_service import sync_once, periodic_sync, DEFAULT_SPACE_ID
from api.routes import router

# 스프링의 @PostConstruct 처럼 서버 켜질 때/꺼질 때 실행될 로직
@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_dir(DB_DIR)
    # ensure_dir(TMP_DIR)

    app.state.vectordb = get_vectorstore()
    app.state.bm25 = None
    app.state.write_lock = asyncio.Lock()
    app.state.rebuild_lock = asyncio.Lock()
    app.state.stop_event = asyncio.Event()

    # (기존처럼 자동 인덱싱은 일단 주석 처리)
    # await sync_once(app, space_id=DEFAULT_SPACE_ID)
    # app.state.sync_task = asyncio.create_task(periodic_sync(app))

    print(">> Server Started (Layered Architecture)")
    yield

    app.state.stop_event.set()
    print(">> Server Shutdown")

app = FastAPI(title="RAG API (Layered)", lifespan=lifespan)

# 컨트롤러들을 메인 앱에 등록
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    # 💡 주의: 파일이 쪼개졌으므로 모듈 이름을 main:app으로 실행해야 합니다!
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)