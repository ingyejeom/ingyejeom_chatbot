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

# colab에서만 주석처리
# if __name__ == "__main__":
#     import uvicorn
#     import nest_asyncio
#     from pyngrok import ngrok

#     # 코랩 환경 루프 충돌 방지
#     nest_asyncio.apply()

#     # 외부 접속용 ngrok 설정 (NGROK_TOKEN에는 본인의 토큰을 넣으세요)
#     NGROK_TOKEN = "3AW7vlF3pmyC2QQrhfmmvLc2IB8_2399FN1LYiPPybrHscyQx"
#     ngrok.set_auth_token(NGROK_TOKEN)
#     public_url = ngrok.connect(8000)
#     print(f"🌍 외부 접속 주소: {public_url}")

#     # 서버 실행 (reload=True는 코랩에서 에러를 유발할 수 있어 제거)
#     uvicorn.run(app, host="0.0.0.0", port=8000)