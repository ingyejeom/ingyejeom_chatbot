from fastapi import APIRouter, Request
from fastapi import UploadFile, File, Form, BackgroundTasks
from schemas.dto import ChatRequest, ChatResponse, IngestRequest
from services.chatbot_service import process_chat, process_ingest

# 스프링의 @RestController, @RequestMapping 역할
router = APIRouter()

@router.post("/chatbot", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request):
    # 서비스 계층으로 넘겨주기
    return await process_chat(req, request.app)

@router.post("/ingest")
async def ingest_file(request: Request, background_tasks: BackgroundTasks, file: UploadFile = File(...), space_id: str = Form(...), user_id: str = Form("Unknown")):
    save_path = f"/content/data_storage/{file.filename}"
    with open(save_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)

    background_tasks.add_task(process_ingest, save_path, space_id, request.app, user_id)

    return {"status" : "processing", "message" : "파일 접수 완료. 백그라운드에서 학습 중입니다."}