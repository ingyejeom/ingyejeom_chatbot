from fastapi import APIRouter, Request
from fastapi import UploadFile, File, Form
from schemas.dto import ChatRequest, ChatResponse, IngestRequest
from services.chatbot_service import process_chat, process_ingest

# 스프링의 @RestController, @RequestMapping 역할
router = APIRouter()

@router.post("/chatbot", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request):
    # 서비스 계층으로 넘겨주기
    return await process_chat(req, request.app)

@router.post("/ingest")
async def ingest_file(request: Request, file: UploadFile = File(...), space_id: str = Form(...)):
    save_path = f"/app/data_storage/{file.filename}"
    with open(save_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)
    
    return await process_ingest(save_path, space_id, request.app)