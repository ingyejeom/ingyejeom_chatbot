from typing import List, Optional
from pydantic import BaseModel

class ChatRequest(BaseModel):
    question: str
    space_id: Optional[str] = None

class SourceInfo(BaseModel):
    source: str
    #snippet: str
    page: Optional[int] = None


class ChatResponse(BaseModel):
    answer: str
    #time_taken: float
    sources: List[SourceInfo]

class IngestRequest(BaseModel):
    space_id: str
    file_path: str
