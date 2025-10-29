from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=2)
    top_k: Optional[int] = Field(default=None, ge=1, le=20)
    schema: Optional[Literal["summary_v1", "qa_v1"]] = "qa_v1"
    filters: Optional[Dict[str, Any]] = None

class Citation(BaseModel):
    doc_id: str
    page: int
    score: float

class ChatResponse(BaseModel):
    answer: str
    citations: List[Citation]
    meta: dict

class IngestRequest(BaseModel):
    input_dir: str
    index_dir: str
    chunk_size: int = 900
    chunk_overlap: int = 120
