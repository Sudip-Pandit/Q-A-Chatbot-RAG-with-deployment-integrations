from fastapi import APIRouter, HTTPException
from app.models.schemas import ChatRequest, ChatResponse, Citation
from app.config import settings
from app.services.retrieval import Retriever
from app.services.providers import generate_completion
from app.utils.observability import new_trace_id, log_event

router = APIRouter()
retriever = Retriever(index_dir=settings.index_dir)

@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    trace_id = new_trace_id()
    k = req.top_k or settings.top_k_default
    hits = retriever.search(req.query, top_k=k, filters=req.filters)

    if not hits:
        raise HTTPException(status_code=404, detail="No context available. Please ingest documents.")

    out = generate_completion(settings.provider, "qa", req.query, hits, schema=req.schema)
    text = out["text"]

    # Simple citation extraction pattern (fallback to top hits)
    cits = []
    for h in hits[:min(5, len(hits))]:
        cits.append(Citation(doc_id=h["doc_id"], page=h["page"], score=h["score"]))

    log_event("chat_complete", provider=settings.provider, trace_id=trace_id, citations=len(cits))
    return ChatResponse(answer=text, citations=cits, meta={"trace_id": trace_id, "provider": settings.provider})
