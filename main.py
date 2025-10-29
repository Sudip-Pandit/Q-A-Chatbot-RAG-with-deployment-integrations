from fastapi import FastAPI
from app.routers.chat import router as chat_router

app = FastAPI(title="Enterprise Q&A Chatbot (RAG)")

@app.get("/healthz")
def healthz():
    return {"status":"ok"}

app.include_router(chat_router, tags=["chat"])
