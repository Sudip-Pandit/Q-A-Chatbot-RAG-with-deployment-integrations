# Enterprise Q&A Chatbot (RAG) — FastAPI + FAISS + OpenAI/Anthropic/Bedrock

**Option B**: Enterprise document Q&A chatbot with secure RAG, provider abstraction, JSON-schema validation, and observability.

## Features
- Document ingestion (PDF/TXT/MD) → chunking → embeddings → FAISS index + metadata store.
- Hybrid retrieval (BM25 via simple keyword fallback + dense vectors) + MMR re-ranking.
- Summarization/Q&A via provider router: OpenAI / Anthropic / AWS Bedrock (SageMaker-ready).
- JSON-schema validated responses, citations, refusal when insufficient context.
- Observability: structured logs, request tracing IDs, latency + token/cost accounting hooks.
- Config via environment; Dockerfile provided.

## Quickstart
```bash
# 1) Install
pip install -r requirements.txt

# 2) Set environment (choose a provider)
export PROVIDER=openai               # or 'anthropic' or 'bedrock'
export OPENAI_API_KEY=sk-...         # if PROVIDER=openai
# export ANTHROPIC_API_KEY=...       # if PROVIDER=anthropic
# AWS creds for bedrock via environment/instance profile

# 3) Build index
python -m app.ingest.ingest --input_dir ./sample_docs --index_dir ./rag_index

# 4) Run API
uvicorn app.main:app --reload --port 8080

# 5) Test chat
curl -s -X POST http://localhost:8080/chat -H "Content-Type: application/json" -d '{
  "query": "Summarize the eligibility criteria",
  "top_k": 8,
  "schema": "summary_v1"
}'
```

### Project Layout
```
app/
  main.py
  config.py
  models/schemas.py
  routers/chat.py
  services/providers.py
  services/retrieval.py
  services/ranker.py
  ingest/ingest.py
  utils/observability.py
  utils/text.py
  security/policy.py
  storage/metadata_store.py
requirements.txt
Dockerfile
sample_docs/ (add your PDFs/TXTs)
tests/
```

## Notes
- FAISS index stores embeddings; metadata stored as JSON alongside.
- Replace models/IDs with your enterprise choices (e.g., Bedrock Claude/LLama 3.1 via Bedrock).
