Manual test steps:
1) Build index:
   python -m app.ingest.ingest --input_dir ./sample_docs --index_dir ./rag_index

2) Run API:
   uvicorn app.main:app --reload --port 8080

3) Query:
   curl -s -X POST http://localhost:8080/chat -H "Content-Type: application/json" -d '{
     "query": "What are the eligibility rules?",
     "top_k": 5,
     "schema": "qa_v1"
   }' | jq .
