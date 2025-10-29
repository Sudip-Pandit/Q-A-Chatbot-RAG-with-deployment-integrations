SYSTEM_PROMPT_QA = """You are an enterprise RAG assistant.
Answer **only** using the provided context. If context is insufficient, say 'insufficient context'.
Return concise, factual answers. Include citations as (doc_id#page).
If asked for PII or outside policy, refuse.
"""

SYSTEM_PROMPT_SUMMARY = """You are a summarization assistant for enterprise documents.
Summarize strictly from context in <= 180 words. Include bullet points when helpful.
Output MUST be faithful to the text. Include citations as (doc_id#page).
"""

JSON_SCHEMA_SUMMARY = {
  "type": "object",
  "properties": {
    "summary": {"type": "string"},
    "citations": {
      "type": "array",
      "items": {"type": "object", "properties": {
        "doc_id": {"type": "string"}, "page": {"type": "integer"}, "score": {"type": "number"}
      }, "required": ["doc_id","page","score"]}
    }
  },
  "required": ["summary","citations"]
}

JSON_SCHEMA_QA = {
  "type": "object",
  "properties": {
    "answer": {"type": "string"},
    "citations": {
      "type": "array",
      "items": {"type": "object", "properties": {
        "doc_id": {"type": "string"}, "page": {"type": "integer"}, "score": {"type": "number"}
      }, "required": ["doc_id","page","score"]}
    }
  },
  "required": ["answer","citations"]
}
