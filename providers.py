import os, json, time
from typing import Dict, Any, List
from app.config import settings
from app.security.policy import SYSTEM_PROMPT_QA, SYSTEM_PROMPT_SUMMARY
from app.utils.observability import log_event

# Optional SDKs
import openai
import anthropic
import boto3

def _render_context(chunks: List[dict]) -> str:
    parts = []
    for c in chunks:
        parts.append(f"[{c['doc_id']}#p{c['page']}] {c['text']}")
    return "\n\n".join(parts[:12])

def generate_completion(provider: str, task: str, query: str, chunks: List[dict], schema: str="qa_v1") -> Dict[str, Any]:
    system = SYSTEM_PROMPT_QA if schema == "qa_v1" else SYSTEM_PROMPT_SUMMARY
    context = _render_context(chunks)
    user_prompt = f"Query: {query}\n\nContext:\n{context}\n\nRespond with concise factual text and list citations as (doc_id#page)."

    if provider == "openai":
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        resp = client.chat.completions.create(
            model=settings.openai_model,
            temperature=settings.temperature,
            max_tokens=settings.max_output_tokens,
            messages=[
                {"role":"system","content":system},
                {"role":"user","content":user_prompt}
            ]
        )
        txt = resp.choices[0].message.content.strip()
        usage = getattr(resp, "usage", None)
        log_event("llm_usage", provider="openai", usage=str(usage))
        return {"text": txt, "usage": usage and usage.model_dump() or {}}

    if provider == "anthropic":
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        msg = client.messages.create(
            model=settings.anthropic_model,
            temperature=settings.temperature,
            max_tokens=settings.max_output_tokens,
            system=system,
            messages=[{"role":"user","content": user_prompt}]
        )
        txt = "".join([b.text for b in msg.content if getattr(b, "type", "")=="text"]).strip()
        return {"text": txt, "usage": {"input": msg.usage.input_tokens, "output": msg.usage.output_tokens}}

    if provider == "bedrock":
        br = boto3.client("bedrock-runtime")
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": settings.max_output_tokens,
            "temperature": settings.temperature,
            "messages": [
                {"role":"user","content": [{"type":"text","text": f"{system}\n\n{user_prompt}"}]}
            ]
        }
        resp = br.invoke_model(
            modelId=settings.bedrock_model,
            body=json.dumps(body)
        )
        payload = json.loads(resp["body"].read())
        txt = "".join([b.get("text","") for b in payload.get("content", [])])
        return {"text": txt, "usage": payload.get("usage", {})}

    raise ValueError(f"Unknown provider: {provider}")
