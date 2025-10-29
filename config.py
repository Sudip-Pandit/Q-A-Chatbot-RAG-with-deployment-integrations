import os
from pydantic import BaseModel, Field

class Settings(BaseModel):
    provider: str = Field(default=os.getenv("PROVIDER", "openai"))
    embedding_model: str = Field(default=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    openai_model: str = Field(default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    anthropic_model: str = Field(default=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20240620"))
    bedrock_model: str = Field(default=os.getenv("BEDROCK_MODEL", "anthropic.claude-3-sonnet-20240229-v1:0"))
    index_dir: str = Field(default=os.getenv("INDEX_DIR", "./rag_index"))
    metadata_path: str = Field(default=os.getenv("METADATA_PATH", "./rag_index/metadata.jsonl"))
    max_input_tokens: int = Field(default=int(os.getenv("MAX_INPUT_TOKENS", "12000")))
    max_output_tokens: int = Field(default=int(os.getenv("MAX_OUTPUT_TOKENS", "450")))
    temperature: float = Field(default=float(os.getenv("TEMPERATURE", "0.1")))
    top_k_default: int = Field(default=int(os.getenv("TOP_K", "8")))

settings = Settings()
