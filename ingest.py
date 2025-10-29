import os, argparse, json, faiss, numpy as np
from pdfminer.high_level import extract_text
from app.utils.text import clean_text, split_into_chunks
from sentence_transformers import SentenceTransformer
from app.storage.metadata_store import MetadataStore

def read_file(path: str) -> str:
    if path.lower().endswith(".pdf"):
        return extract_text(path) or ""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def build_index(input_dir: str, index_dir: str, chunk_size: int=900, chunk_overlap: int=120, model_name: str="sentence-transformers/all-MiniLM-L6-v2"):
    os.makedirs(index_dir, exist_ok=True)
    chunks, metas = [], []
    model = SentenceTransformer(model_name)

    for root, _, files in os.walk(input_dir):
        for fn in files:
            if not fn.lower().endswith((".pdf",".txt",".md")):
                continue
            path = os.path.join(root, fn)
            doc_id = os.path.relpath(path, input_dir).replace("\\","/")
            text = clean_text(read_file(path))
            if not text:
                continue
            parts = split_into_chunks(text, size=chunk_size, overlap=chunk_overlap)
            for page, ch in enumerate(parts, start=1):
                chunks.append(ch)
                metas.append({"doc_id": doc_id, "page": page, "text": ch})

    if not chunks:
        raise SystemExit("No ingestible documents found.")

    X = model.encode(chunks, normalize_embeddings=True).astype("float32")
    index = faiss.IndexFlatIP(X.shape[1])
    index.add(X)

    faiss.write_index(index, os.path.join(index_dir, "faiss.index"))
    ms = MetadataStore(os.path.join(index_dir, "chunks.jsonl"))
    ms.write_lines(metas)
    print(f"Ingested {len(chunks)} chunks from {input_dir} into {index_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--index_dir", required=True)
    ap.add_argument("--chunk_size", type=int, default=900)
    ap.add_argument("--chunk_overlap", type=int, default=120)
    args = ap.parse_args()
    build_index(args.input_dir, args.index_dir, args.chunk_size, args.chunk_overlap)
