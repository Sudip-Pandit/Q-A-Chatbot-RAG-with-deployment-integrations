import re

def clean_text(s: str) -> str:
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def split_into_chunks(text: str, size: int=900, overlap: int=120):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+size]
        chunks.append(' '.join(chunk))
        i += size - overlap if size > overlap else size
    return chunks
