import os, json, faiss, numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from app.config import settings

class Retriever:
    def __init__(self, index_dir: str = None):
        self.index_dir = index_dir or settings.index_dir
        self.index_path = os.path.join(self.index_dir, "faiss.index")
        self.meta_path = os.path.join(self.index_dir, "chunks.jsonl")
        self.model = SentenceTransformer(settings.embedding_model)
        self._load()

    def _load(self):
        self.index = faiss.read_index(self.index_path) if os.path.exists(self.index_path) else None
        self.meta = []
        if os.path.exists(self.meta_path):
            with open(self.meta_path, "r", encoding="utf-8") as f:
                for line in f:
                    self.meta.append(json.loads(line))
        # Prepare BM25
        self._bm25 = BM25Okapi([m["text"].split() for m in self.meta]) if self.meta else None

    def embed(self, texts: List[str]):
        return self.model.encode(texts, normalize_embeddings=True)

    def search(self, query: str, top_k: int = 8, filters: Dict[str, Any] = None):
        if not self.index or not self.meta:
            return []

        qv = self.embed([query]).astype('float32')
        D, I = self.index.search(qv, top_k*3)  # oversample
        dense_hits = [(i, float(D[0][rank])) for rank, i in enumerate(I[0]) if i != -1]

        # Keyword fallback & hybrid
        bm25_scores = []
        if self._bm25:
            bm25 = self._bm25.get_scores(query.split())
            bm25_scores = list(enumerate(bm25))

        # Merge dense + sparse (normalize and combine)
        sparse_dict = {i: s for i, s in bm25_scores}
        merged = []
        for i, dscore in dense_hits:
            sscore = sparse_dict.get(i, 0.0)
            score = (dscore / (abs(dscore) + 1e-9)) * 0.7 + (sscore / (max(1.0, max(sparse_dict.values(), default=1)))) * 0.3
            merged.append((i, score))

        # Basic filter by metadata if provided
        if filters:
            def ok(m):
                return all(m.get(k) == v for k, v in filters.items())
            merged = [(i,s) for (i,s) in merged if ok(self.meta[i])]

        # Deduplicate by doc_id+page; take top_k
        seen = set()
        results = []
        for i, s in sorted(merged, key=lambda x: x[1], reverse=True):
            key = (self.meta[i]["doc_id"], self.meta[i]["page"])
            if key in seen:
                continue
            seen.add(key)
            m = self.meta[i].copy()
            m["score"] = float(s)
            results.append(m)
            if len(results) >= top_k:
                break
        return results
