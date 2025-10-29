import json, os
from typing import Iterator, Dict, Any

class MetadataStore:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def write_lines(self, items: Iterator[Dict[str, Any]]):
        with open(self.path, "a", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
