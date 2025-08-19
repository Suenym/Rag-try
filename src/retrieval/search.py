from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import numpy as np
import pandas as pd

from .embed import EmbeddingModel


class Retriever:
    def __init__(self, index_dir: Path, device: Optional[str] = None):
        self.index_dir = Path(index_dir)

        meta = json.loads((self.index_dir / "meta.json").read_text(encoding="utf-8"))
        # поддержим старое поле 'model'
        self.model_name = meta.get("model_name") or meta.get("model")
        if not self.model_name:
            raise KeyError("meta.json должен содержать 'model_name'")

        self.emb = EmbeddingModel(self.model_name, device=device)

        self.df = pd.read_parquet(self.index_dir / "chunks.parquet")
        self.embeddings = np.load(self.index_dir / "embeddings.npy").astype("float32")
        if self.embeddings.ndim != 2:
            raise ValueError("embeddings.npy должен быть матрицей (N, D)")

    @classmethod
    def from_dir(cls, index_dir: Path, device: Optional[str] = None) -> "Retriever":
        return cls(index_dir, device=device)

    def search(self, query: str, k: int = 5, overfetch: Optional[int] = None) -> List[Dict[str, Any]]:
        # embed запроса c e5-префиксом
        qv = self.emb.encode([query], is_query=True)[0]  # (D,)
        # cos sim = dot т.к. векторы уже нормированы
        sims = self.embeddings @ qv  # (N,)

        topn = int(min(overfetch or k, sims.shape[0]))
        idxs = np.argpartition(-sims, topn - 1)[:topn]
        idxs = idxs[np.argsort(-sims[idxs])]

        hits: List[Dict[str, Any]] = []
        for i in idxs:
            row = self.df.iloc[int(i)]
            preview = str(row.get("text", ""))[:200]
            hits.append(
                {
                    "score": float(sims[int(i)]),
                    "doc_name": str(row.get("doc_name", "")),
                    "page_number": int(row.get("page_number", -1)),
                    "kind": str(row.get("kind", "page")),
                    "preview": preview,
                    "row_index": int(i),
                }
            )
        return hits[:topn]
