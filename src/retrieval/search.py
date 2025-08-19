from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import re
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

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

        self._bm25: Optional[BM25Okapi] = None
        self._bm25_corpus: Optional[List[List[str]]] = None

    @classmethod
    def from_dir(cls, index_dir: Path, device: Optional[str] = None) -> "Retriever":
        return cls(index_dir, device=device)

    def search(self, query: str, k: int = 5, overfetch: Optional[int] = None) -> List[Dict[str, Any]]:
        qv = self.emb.encode([query], is_query=True)[0]
        sims = self.embeddings @ qv

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

    # --- BM25 -----------------------------------------------------------------

    def _ensure_bm25(self) -> None:
        if self._bm25 is not None:
            return
        texts = self.df["text"].astype(str).tolist()
        tok_corpus = [re.findall(r"\w+", t.lower(), flags=re.UNICODE) for t in texts]
        self._bm25_corpus = tok_corpus
        self._bm25 = BM25Okapi(tok_corpus)

    def _bm25_search(self, query: str, k: int) -> List[Dict[str, Any]]:
        self._ensure_bm25()
        assert self._bm25 is not None
        tokens = re.findall(r"\w+", query.lower(), flags=re.UNICODE)
        scores = self._bm25.get_scores(tokens)
        topn = int(min(k, len(scores)))
        idxs = np.argpartition(-scores, topn - 1)[:topn]
        idxs = idxs[np.argsort(-scores[idxs])]
        hits: List[Dict[str, Any]] = []
        for i in idxs:
            row = self.df.iloc[int(i)]
            preview = str(row.get("text", ""))[:200]
            hits.append(
                {
                    "score": float(scores[int(i)]),
                    "doc_name": str(row.get("doc_name", "")),
                    "page_number": int(row.get("page_number", -1)),
                    "kind": str(row.get("kind", "page")),
                    "preview": preview,
                    "row_index": int(i),
                }
            )
        return hits

    def search_hybrid(self, query: str, k: int = 5, rrf_k: int = 60) -> List[Dict[str, Any]]:
        dense_hits = self.search(query, k=k)
        bm25_hits = self._bm25_search(query, k)

        scores: Dict[int, float] = {}
        merged: Dict[int, Dict[str, Any]] = {}
        for rank, h in enumerate(dense_hits, start=1):
            key = h["row_index"]
            merged[key] = h
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
        for rank, h in enumerate(bm25_hits, start=1):
            key = h["row_index"]
            if key not in merged:
                merged[key] = h
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)

        keys_sorted = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:k]
        hits: List[Dict[str, Any]] = []
        for key in keys_sorted:
            h = merged[key].copy()
            h["score"] = float(scores[key])
            hits.append(h)
        return hits
