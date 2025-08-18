from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


class SearchIndex:
    def __init__(self, index_dir: str, device: str = "cpu"):
        self.index_path = Path(index_dir)
        self.chunks = pd.read_parquet(self.index_path / "chunks.parquet")
        self.embeddings = np.load(self.index_path / "embeddings.npy")
        meta = json.loads((self.index_path / "meta.json").read_text(encoding="utf-8"))
        self.model_name = meta.get("model", "sentence-transformers/all-MiniLM-L6-v2")
        self.encoder = SentenceTransformer(self.model_name, device=device)

        self.use_faiss = False
        faiss_path = self.index_path / "index.faiss"
        if faiss_path.exists():
            try:
                import faiss  # type: ignore

                self.faiss_index = faiss.read_index(str(faiss_path))
                self.use_faiss = True
            except Exception:
                self.faiss_index = None
        else:
            self.faiss_index = None

    def search(self, query: str, k: int = 5) -> pd.DataFrame:
        q_emb = self.encoder.encode([query], normalize_embeddings=True)
        q_emb = np.asarray(q_emb, dtype="float32")

        if self.use_faiss and self.faiss_index is not None:
            D, I = self.faiss_index.search(q_emb, k)
            scores = D[0]
            idxs = I[0]
        else:
            scores = self.embeddings @ q_emb[0]
            idxs = np.argsort(-scores)[:k]
            scores = scores[idxs]

        res = self.chunks.iloc[idxs].copy()
        res["score"] = scores
        return res.reset_index(drop=True)

