from __future__ import annotations

from pathlib import Path
from typing import List, Optional
import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """Единая обёртка над ST-моделями с e5-префиксами для запросов/пассажей."""
    def __init__(self, model_name: str, device: Optional[str] = None):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device if device else None)
        self.dim = int(self.model.get_sentence_embedding_dimension())

    def _maybe_prefix(self, text: str, is_query: bool) -> str:
        name = self.model_name.lower()
        # e5 семейство — строго требуют префиксы
        if "e5" in name:
            return f"{'query' if is_query else 'passage'}: {text}"
        # gte-модели обычно без префиксов
        return text

    def encode(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        texts = [self._maybe_prefix(t, is_query=is_query) for t in texts]
        vecs = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vecs)


def build_index(index_dir: str | Path, model: str, device: Optional[str] = None) -> None:
    """Читает chunks.parquet, строит embeddings.npy и meta.json"""
    index_path = Path(index_dir)
    chunks_path = index_path / "chunks.parquet"
    if not chunks_path.exists():
        raise FileNotFoundError(f"not found: {chunks_path}")

    df = pd.read_parquet(chunks_path)
    if "text" not in df.columns:
        raise ValueError("В chunks.parquet нет колонки 'text'")

    emb = EmbeddingModel(model, device=device)
    vecs = emb.encode(df["text"].astype(str).tolist(), is_query=False)

    # float32 для экономии места и скорости dot
    np.save(index_path / "embeddings.npy", vecs.astype("float32"))

    meta = {"model_name": model, "dim": emb.dim}
    (index_path / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
