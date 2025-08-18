from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


def build_index(
    index_dir: str,
    model: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: str = "cpu",
    batch_size: int = 64,
) -> np.ndarray:
    """Encode chunks and save embeddings (+optional FAISS index)."""
    index_path = Path(index_dir)
    index_path.mkdir(parents=True, exist_ok=True)

    chunks = pd.read_parquet(index_path / "chunks.parquet")
    texts = chunks["text"].astype(str).tolist()

    encoder = SentenceTransformer(model, device=device)
    emb = encoder.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    emb = np.asarray(emb, dtype="float32")
    np.save(index_path / "embeddings.npy", emb)

    meta = {"model": model, "dim": int(emb.shape[1])}
    (index_path / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    try:
        import faiss  # type: ignore

        faiss_index = faiss.IndexFlatIP(emb.shape[1])
        faiss_index.add(emb)
        faiss.write_index(faiss_index, str(index_path / "index.faiss"))
    except Exception:
        pass

    return emb

