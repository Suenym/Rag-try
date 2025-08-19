from __future__ import annotations

from typing import Any, Dict, List, Optional
from sentence_transformers import CrossEncoder


class Reranker:
    """Переранжировка кросс-энкодером (например, Jina reranker v2)."""

    def __init__(self, model_name: str, device: Optional[str] = None):
        self.model_name = model_name
        # На MPS CrossEncoder нестабилен — уводим на CPU
        ce_device = device if device != "mps" else None
        # У Jina — кастомный код
        trust_remote = "jinaai/" in model_name or "reranker-v2" in model_name
        self.model = CrossEncoder(
            model_name,
            device=ce_device,
            trust_remote_code=trust_remote,
        )

    def rerank(self, query: str, hits: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        if not hits:
            return []
        pairs = [[query, h["preview"]] for h in hits]
        scores = self.model.predict(pairs, apply_softmax=False, convert_to_numpy=True, show_progress_bar=False)
        for h, s in zip(hits, scores):
            h["rerank_score"] = float(s)
        hits.sort(key=lambda x: x.get("rerank_score", x["score"]), reverse=True)
        return hits[:top_k]
