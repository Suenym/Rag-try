# --- GeminiReranker (короткая версия) ---
import os, re, json
from typing import Any, Dict, List
import google.generativeai as genai

def _minmax(xs):
    if not xs: return []
    lo, hi = min(xs), max(xs)
    if hi - lo < 1e-12: return [0.5]*len(xs)
    return [(x-lo)/(hi-lo) for x in xs]

class GeminiReranker:
    def __init__(self, model="gemini-2.5-flash", max_chars=600, alpha=0.25):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key: raise RuntimeError("GEMINI_API_KEY is not set")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)
        self.max_chars = max_chars
        self.alpha = float(alpha)

    def rerank(self, query: str, hits: List[Dict[str, Any]], top_k: int):
        if not hits: return []
        snippets = [{"id": i, "text": str(h.get("preview",""))[:self.max_chars]}
                    for i, h in enumerate(hits)]
        prompt = "\n".join([
            "You are a retrieval reranker.",
            "Given the query and snippets, assign a relevance score 0-100 to EACH snippet.",
            'Return ONLY JSON: {"scores":[{"id":int,"score":float}, ...]}',
            f"Query: {query}"
        ] + [f"Snippet {s['id']}: {s['text']}" for s in snippets])

        score_map: Dict[int,float] = {}
        try:
            resp = self.model.generate_content(prompt)
            m = re.search(r"\{.*\}", getattr(resp, "text", "") or "", re.DOTALL)
            if m:
                data = json.loads(m.group(0))
                for it in data.get("scores", []):
                    score_map[int(it["id"])] = float(it["score"])
        except Exception:
            return hits[:top_k]  # на сбое — оставляем HF порядок

        hf = [float(h.get("rerank_score", h.get("score", 0.0))) for h in hits]
        hf_norm = _minmax(hf)
        llm_norm = _minmax([score_map.get(i, 0.0)/100.0 for i in range(len(hits))])

        for i, h in enumerate(hits):
            h["rerank_score"] = (1-self.alpha)*hf_norm[i] + self.alpha*llm_norm[i]
        hits.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
        return hits[:top_k]
