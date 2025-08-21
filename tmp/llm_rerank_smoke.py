import os
from src.retrieval.llm_rerank import GeminiReranker

assert os.getenv("GEMINI_API_KEY"), "Set GEMINI_API_KEY first"

rr = GeminiReranker(model="gemini-2.5-flash")  # можно сменить на то, что показал пинг

query = "Какая еженедельная частота рейсов было в 2024 году у компании FlyArystan?"
hits = [
  {"doc_name":"airap_2024_rus.pdf","page_number":11,
   "preview":"... В 2024 году еженедельная частота рейсов увеличилась до 18 ..."},
  {"doc_name":"kztkp_2024_rus.pdf","page_number":101,
   "preview":"... 2024 ... 18257 работников ..."},
  {"doc_name":"kztkf5m1_2025_cons_rus_pdf.pdf","page_number":24,
   "preview":"... Обязательный размер дивидендов составляет 300 тенге на одну акцию ..."},
]

res = rr.rerank(query, hits, top_k=3)
for i, h in enumerate(res, 1):
    print(i, h.get("doc_name"), h.get("page_number"), round(h.get("rerank_score",0),3),
          "", (h.get("preview","")[:80]).replace("\n"," "))
