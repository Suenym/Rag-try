from typing import Any, List, Dict
import re


def check_relevant_chunks(item: dict, allow_empty: bool = False) -> list[str]:
    errs: list[str] = []
    chunks = item.get("relevant_chunks", [])
    if not allow_empty and (not isinstance(chunks, list) or len(chunks) == 0):
        errs.append("relevant_chunks must be non-empty list")
        return errs
    for ch in chunks:
        if not isinstance(ch, dict):
            errs.append("each relevant_chunk must be object")
            continue
        if not ch.get("document_name"):
            errs.append("document_name is required and non-empty")
        pn = ch.get("page_number")
        if not isinstance(pn, int) or pn < 1:
            errs.append("page_number must be integer >= 1")
    return errs


def norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").replace("\u00A0", " ")).strip().lower()


def number_variants(x) -> List[str]:
    s = str(x)
    if isinstance(x, float) and s.endswith(".0"):
        s = s[:-2]
    raw = s.replace("\u00A0", " ").replace(" ", "")
    out = {raw, raw.replace(".", ","), raw.replace(",", ".")}
    if raw.isdigit() and len(raw) > 4:
        with_spaces = " ".join([raw[::-1][i:i+3] for i in range(0, len(raw), 3)])[::-1]
        out.update({with_spaces, with_spaces.replace(" ", "\u00A0")})
    return list(out)

def linkcheck(answer_value, atype: str, hits: List[Dict], topk: int = 3) -> int:
    pool = hits[:topk]
    if atype in ("int","float"):
        variants = number_variants(answer_value)
        q = ""  # передавайте сюда исходный вопрос при желании усилить проверку
        for i, h in enumerate(pool):
            text = (h.get("preview") or "")
            low = text.lower()
            for v in variants:
                pos = low.find(v.lower())
                if v and pos >= 0:
                    # рядом должно быть одно из смысловых слов (очень простой вариант)
                    ctx = low[max(0,pos-48):pos+len(v)+48]
                    if re.search(r"(тонн|выброс|частот|рейс|работник|дивиденд|убыт|капитал|рентабел|бухгалтер|директор)", ctx):
                        return i
        return -1
    else:
        tgt = norm_space(str(answer_value))
        for i, h in enumerate(pool):
            if tgt and tgt in norm_space(h.get("preview") or ""):
                return i
        return -1
