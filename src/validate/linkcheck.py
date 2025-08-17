from typing import Any


def check_relevant_chunks(item: dict, allow_empty: bool = False) -> list[str]:
    errs = []
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
