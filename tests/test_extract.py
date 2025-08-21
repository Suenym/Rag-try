from src.reader.extract import cleanup_hyphenation, extract_answer_from_hits


def test_cleanup_hyphenation_basic():
    text = "Ива-\nнов\nИван"
    assert cleanup_hyphenation(text) == "Иванов Иван"


def test_extract_answer_with_hyphenated_fio():
    question = "Кто подписал отчёт?"
    hits = [{"preview": "Отчёт подписал Ива-\nнов Иван Иванович."}]
    ans = extract_answer_from_hits(question, hits, topk=1)
    assert ans.value == "Иванов Иван Иванович"

