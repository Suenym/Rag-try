from pydantic import BaseModel, Field
from typing import List, Union


class RelevantChunk(BaseModel):
    document_name: str = Field(min_length=1)
    page_number: int = Field(ge=1)


class SubmissionItem(BaseModel):
    question_id: int = Field(ge=0)
    relevant_chunks: List[RelevantChunk]
    answer: Union[str, int, float]


Submission = List[SubmissionItem]
