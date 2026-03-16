# app/schemas.py
from pydantic import BaseModel

class QuestionRequest(BaseModel):
    question: str
    system_prompt: str = "You are a helpful assistant. Answer concisely."

class AnswerResponse(BaseModel):
    question: str
    answer: str
    model: str
    latency_ms: float