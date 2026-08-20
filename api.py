"""FastAPI programmatic endpoint, as an alternative to the Streamlit UI.

Run:
    uvicorn api:app --reload --port 8000

POST /chat  {"query": "...", "role": "finance", "history": [{"role":"user","content":"..."}]}
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.access_control import get_allowed_departments
from src.config import Settings
from src.rag_pipeline import RAGPipeline

app = FastAPI(title="Northwind Knowledge Assistant API")

_settings = Settings()
_pipeline = RAGPipeline(_settings)


class Turn(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    query: str
    role: str = "admin"
    history: list[Turn] = []


class Citation(BaseModel):
    index: int
    title: str
    section: str
    source_path: str
    effective_date: str | None
    version: str


class ChatResponse(BaseModel):
    answer: str
    ambiguous: bool
    clarification_options: list[str]
    insufficient_evidence: bool
    confidence: float
    citations: list[Citation]
    timings_ms: dict[str, float]


@app.get("/health")
def health():
    return {"status": "ok", "backend": _settings.backend, "profile": _settings.profile_name}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    try:
        allowed = get_allowed_departments(req.role)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    history = [{"role": t.role, "content": t.content} for t in req.history]
    result = _pipeline.ask(req.query, allowed, history)

    if result.ambiguous:
        return ChatResponse(
            answer="",
            ambiguous=True,
            clarification_options=result.clarification_options,
            insufficient_evidence=False,
            confidence=0.0,
            citations=[],
            timings_ms=result.timings_ms,
        )

    answer = result.answer
    return ChatResponse(
        answer=answer.text,
        ambiguous=False,
        clarification_options=[],
        insufficient_evidence=answer.insufficient_evidence,
        confidence=answer.confidence,
        citations=[Citation(**vars(c)) for c in answer.citations],
        timings_ms=result.timings_ms,
    )
