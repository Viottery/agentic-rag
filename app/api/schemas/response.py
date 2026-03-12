from pydantic import BaseModel, Field


class ChatResponse(BaseModel):
    answer: str
    evidence: list[str] = Field(default_factory=list)
    steps: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    status: str = "ok"
    error: str | None = None