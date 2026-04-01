from typing import Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    conversation_id: str | None = None
    session_id: str | None = None
    mode: Literal["wait", "background"] = "wait"
    debug: bool = False

    def resolved_conversation_id(self) -> str | None:
        for candidate in (self.conversation_id, self.session_id):
            cleaned = (candidate or "").strip()
            if cleaned:
                return cleaned
        return None
