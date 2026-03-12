from fastapi import APIRouter

router = APIRouter(tags=["chat"])


@router.post("/chat")
def chat(payload: dict) -> dict:
    question = payload.get("question", "")
    return {
        "answer": f"收到问题：{question}",
        "status": "ok",
    }