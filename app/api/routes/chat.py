from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.api.schemas.request import ChatRequest
from app.runtime.conversation_queue import get_conversation_queue_manager

router = APIRouter(tags=["chat"])


@router.post("/chat")
async def chat(req: ChatRequest) -> dict:
    """
    Chat API 入口。

    当前支持两种模式：
    - `mode=wait`：按现有语义等待本轮完成，并返回完整执行结果
    - `mode=background`：异步排队执行，立即返回 job 元数据

    同一 `conversation_id` 内的请求会通过队列串行执行，
    不同 conversation 可以并发处理。
    """
    manager = get_conversation_queue_manager()
    job = await manager.submit(
        question=req.question,
        conversation_id=req.resolved_conversation_id(),
        mode=req.mode,
    )

    if req.mode == "background":
        return job.to_public_dict()

    if job.status == "failed":
        raise HTTPException(status_code=500, detail=job.error or "chat execution failed")

    return job.result or {
        "conversation_id": job.conversation_id,
        "turn_id": job.turn_id,
        "job_id": job.job_id,
        "status": job.status,
        "error": job.error,
    }


@router.get("/chat/jobs/{job_id}")
async def get_chat_job(job_id: str) -> dict:
    """
    查询后台 chat job 状态。
    """
    manager = get_conversation_queue_manager()
    job = manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job.to_public_dict()
