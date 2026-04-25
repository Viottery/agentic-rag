from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, HTTPException

from app.runtime.shell_runtime import (
    get_pending_shell_approval,
    list_pending_shell_approvals,
    reject_shell_approval,
    run_shell_command,
)

router = APIRouter(prefix="/shell", tags=["shell"])


@router.get("/approvals")
async def list_shell_approvals() -> dict[str, object]:
    return {"approvals": list_pending_shell_approvals()}


@router.get("/approvals/{approval_id}")
async def get_shell_approval(approval_id: str) -> dict[str, object]:
    approval = get_pending_shell_approval(approval_id)
    if approval is None:
        raise HTTPException(status_code=404, detail="shell approval not found or expired")
    return approval


@router.post("/approvals/{approval_id}/approve")
async def approve_shell_approval(approval_id: str) -> dict[str, object]:
    approval = get_pending_shell_approval(approval_id)
    if approval is None:
        raise HTTPException(status_code=404, detail="shell approval not found or expired")

    result = run_shell_command(
        str(approval["command"]),
        cwd=str(approval["cwd"]),
        approval_id=approval_id,
    )
    return {"approval": approval, "result": asdict(result)}


@router.post("/approvals/{approval_id}/reject")
async def reject_shell_approval_endpoint(approval_id: str) -> dict[str, object]:
    rejected = reject_shell_approval(approval_id)
    if not rejected:
        raise HTTPException(status_code=404, detail="shell approval not found or expired")
    return {"approval_id": approval_id, "rejected": True}
