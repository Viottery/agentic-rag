from contextlib import asynccontextmanager
import asyncio

from fastapi import FastAPI

from app.agent.services.local_rag_socket_service import get_local_rag_socket_service
from app.api.routes.chat import router as chat_router
from app.api.routes.health import router as health_router
from app.api.routes.shell import router as shell_router
from app.runtime.conversation_store import get_conversation_store


@asynccontextmanager
async def lifespan(_app: FastAPI):
    await asyncio.to_thread(get_conversation_store().initialize)
    rag_service = get_local_rag_socket_service()
    await rag_service.start()
    try:
        yield
    finally:
        await rag_service.stop()


app = FastAPI(title="Agentic RAG", debug=True, lifespan=lifespan)

app.include_router(health_router)
app.include_router(chat_router)
app.include_router(shell_router)
