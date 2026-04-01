from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.agent.services.local_rag_socket_service import get_local_rag_socket_service
from app.api.routes.chat import router as chat_router
from app.api.routes.health import router as health_router


@asynccontextmanager
async def lifespan(_app: FastAPI):
    rag_service = get_local_rag_socket_service()
    await rag_service.start()
    try:
        yield
    finally:
        await rag_service.stop()


app = FastAPI(title="Agentic RAG", debug=True, lifespan=lifespan)

app.include_router(health_router)
app.include_router(chat_router)
