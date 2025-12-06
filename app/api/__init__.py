from fastapi import APIRouter

from .routes.documents import router as documents_router
from .routes.query import router as query_router

app = APIRouter()

# Document Management
app.include_router(documents_router, prefix="/documents", tags=["documents"])

# RAG Query
app.include_router(query_router, prefix="/query", tags=["query"])
