from threading import Thread

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1 import chat, search, recommend, analyze, map_tools, interactions
from app.core.config import settings
from app.core.logging import logger
from app.services.rag_service import get_rag_service

app = FastAPI(
    title=settings.project_name,
    description="Conversational AI API for Real Estate Recommendations (Clean Architecture)",
    version=settings.version,
)

# CORS configuration
def _cors_origins():
    raw = (settings.cors_allowed_origins or "*").strip()
    if raw == "*":
        return ["*"]
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


cors_origins = _cors_origins()
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=cors_origins != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Attach Modular API Routers
app.include_router(chat.router, prefix="/api/v1", tags=["Chat"])
app.include_router(search.router, prefix="/api/v1", tags=["Search"])
app.include_router(recommend.router, prefix="/api/v1", tags=["Recommend"])
app.include_router(analyze.router, prefix="/api/v1", tags=["Analyze"])
app.include_router(map_tools.router, prefix="/api/v1", tags=["Map"])
app.include_router(interactions.router, prefix="/api/v1", tags=["Interactions"])

@app.get("/health", tags=["System"])
async def health_check():
    """System health checkpoint."""
    return {
        "status": "healthy",
        "version": settings.version
    }


@app.on_event("startup")
def warm_rag_service():
    """Preload the heavy retrieval stack after Uvicorn starts accepting traffic."""
    def _warm():
        try:
            get_rag_service()
            logger.info("RAG service warmed successfully.")
        except Exception as e:
            logger.error(f"RAG service warmup failed: {e}")

    Thread(target=_warm, daemon=True).start()

@app.get("/", tags=["System"])
async def root():
    """API Root standard greeting."""
    return {
        "message": "Welcome to Aqar-AI Intelligence API!",
        "docs_url": "/docs"
    }
