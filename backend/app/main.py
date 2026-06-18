from threading import Thread, Timer

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
    """System health checkpoint with diagnostics."""
    status = "healthy"
    diagnostics = {}
    try:
        rag = get_rag_service()
        vs = getattr(rag, "vector_store", None)
        if vs:
            diagnostics["embeddings_class"] = vs.embeddings.__class__.__name__ if vs.embeddings else None
            diagnostics["vectorstore_loaded"] = vs.vectorstore is not None
            diagnostics["embeddings_error"] = vs.embeddings_error
            diagnostics["load_index_error"] = vs.load_index_error
            diagnostics["properties_count"] = len(vs.all_docs_list)
            diagnostics["locations_count"] = len(vs.available_locations)
            if not vs.vectorstore:
                status = "degraded"
        else:
            status = "degraded"
            diagnostics["error"] = "vector_store not initialized on RAG service"
    except Exception as e:
        status = "unhealthy"
        diagnostics["error"] = str(e)

    # Check for crucial environment variables (mask content)
    import os
    diagnostics["env_groq_api_key_set"] = bool(os.getenv("GROQ_API_KEY"))
    diagnostics["env_huggingface_token_set"] = bool(os.getenv("HUGGINGFACEHUB_API_TOKEN"))

    return {
        "status": status,
        "version": settings.version,
        "diagnostics": diagnostics
    }


# ===== Periodic Data Sync =====

_sync_timer = None


def _run_periodic_sync():
    """Background task that refreshes data from the external API and rebuilds the index."""
    global _sync_timer
    try:
        rag = get_rag_service()
        vs = getattr(rag, "vector_store", None)
        if vs and hasattr(vs, "refresh_from_external"):
            logger.info("Periodic data sync: starting refresh from external API...")
            success = vs.refresh_from_external()
            if success:
                logger.info("Periodic data sync: completed successfully.")
            else:
                logger.warning("Periodic data sync: refresh returned False.")
    except Exception as e:
        logger.error(f"Periodic data sync failed: {e}")
    finally:
        # Schedule the next sync
        interval = max(settings.data_sync_interval_minutes, 1) * 60
        _sync_timer = Timer(interval, _run_periodic_sync)
        _sync_timer.daemon = True
        _sync_timer.start()


@app.on_event("startup")
def warm_rag_service():
    """Preload the heavy retrieval stack after Uvicorn starts accepting traffic."""
    def _warm():
        try:
            get_rag_service()
            logger.info("RAG service warmed successfully.")
        except Exception as e:
            logger.error(f"RAG service warmup failed: {e}")

        # Start periodic sync after initial warmup
        _schedule_first_sync()

    Thread(target=_warm, daemon=True).start()


def _schedule_first_sync():
    """Kick off the first periodic data sync after a brief delay."""
    global _sync_timer
    # First sync after 10 seconds (let startup complete), then every N minutes
    _sync_timer = Timer(10, _run_periodic_sync)
    _sync_timer.daemon = True
    _sync_timer.start()
    interval = settings.data_sync_interval_minutes
    logger.info(f"Periodic data sync scheduled every {interval} minutes.")


@app.get("/", tags=["System"])
async def root():
    """API Root standard greeting."""
    return {
        "message": "Welcome to Aqar-AI Intelligence API!",
        "docs_url": "/docs"
    }
