from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1 import chat, search, recommend
from app.core.config import settings

app = FastAPI(
    title=settings.project_name,
    description="Conversational AI API for Real Estate Recommendations (Clean Architecture)",
    version=settings.version,
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Attach Modular API Routers
app.include_router(chat.router, prefix="/api/v1", tags=["Chat"])
app.include_router(search.router, prefix="/api/v1", tags=["Search"])
app.include_router(recommend.router, prefix="/api/v1", tags=["Recommend"])

@app.get("/health", tags=["System"])
async def health_check():
    """System health checkpoint."""
    return {
        "status": "healthy",
        "version": settings.version
    }

@app.get("/", tags=["System"])
async def root():
    """API Root standard greeting."""
    return {
        "message": "Welcome to Aqar-AI Intelligence API!",
        "docs_url": "/docs"
    }
