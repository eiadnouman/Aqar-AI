import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from app.models import ChatRequest, ChatResponse, Property, SearchRequest, SearchResponse, RecommendRequest, RecommendResponse
from app.core.rag import RealEstateRAG
from dotenv import load_dotenv

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load Env
load_dotenv()

# Global Engine Instance
rag_engine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_engine
    logger.info("Startup: Initializing AI Engine...")
    try:
        # Determine paths relative to root
        # If running from /backend/, data should be ../data
        index_path = os.getenv("FAISS_INDEX_PATH", "../data/faiss_index_cloud")
        rag_engine = RealEstateRAG(index_path=index_path)
        logger.info("Startup: AI Engine Ready.")
    except Exception as e:
        logger.error(f"Startup Failed: {e}")
    yield
    # Cleanup if needed
    logger.info("Shutdown: Clearing resources.")

app = FastAPI(
    title="Aqar-AI Intelligence Microservice",
    description="Conversational AI API for Real Estate Recommendations",
    version="2.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "engine_status": "ready" if rag_engine else "not_initialized"
    }

@app.get("/")
async def root():
    return {
        "message": "Welcome to Aqar-AI Intelligence Microservice!",
        "docs_url": "/docs"
    }

@app.post("/v1/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    if not rag_engine:
        raise HTTPException(status_code=503, detail="AI Engine is initializing. Please wait.")
    
    try:
        logger.info(f"Processing Request: {request.message} (Session: {request.session_id})")
        response_text, docs = rag_engine.generate_recommendation(request.message, session_id=request.session_id)
        
        # Map output to Pydantic models
        properties = []
        for doc in docs:
            meta = doc.metadata
            properties.append(Property(
                title=meta.get("title", "Property Listing"),
                location=meta.get("location", "Unknown Location"),
                price=float(meta.get("price", 0)),
                bedrooms=int(meta.get("bedrooms", 0)),
                bathrooms=int(meta.get("bathrooms", 0)),
                size=float(meta.get("size", 0)),
                image_url=meta.get("image", ""),
                description=doc.page_content.split('Description: ')[-1][:200] + "...",
                url=meta.get("url", "#")
            ))
            
        return ChatResponse(
            answer=response_text,
            properties=properties
        )
        
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/search", response_model=SearchResponse)
async def search_endpoint(request: SearchRequest):
    if not rag_engine:
        raise HTTPException(status_code=503, detail="AI Engine is initializing. Please wait.")
    
    try:
        logger.info(f"Processing Search Request: {request.query}")
        filters, docs = rag_engine.search_properties(request.query)
        
        properties = []
        for doc in docs:
            meta = doc.metadata
            properties.append(Property(
                title=meta.get("title", "Property Listing"),
                location=meta.get("location", "Unknown Location"),
                price=float(meta.get("price", 0)),
                bedrooms=int(meta.get("bedrooms", 0)),
                bathrooms=int(meta.get("bathrooms", 0)),
                size=float(meta.get("size", 0)),
                image_url=meta.get("image", ""),
                description=doc.page_content.split('Description: ')[-1][:200] + "...",
                url=meta.get("url", "#")
            ))
            
        return SearchResponse(
            properties=properties,
            filters_used=filters
        )
        
    except Exception as e:
        logger.error(f"Error processing search request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/recommend/similar", response_model=RecommendResponse)
async def recommend_similar_endpoint(request: RecommendRequest):
    if not rag_engine:
        raise HTTPException(status_code=503, detail="AI Engine is initializing. Please wait.")
    
    try:
        logger.info(f"Processing Recommendation Request")
        docs = rag_engine.get_similar_properties(request.property_description)
        
        properties = []
        for doc in docs:
            meta = doc.metadata
            properties.append(Property(
                title=meta.get("title", "Property Listing"),
                location=meta.get("location", "Unknown Location"),
                price=float(meta.get("price", 0)),
                bedrooms=int(meta.get("bedrooms", 0)),
                bathrooms=int(meta.get("bathrooms", 0)),
                size=float(meta.get("size", 0)),
                image_url=meta.get("image", ""),
                description=doc.page_content.split('Description: ')[-1][:200] + "...",
                url=meta.get("url", "#")
            ))
            
        return RecommendResponse(
            properties=properties
        )
        
    except Exception as e:
        logger.error(f"Error processing recommend request: {e}")
        raise HTTPException(status_code=500, detail=str(e))
