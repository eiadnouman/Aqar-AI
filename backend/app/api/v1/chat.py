from fastapi import APIRouter, HTTPException, Depends
from app.models.schemas import ChatRequest, ChatResponse, Property
from app.services.rag_service import RAGService, get_rag_service
import logging

logger = logging.getLogger("aqarai")
router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, rag_engine: RAGService = Depends(get_rag_service)):
    """Handles natural language conversational queries."""
    try:
        logger.info(f"Processing Chat Request: {request.message} (Session: {request.session_id})")
        response_text, docs = rag_engine.get_recommendation(request.message, session_id=request.session_id)
        
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
        logger.error(f"Error processing chat request: {e}")
        raise HTTPException(status_code=500, detail=str(e))
