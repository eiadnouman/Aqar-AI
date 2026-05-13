from fastapi import APIRouter, HTTPException, Depends
from app.api.v1.serializers import doc_to_property
from app.models.schemas import ChatRequest, ChatResponse
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
        
        properties = [doc_to_property(doc) for doc in docs]
            
        return ChatResponse(
            answer=response_text,
            properties=properties
        )
        
    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        raise HTTPException(status_code=500, detail=str(e))
