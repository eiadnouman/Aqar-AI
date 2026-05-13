from fastapi import APIRouter, HTTPException, Depends
from app.api.v1.serializers import doc_to_property
from app.models.schemas import SearchRequest, SearchResponse
from app.services.rag_service import RAGService, get_rag_service
import logging

logger = logging.getLogger("aqarai")
router = APIRouter()


@router.post("/search", response_model=SearchResponse)
async def search_endpoint(request: SearchRequest, rag_engine: RAGService = Depends(get_rag_service)):
    """Handles pure search queries without conversational generation."""
    try:
        logger.info(f"Processing Search Request: {request.query or 'Explicit UI Filters'}")
        explicit_filters = request.model_dump(exclude_unset=True, exclude_none=True)
        query = explicit_filters.pop('query', None)
        
        filters, docs = rag_engine.search_properties(query=query, explicit_filters=explicit_filters)
        
        properties = [doc_to_property(doc) for doc in docs]
            
        return SearchResponse(
            properties=properties,
            filters_used=filters
        )
        
    except Exception as e:
        logger.error(f"Error processing search request: {e}")
        raise HTTPException(status_code=500, detail=str(e))
