from fastapi import APIRouter, HTTPException, Depends
from app.models.schemas import SearchRequest, SearchResponse, Property
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
