from fastapi import APIRouter, HTTPException, Depends
from app.models.schemas import RecommendRequest, RecommendResponse, Property
from app.services.rag_service import RAGService, get_rag_service
import logging

logger = logging.getLogger("aqarai")
router = APIRouter()

@router.post("/recommend/similar", response_model=RecommendResponse)
async def recommend_similar_endpoint(request: RecommendRequest, rag_engine: RAGService = Depends(get_rag_service)):
    """Handles returning semantically similar properties to a given description."""
    try:
        logger.info(f"Processing Recommendation Request")
        docs = rag_engine.recommend_similar(request.property_description)
        
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
