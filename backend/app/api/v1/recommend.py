from fastapi import APIRouter, HTTPException, Depends
from app.models.schemas import RecommendRequest, RecommendResponse, Property
from app.services.rag_service import RAGService, get_rag_service
import logging

logger = logging.getLogger("aqarai")
router = APIRouter()


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value, default=0):
    try:
        return int(float(value))
    except Exception:
        return default

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
                price=_safe_float(meta.get("price", 0)),
                bedrooms=_safe_int(meta.get("bedrooms", 0)),
                bathrooms=_safe_int(meta.get("bathrooms", 0)),
                size=_safe_float(meta.get("size", 0)),
                image_url=meta.get("image", ""),
                description=doc.page_content.split('Description: ')[-1][:200] + "...",
                url=meta.get("url", "#"),
                latitude=_safe_float(meta.get("lat") or meta.get("latitude") or 0) or None,
                longitude=_safe_float(meta.get("lon") or meta.get("longitude") or 0) or None,
                distance_km=_safe_float(meta.get("distance_km", 0)) or None,
                nearby_services=list(meta.get("nearby_services", []) or []),
                recommendation_score=_safe_float(meta.get("recommendation_score", 0)) or None,
            ))
            
        return RecommendResponse(
            properties=properties
        )
        
    except Exception as e:
        logger.error(f"Error processing recommend request: {e}")
        raise HTTPException(status_code=500, detail=str(e))
