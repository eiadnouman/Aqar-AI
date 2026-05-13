from fastapi import APIRouter, HTTPException, Depends
from app.api.v1.serializers import doc_to_property
from app.models.schemas import RecommendRequest, RecommendResponse
from app.services.rag_service import RAGService, get_rag_service
import logging

logger = logging.getLogger("aqarai")
router = APIRouter()


@router.post("/recommend/similar", response_model=RecommendResponse)
async def recommend_similar_endpoint(request: RecommendRequest, rag_engine: RAGService = Depends(get_rag_service)):
    """Handles returning semantically similar properties to a given description."""
    try:
        logger.info(f"Processing Recommendation Request")
        if request.session_id or request.property_ids:
            docs = rag_engine.recommend_from_interactions(
                session_id=request.session_id or "_anonymous",
                property_ids=request.property_ids,
                limit=5,
            )
        else:
            docs = rag_engine.recommend_similar(request.property_description)
        
        properties = [doc_to_property(doc) for doc in docs]
            
        return RecommendResponse(
            properties=properties
        )
        
    except Exception as e:
        logger.error(f"Error processing recommend request: {e}")
        raise HTTPException(status_code=500, detail=str(e))
