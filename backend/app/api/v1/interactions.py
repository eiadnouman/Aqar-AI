from fastapi import APIRouter, Depends, HTTPException

from app.api.v1.serializers import doc_to_property
from app.models.schemas import (
    PropertyInteractionRequest,
    PropertyInteractionResponse,
    RecommendResponse,
    SessionRecommendRequest,
)
from app.services.rag_service import RAGService, get_rag_service
from app.core.logging import logger


router = APIRouter()


@router.post("/interactions/property-click", response_model=PropertyInteractionResponse)
async def record_property_click(
    request: PropertyInteractionRequest,
    rag_engine: RAGService = Depends(get_rag_service),
):
    try:
        property_ids = rag_engine.record_property_interaction(
            session_id=request.session_id,
            property_id=request.property_id,
            event_type=request.event_type,
        )
        return PropertyInteractionResponse(
            saved=True,
            session_id=request.session_id,
            property_ids=property_ids,
        )
    except Exception as e:
        logger.error(f"Error recording property interaction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/recommend/session", response_model=RecommendResponse)
async def recommend_from_session(
    request: SessionRecommendRequest,
    rag_engine: RAGService = Depends(get_rag_service),
):
    try:
        docs = rag_engine.recommend_from_interactions(
            session_id=request.session_id,
            property_ids=request.property_ids,
            limit=request.limit,
        )
        return RecommendResponse(properties=[doc_to_property(doc) for doc in docs])
    except Exception as e:
        logger.error(f"Error generating session recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))
