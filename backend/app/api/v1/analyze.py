from fastapi import APIRouter, HTTPException, Depends
from app.api.v1.serializers import doc_to_property
from app.models.schemas import AnalyzeRequest, AnalyzeResponse, BuyDecision, SegmentStat
from app.services.rag_service import RAGService, get_rag_service
import logging

logger = logging.getLogger("aqarai")
router = APIRouter()


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_endpoint(request: AnalyzeRequest, rag_engine: RAGService = Depends(get_rag_service)):
    """Returns aggregated market analytics over matched properties."""
    try:
        explicit_filters = request.model_dump(exclude_unset=True, exclude_none=True)
        query = explicit_filters.pop("query", None)
        logger.info(f"Processing Analyze Request: {query or explicit_filters or 'all inventory'}")

        result = rag_engine.analyze_market(query=query, explicit_filters=explicit_filters)
        sample_docs = result.get("sample_docs", [])

        sample_properties = [doc_to_property(doc) for doc in sample_docs]

        top_locations = [SegmentStat(**item) for item in result.get("top_locations", [])]
        top_property_types = [SegmentStat(**item) for item in result.get("top_property_types", [])]
        buy_decision = BuyDecision(**result.get("buy_decision", {
            "decision": "insufficient_data",
            "headline": "بيانات غير كافية",
            "confidence": 0.0,
            "reasons": ["لا توجد بيانات كافية لإصدار قرار شراء."],
        }))
        better_option_doc = result.get("better_option_doc")
        better_option = doc_to_property(better_option_doc) if better_option_doc else None

        return AnalyzeResponse(
            insight=result.get("insight", ""),
            filters_used=result.get("filters_used", {}),
            match_scope=result.get("match_scope", "none"),
            total_candidates=result.get("total_candidates", 0),
            matched_count=result.get("matched_count", 0),
            stats=result.get("stats", {}),
            top_locations=top_locations,
            top_property_types=top_property_types,
            buy_decision=buy_decision,
            better_option_found=result.get("better_option_found", False),
            better_option_reason=result.get("better_option_reason", "لا يوجد بديل أوضح حاليًا."),
            better_option=better_option,
            sample_properties=sample_properties,
        )

    except Exception as e:
        logger.error(f"Error processing analyze request: {e}")
        raise HTTPException(status_code=500, detail=str(e))
