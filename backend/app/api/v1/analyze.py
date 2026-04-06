from fastapi import APIRouter, HTTPException, Depends
from app.models.schemas import AnalyzeRequest, AnalyzeResponse, BuyDecision, Property, SegmentStat
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


def _doc_to_property(doc):
    meta = doc.metadata
    return Property(
        title=meta.get("title", "Property Listing"),
        location=meta.get("location", "Unknown Location"),
        price=_safe_float(meta.get("price", 0)),
        bedrooms=_safe_int(meta.get("bedrooms", 0)),
        bathrooms=_safe_int(meta.get("bathrooms", 0)),
        size=_safe_float(meta.get("size", 0)),
        image_url=meta.get("image", ""),
        description=doc.page_content.split("Description: ")[-1][:200] + "...",
        url=meta.get("url", "#"),
        latitude=_safe_float(meta.get("lat") or meta.get("latitude") or 0) or None,
        longitude=_safe_float(meta.get("lon") or meta.get("longitude") or 0) or None,
        distance_km=_safe_float(meta.get("distance_km", 0)) or None,
        nearby_services=list(meta.get("nearby_services", []) or []),
        recommendation_score=_safe_float(meta.get("recommendation_score", 0)) or None,
    )


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_endpoint(request: AnalyzeRequest, rag_engine: RAGService = Depends(get_rag_service)):
    """Returns aggregated market analytics over matched properties."""
    try:
        explicit_filters = request.model_dump(exclude_unset=True, exclude_none=True)
        query = explicit_filters.pop("query", None)
        logger.info(f"Processing Analyze Request: {query or explicit_filters or 'all inventory'}")

        result = rag_engine.analyze_market(query=query, explicit_filters=explicit_filters)
        sample_docs = result.get("sample_docs", [])

        sample_properties = [_doc_to_property(doc) for doc in sample_docs]

        top_locations = [SegmentStat(**item) for item in result.get("top_locations", [])]
        top_property_types = [SegmentStat(**item) for item in result.get("top_property_types", [])]
        buy_decision = BuyDecision(**result.get("buy_decision", {
            "decision": "insufficient_data",
            "headline": "بيانات غير كافية",
            "confidence": 0.0,
            "reasons": ["لا توجد بيانات كافية لإصدار قرار شراء."],
        }))
        better_option_doc = result.get("better_option_doc")
        better_option = _doc_to_property(better_option_doc) if better_option_doc else None

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
