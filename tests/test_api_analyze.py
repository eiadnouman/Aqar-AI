import pytest
from langchain_core.documents import Document

from app.api.v1.analyze import analyze_endpoint
from app.models.schemas import AnalyzeRequest


def _sample_doc(idx: int) -> Document:
    return Document(
        page_content=f"Description: Sample property {idx} description.",
        metadata={
            "title": f"Property {idx}",
            "location": "New Cairo",
            "price": 3_000_000,
            "bedrooms": 3,
            "bathrooms": 2,
            "size": 175,
            "image": "images/images/property_1/1.jpg",
            "url": f"https://example.com/property/{idx}",
            "type": "Apartment",
        },
    )


class _FakeRAGService:
    def analyze_market(self, query=None, explicit_filters=None):
        _ = query, explicit_filters
        return {
            "insight": "analysis-ready",
            "filters_used": {"location": "New Cairo"},
            "match_scope": "strict",
            "total_candidates": 12,
            "matched_count": 4,
            "stats": {"count": 4, "avg_price": 3_000_000.0},
            "top_locations": [
                {
                    "name": "New Cairo",
                    "count": 4,
                    "avg_price": 3_000_000.0,
                    "median_price": 3_000_000.0,
                    "avg_price_per_sqm": 17_142.86,
                }
            ],
            "top_property_types": [
                {
                    "name": "Apartment",
                    "count": 4,
                    "avg_price": 3_000_000.0,
                    "median_price": 3_000_000.0,
                    "avg_price_per_sqm": 17_142.86,
                }
            ],
            "buy_decision": {
                "decision": "buy_now",
                "headline": "مؤشرات الشراء إيجابية حاليًا",
                "confidence": 0.84,
                "reasons": ["3 من 4 عقارات في نطاق الميزانية."],
            },
            "better_option_found": True,
            "better_option_reason": "أفضل خيار من حيث السعر والخدمات.",
            "better_option_doc": _sample_doc(1),
            "sample_docs": [_sample_doc(1)],
        }


@pytest.mark.anyio
async def test_analyze_handler_returns_aggregations_and_sample_properties():
    response = await analyze_endpoint(
        AnalyzeRequest(query="حلل السوق"),
        rag_engine=_FakeRAGService(),
    )

    assert response.match_scope == "strict"
    assert response.total_candidates == 12
    assert response.stats["count"] == 4
    assert len(response.top_locations) == 1
    assert response.buy_decision.decision == "buy_now"
    assert response.better_option_found is True
    assert response.better_option is not None
    assert response.better_option.id == 1
    assert len(response.sample_properties) == 1
    assert response.sample_properties[0].id == 1
