import pytest
from langchain_core.documents import Document

from app.api.v1.interactions import record_property_click, recommend_from_session
from app.api.v1.recommend import recommend_similar_endpoint
from app.api.v1.search import search_endpoint
from app.models.schemas import PropertyInteractionRequest, RecommendRequest, SearchRequest, SessionRecommendRequest


def _sample_doc(idx: int) -> Document:
    return Document(
        page_content=f"Description: Sample property {idx} description.",
        metadata={
            "title": f"Property {idx}",
            "location": "New Cairo",
            "price": 2_500_000,
            "bedrooms": 3,
            "bathrooms": 2,
            "size": 180,
            "image": "images/images/property_1/1.jpg",
            "url": f"https://example.com/property/{idx}",
        },
    )


class _FakeRAGService:
    def search_properties(self, query=None, explicit_filters=None):
        return {"query": query, **(explicit_filters or {})}, [_sample_doc(1)]

    def recommend_similar(self, _description, k=5):
        limit = k if k is not None else 5
        return [_sample_doc(i) for i in range(1, min(limit, 2) + 1)]

    def get_recommendation(self, _query, session_id=None):
        return "ok", []

    def record_property_interaction(self, session_id, property_id, event_type="click"):
        _ = event_type
        return [int(property_id)]

    def recommend_from_interactions(self, session_id, property_ids=None, limit=5):
        _ = session_id, property_ids, limit
        return [_sample_doc(2)]


@pytest.mark.anyio
async def test_search_and_recommend_handlers_smoke():
    fake_service = _FakeRAGService()

    search_response = await search_endpoint(
        SearchRequest(query="شقة في التجمع"),
        rag_engine=fake_service,
    )
    assert len(search_response.properties) == 1
    assert search_response.properties[0].id == 1
    assert search_response.properties[0].location == "New Cairo"
    assert "query" in search_response.filters_used

    recommend_response = await recommend_similar_endpoint(
        RecommendRequest(property_description="شقة واسعة قريبة من الخدمات"),
        rag_engine=fake_service,
    )
    assert len(recommend_response.properties) >= 1
    assert recommend_response.properties[0].id == 1

    click_response = await record_property_click(
        PropertyInteractionRequest(session_id="session-1", property_id=2),
        rag_engine=fake_service,
    )
    assert click_response.saved is True
    assert click_response.property_ids == [2]

    session_recommend_response = await recommend_from_session(
        SessionRecommendRequest(session_id="session-1"),
        rag_engine=fake_service,
    )
    assert session_recommend_response.properties[0].id == 2
