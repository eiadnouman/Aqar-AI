from langchain_core.documents import Document

from app.services.rag_service import RAGService


def _doc(
    title: str,
    location: str,
    lat: float,
    lon: float,
    price: float,
    size: float,
    services: list[str],
):
    return Document(
        page_content=f"Description: {title}",
        metadata={
            "title": title,
            "location": location,
            "lat": lat,
            "lon": lon,
            "price": price,
            "size": size,
            "bedrooms": 3,
            "bathrooms": 2,
            "type": "Apartment",
            "nearby_services": services,
            "url": f"https://example.com/{title}",
        },
    )


def test_rank_recommendations_prioritizes_distance_and_services():
    svc = RAGService.__new__(RAGService)
    docs = [
        _doc("near_school", "New Cairo", 30.01, 31.50, 3_000_000, 180, ["schools", "transport"]),
        _doc("far_no_services", "Giza", 29.85, 31.10, 2_950_000, 180, []),
    ]

    ranked = svc._rank_recommendations(
        docs,
        {
            "location": "New Cairo",
            "max_price": 3_200_000,
            "desired_services": ["schools"],
            "property_type": "Apartment",
        },
    )

    assert ranked[0].metadata["title"] == "near_school"
    assert ranked[0].metadata.get("distance_km", 0) <= ranked[1].metadata.get("distance_km", 999)
    assert ranked[0].metadata.get("recommendation_score", 0) >= ranked[1].metadata.get("recommendation_score", 0)


def test_identify_better_option_returns_top_candidate_when_signals_are_strong():
    svc = RAGService.__new__(RAGService)
    docs = [
        _doc("best_option", "New Cairo", 30.01, 31.50, 2_800_000, 190, ["schools", "hospitals", "transport"]),
        _doc("other_option", "New Cairo", 30.03, 31.55, 3_500_000, 180, ["security"]),
    ]
    ranked = svc._rank_recommendations(
        docs,
        {"location": "New Cairo", "max_price": 3_000_000, "desired_services": ["schools"]},
    )
    stats = svc._compute_market_stats(ranked)

    result = svc._identify_better_option(ranked, stats, {"max_price": 3_000_000})

    assert result["found"] is True
    assert result["doc"] is not None
    assert result["doc"].metadata["title"] == "best_option"
