from langchain_core.documents import Document

from app.services.rag_service import RAGService


def _doc(title: str, url: str) -> Document:
    return Document(
        page_content=f"Description: {title}",
        metadata={
            "title": title,
            "location": "New Cairo",
            "price": 3_100_000,
            "url": url,
        },
    )


def _doc_with_location(title: str, location: str) -> Document:
    return Document(
        page_content=f"Description: {title}",
        metadata={"title": title, "location": location, "url": f"https://example.com/{title}"},
    )


def test_analysis_intent_detects_buy_decision_with_session_filters():
    svc = RAGService.__new__(RAGService)
    is_analysis = svc._is_analysis_intent(
        "اشتري ولا لا؟",
        {"location": "New Cairo", "max_price": 3_500_000},
    )
    assert is_analysis is True


def test_collect_analysis_docs_deduplicates_and_prioritizes_better_option():
    svc = RAGService.__new__(RAGService)
    better = _doc("better", "https://example.com/better")
    duplicate = _doc("better copy", "https://example.com/better")
    sample = _doc("sample", "https://example.com/sample")

    docs = svc._collect_analysis_docs(
        {
            "better_option_doc": better,
            "sample_docs": [duplicate, sample],
        },
        max_docs=5,
    )

    assert len(docs) == 2
    assert docs[0].metadata["title"] == "better"
    assert docs[1].metadata["title"] == "sample"


def test_build_analysis_chat_response_includes_decision_and_better_option():
    svc = RAGService.__new__(RAGService)
    better = _doc("Garden View", "https://example.com/garden")
    text = svc._build_analysis_chat_response(
        {
            "insight": "تحليل السوق إيجابي في المنطقة المطلوبة.",
            "matched_count": 8,
            "total_candidates": 40,
            "buy_decision": {
                "headline": "مؤشرات الشراء إيجابية حاليًا",
                "confidence": 0.84,
                "reasons": ["5 من 8 عقارات مناسبة للميزانية."],
            },
            "better_option_found": True,
            "better_option_reason": "أفضل سعر متر مقارنة بالمتوسط.",
            "better_option_doc": better,
        }
    )

    assert "قرار الشراء" in text
    assert "Garden View" in text
    assert "سبب الترشيح" in text


def test_sanitize_filters_normalizes_arabic_location_alias():
    svc = RAGService.__new__(RAGService)
    svc.vector_store = type("VectorStoreStub", (), {"available_locations": {"New Cairo"}})()

    cleaned = svc._sanitize_filters({"location": "التجمع", "max_price": 4_000_000})

    assert cleaned["location"] == "New Cairo"


def test_sanitize_filters_keeps_generic_cairo_not_random_specific_compound():
    svc = RAGService.__new__(RAGService)
    svc.vector_store = type(
        "VectorStoreStub",
        (),
        {
            "available_locations": {
                "O West, 6 October Compounds, 6 October City, Giza",
                "Hyde Park, New Cairo City, Cairo",
            }
        },
    )()

    cleaned = svc._sanitize_filters({"location": "القاهرة"})

    assert cleaned["location"] == "Cairo"


def test_filter_docs_by_location_keeps_requested_area_on_fallback():
    svc = RAGService.__new__(RAGService)
    docs = [
        _doc_with_location("cairo-option", "The 5th Settlement, New Cairo, Cairo"),
        _doc_with_location("remote-option", "Al Mansoura, Al Daqahlya"),
    ]

    filtered = svc._filter_docs_by_location(docs, "the 5th settlement")

    assert len(filtered) == 1
    assert filtered[0].metadata["title"] == "cairo-option"


def test_effective_search_query_folds_structured_fields_into_query_text():
    svc = RAGService.__new__(RAGService)

    query = svc._build_effective_search_query(
        "عاوز شقة",
        {
            "location": "The 5th Settlement",
            "property_type": "apartment",
            "max_price": 4_000_000,
            "desired_services": ["schools", "transport"],
        },
    )

    assert "عاوز شقة" in query
    assert "Location: The 5th Settlement" in query
    assert "Type: apartment" in query
    assert "Maximum price: 4000000" in query
    assert "Nearby services: schools, transport" in query


def test_fast_property_response_returns_cards_without_llm():
    svc = RAGService.__new__(RAGService)
    doc = Document(
        page_content="Description",
        metadata={
            "title": "Garden Apartment",
            "location": "New Cairo",
            "price": 3_500_000,
            "bedrooms": 3,
            "size": 160,
            "nearby_services": ["schools", "transport"],
        },
    )

    text, docs = svc._generate_fast_property_response("عاوز شقة في التجمع", "Excellent Match", [doc], [])

    assert "Garden Apartment" in text
    assert "3.5 مليون جنيه" in text
    assert docs == [doc]


def test_compute_buy_decision_without_budget_uses_market_value_reason():
    svc = RAGService.__new__(RAGService)
    docs = [
        Document(
            page_content="Description: A",
            metadata={
                "title": "A",
                "location": "New Cairo",
                "price": 3_000_000,
                "size": 150,
                "nearby_services": ["schools", "transport"],
                "distance_km": 3.2,
            },
        ),
        Document(
            page_content="Description: B",
            metadata={
                "title": "B",
                "location": "New Cairo",
                "price": 3_400_000,
                "size": 170,
            },
        ),
        Document(
            page_content="Description: C",
            metadata={
                "title": "C",
                "location": "New Cairo",
                "price": 2_900_000,
                "size": 145,
            },
        ),
    ]

    stats = svc._compute_market_stats(docs)
    result = svc._compute_buy_decision(docs, stats, {"location": "New Cairo"}, "strict")

    joined_reasons = " ".join(result["reasons"])
    assert "لم يتم تحديد ميزانية" in joined_reasons
    assert "ضمن أو قريب من ميزانيتك" not in joined_reasons


def test_inventory_stats_intent_and_response_uses_internal_docs_only():
    svc = RAGService.__new__(RAGService)
    docs = [
        Document(
            page_content="Description: A",
            metadata={
                "title": "A",
                "location": "New Cairo City, Cairo",
                "type": "Apartment",
                "price": 3_000_000,
                "size": 150,
                "url": "https://example.com/a",
            },
        ),
        Document(
            page_content="Description: B",
            metadata={
                "title": "B",
                "location": "Downtown, Cairo",
                "type": "Apartment",
                "price": 4_000_000,
                "size": 160,
                "url": "https://example.com/b",
            },
        ),
        Document(
            page_content="Description: C",
            metadata={
                "title": "C",
                "location": "Sheikh Zayed, Giza",
                "type": "Villa",
                "price": 8_000_000,
                "size": 250,
                "url": "https://example.com/c",
            },
        ),
    ]
    svc.vector_store = type(
        "VectorStoreStub",
        (),
        {"all_docs_list": docs, "_enrich_docs": staticmethod(lambda input_docs: input_docs)},
    )()

    assert svc._is_inventory_stats_intent("عدد الشقق في القاهرة كام", {"location": "Cairo"}) is True

    text = svc._build_inventory_stats_response(
        query="عدد الشقق في القاهرة كام",
        filters={"location": "Cairo", "property_type": "apartment"},
    )
    assert "قاعدة بيانات موقعنا الحالية فقط" in text
    assert "عدد النتائج المطابقة لطلبك: 2" in text


def test_scoring_explanation_intent_returns_weighted_formula():
    svc = RAGService.__new__(RAGService)
    assert svc._is_scoring_explanation_intent("التقييم بيتحسب بناء على ايه؟") is True
    assert svc._is_scoring_explanation_intent("تقييم 0.47 بناءا على ايه؟") is True
    text = svc._build_scoring_explanation_response()
    assert "35%" in text
    assert "25%" in text
    assert "15%" in text
    assert "0.47" in text


def test_normalize_listing_intent_maps_arabic_and_english():
    svc = RAGService.__new__(RAGService)
    assert svc._normalize_listing_intent("للايجار") == "rent"
    assert svc._normalize_listing_intent("rent") == "rent"
    assert svc._normalize_listing_intent("للبيع") == "buy"
    assert svc._normalize_listing_intent("buy") == "buy"


def test_build_listing_intent_unavailable_response_uses_internal_counts():
    svc = RAGService.__new__(RAGService)
    docs = [
        Document(
            page_content="Description: A",
            metadata={"url": "https://example.com/buy/a", "listing_intent": "buy"},
        ),
        Document(
            page_content="Description: B",
            metadata={"url": "https://example.com/buy/b", "listing_intent": "buy"},
        ),
    ]
    svc.vector_store = type(
        "VectorStoreStub",
        (),
        {"all_docs_list": docs, "_enrich_docs": staticmethod(lambda input_docs: input_docs)},
    )()

    text = svc._build_listing_intent_unavailable_response("rent", {"location": "Cairo", "property_type": "apartment"})

    assert "مفيش وحدات إيجار" in text
    assert "إجمالي الإيجار المتاح في الداتا الآن: 0" in text
    assert "إجمالي وحدات البيع المتاحة: 2" in text


def test_inventory_stats_response_can_infer_rental_intent_from_query():
    svc = RAGService.__new__(RAGService)
    docs = [
        Document(
            page_content="Description: A",
            metadata={"url": "https://example.com/buy/a", "listing_intent": "buy", "location": "Cairo", "type": "Apartment", "price": 3_000_000, "size": 120},
        ),
    ]
    svc.vector_store = type(
        "VectorStoreStub",
        (),
        {"all_docs_list": docs, "_enrich_docs": staticmethod(lambda input_docs: input_docs)},
    )()

    text = svc._build_inventory_stats_response("عدد الشقق للايجار في القاهرة كام", {"location": "Cairo", "property_type": "apartment"})
    assert "نوع العرض المطلوب: إيجار" in text
    assert "عدد النتائج المطابقة لطلبك: 0" in text


def test_apply_exact_filters_respects_listing_intent():
    svc = RAGService.__new__(RAGService)
    docs = [
        Document(
            page_content="Description: A",
            metadata={
                "title": "Buy Listing",
                "location": "Cairo",
                "type": "Apartment",
                "price": 3_000_000,
                "url": "https://example.com/buy/a",
            },
        ),
        Document(
            page_content="Description: B",
            metadata={
                "title": "Rent Listing",
                "location": "Cairo",
                "type": "Apartment",
                "price": 30_000,
                "url": "https://example.com/rent/b",
            },
        ),
    ]

    filtered = svc._apply_exact_filters(docs, {"listing_intent": "rent", "property_type": "apartment"})

    assert len(filtered) == 1
    assert filtered[0].metadata["title"] == "Rent Listing"


def test_coverage_intent_and_response_returns_location_list():
    svc = RAGService.__new__(RAGService)
    docs = [
        Document(
            page_content="Description: A",
            metadata={"location": "New Cairo City, Cairo", "type": "Apartment", "url": "https://example.com/buy/a", "price": 3_000_000, "size": 120},
        ),
        Document(
            page_content="Description: B",
            metadata={"location": "Sheikh Zayed, Giza", "type": "Apartment", "url": "https://example.com/buy/b", "price": 4_000_000, "size": 140},
        ),
        Document(
            page_content="Description: C",
            metadata={"location": "New Cairo City, Cairo", "type": "Apartment", "url": "https://example.com/buy/c", "price": 3_500_000, "size": 130},
        ),
    ]
    svc.vector_store = type(
        "VectorStoreStub",
        (),
        {"all_docs_list": docs, "_enrich_docs": staticmethod(lambda input_docs: input_docs)},
    )()

    assert svc._is_coverage_intent("عندكو شقق في مناطق ايه؟", {"property_type": "apartment"}) is True
    text = svc._build_coverage_response("عندكو شقق في مناطق ايه؟", {"property_type": "apartment", "listing_intent": "buy"})

    assert "المناطق المتاحة عندنا" in text
    assert "New Cairo City, Cairo" in text
    assert "Sheikh Zayed, Giza" in text
