from app.services.filter_engine import FilterEngine


class _BrokenLLM:
    def with_structured_output(self, *_args, **_kwargs):
        raise RuntimeError("forced structured output failure")


class _DummyLLMManager:
    def get_llm(self):
        return _BrokenLLM()


def test_filter_engine_extracts_desired_services_from_regex_fallback():
    engine = FilterEngine(_DummyLLMManager())
    filters = engine.extract_filters(
        "عايز شقة في التجمع قريبة من مدارس ومواصلات ومول",
        {"New Cairo"},
    )

    assert "desired_services" in filters
    assert "schools" in filters["desired_services"]
    assert "transport" in filters["desired_services"]
    assert "commercial_area" in filters["desired_services"]


def test_filter_engine_extracts_listing_intent_from_regex_fallback():
    engine = FilterEngine(_DummyLLMManager())

    rent_filters = engine.extract_filters("فيه شقق للإيجار في القاهرة؟", {"Cairo"})
    assert rent_filters.get("listing_intent") == "rent"

    buy_filters = engine.extract_filters("عايز شقة للبيع في التجمع", {"New Cairo"})
    assert buy_filters.get("listing_intent") == "buy"


def test_filter_engine_backfills_missing_llm_fields_from_regex(monkeypatch):
    class _LLMManager:
        def get_llm(self):
            return object()

    engine = FilterEngine(_LLMManager())
    monkeypatch.setattr(
        engine,
        "_extract_filters_llm",
        lambda *_args, **_kwargs: {"property_type": "apartment"},
    )

    filters = engine.extract_filters("عايز شقة للبيع في التجمع", {"New Cairo"})

    assert filters.get("property_type") == "apartment"
    assert filters.get("location") == "New Cairo"
    assert filters.get("listing_intent") == "buy"
