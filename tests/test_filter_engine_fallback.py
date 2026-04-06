from app.services.filter_engine import FilterEngine


class _BrokenLLM:
    def with_structured_output(self, *_args, **_kwargs):
        raise RuntimeError("forced structured output failure")


class _DummyLLMManager:
    def get_llm(self):
        return _BrokenLLM()


def test_filter_engine_falls_back_to_regex_when_tool_calling_fails():
    engine = FilterEngine(_DummyLLMManager())
    filters = engine.extract_filters("عاوز شقة غرفتين في التجمع 3 مليون", {"New Cairo"})

    assert filters["location"] == "New Cairo"
    assert filters["min_bedrooms"] == 2
    assert filters["max_price"] == 3_000_000.0
