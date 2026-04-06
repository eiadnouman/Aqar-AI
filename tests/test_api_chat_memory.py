import pytest

from app.api.v1.chat import chat_endpoint
from app.models.schemas import ChatRequest
from app.services import rag_service as rag_service_module


class _StatefulStubRAGService:
    def __init__(self):
        self.turns = {}

    def get_recommendation(self, _query, session_id=None):
        sid = session_id or "_anon"
        self.turns[sid] = self.turns.get(sid, 0) + 1
        return f"turn={self.turns[sid]}", []


@pytest.mark.anyio
async def test_chat_handler_conversation_is_stable_and_session_memory_persists(monkeypatch):
    monkeypatch.setattr(rag_service_module, "RAGService", _StatefulStubRAGService)
    rag_service_module.get_rag_service.cache_clear()

    try:
        # Shared singleton-like provider should return the same service object.
        shared_service_a = rag_service_module.get_rag_service()
        shared_service_b = rag_service_module.get_rag_service()
        assert shared_service_a is shared_service_b

        first_response = await chat_endpoint(
            ChatRequest(message="ازيك", session_id="session-1"),
            rag_engine=shared_service_a,
        )
        assert first_response.answer == "turn=1"
        assert first_response.properties == []

        second_response = await chat_endpoint(
            ChatRequest(message="عامل ايه", session_id="session-1"),
            rag_engine=shared_service_a,
        )
        assert second_response.answer == "turn=2"
        assert second_response.properties == []
    finally:
        rag_service_module.get_rag_service.cache_clear()
