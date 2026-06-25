import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_unified_schema_property_search():
    """Verify that a property search query returns the full unified schema with answers and properties."""
    response = client.post("/api/v1/chat", json={"message": "عاوز شقة في التجمع الخامس 3 غرف تحت 15 مليون"})
    assert response.status_code == 200
    data = response.json()
    
    assert "answer" in data
    assert "properties" in data


def test_unified_schema_non_property_chitchat():
    """Verify that a non-property chitchat query returns no properties."""
    queries = [
        "ازيك يا عقاري",
        "مين انت",
        "ايه الطقس النهارده",
        "عاوز حاجة حلوة",
        "شكرا يا غالي"
    ]
    for q in queries:
        response = client.post("/api/v1/chat", json={"message": q})
        assert response.status_code == 200
        data = response.json()
        
        assert "answer" in data
        assert len(data["properties"]) == 0
        assert "[SHOW_CARDS]" not in data["answer"]


def test_unified_schema_hard_arabic_query():
    """Verify that a complex real-estate query with filters works and extracts correct properties."""
    response = client.post("/api/v1/chat", json={"message": "فيه شقق في زايد قريبة من المدارس تحت 18 مليون؟"})
    assert response.status_code == 200
    data = response.json()
    
    assert "answer" in data
    assert len(data["properties"]) > 0
    
    # The top properties should contain the location 'Zayed' or 'زايد'
    has_zayed = any(
        "zayed" in prop["location"].lower() or "زايد" in prop["location"]
        for prop in data["properties"]
    )
    assert has_zayed is True


def test_bilingual_response():
    """Verify that English queries receive responses in English and Arabic queries in Arabic."""
    # Arabic greeting -> deterministic Arabic response
    response_ar = client.post("/api/v1/chat", json={"message": "ازيك"})
    assert response_ar.status_code == 200
    data_ar = response_ar.json()
    # Should contain Arabic text from _build_greeting_response
    assert "AqarAI" in data_ar["answer"]
    assert any(c >= '\u0600' and c <= '\u06FF' for c in data_ar["answer"]), "Arabic greeting should respond in Arabic"

    # English greeting -> deterministic English response
    response_en = client.post("/api/v1/chat", json={"message": "hello"})
    assert response_en.status_code == 200
    data_en = response_en.json()
    # Should contain English text from _build_greeting_response
    assert "AqarAI" in data_en["answer"]
    assert "Hello" in data_en["answer"] or "hello" in data_en["answer"].lower()

