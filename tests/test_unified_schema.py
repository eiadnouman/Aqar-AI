import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_unified_schema_property_search():
    """Verify that a property search query returns the full unified schema with answers, properties, and comparison items."""
    response = client.post("/api/v1/chat", json={"message": "عاوز شقة في التجمع الخامس 3 غرف تحت 15 مليون"})
    assert response.status_code == 200
    data = response.json()
    
    assert "answer" in data
    assert "properties" in data
    assert "comparison" in data
    
    # Check that comparison matches properties count
    assert len(data["properties"]) == len(data["comparison"])
    
    # Validate comparison fields
    for item in data["comparison"]:
        assert "id" in item
        assert "title" in item
        assert "image_url" in item
        if item["image_url"]:
            assert item["image_url"].startswith("http")


def test_unified_schema_non_property_chitchat():
    """Verify that a non-property chitchat query returns no properties and no comparison items."""
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
        assert len(data["comparison"]) == 0
        assert "[SHOW_CARDS]" not in data["answer"]


def test_unified_schema_hard_arabic_query():
    """Verify that a complex real-estate query with filters works and extracts correct properties."""
    response = client.post("/api/v1/chat", json={"message": "فيه شقق في زايد قريبة من المدارس تحت 18 مليون؟"})
    assert response.status_code == 200
    data = response.json()
    
    assert "answer" in data
    assert len(data["properties"]) > 0
    assert len(data["comparison"]) > 0
    
    # The top properties should contain the location 'Zayed' or 'زايد'
    has_zayed = any(
        "zayed" in prop["location"].lower() or "زايد" in prop["location"]
        for prop in data["properties"]
    )
    assert has_zayed is True


def test_bilingual_response():
    """Verify that English queries receive responses in English and Arabic queries in Arabic."""
    # Arabic query -> Arabic response
    response_ar = client.post("/api/v1/chat", json={"message": "مرحبا، مين انت؟"})
    assert response_ar.status_code == 200
    data_ar = response_ar.json()
    assert any(word in data_ar["answer"] for word in ["أنا", "عقار", "مساعد", "مرحباً", "اهلا"])

    # English query -> English response
    response_en = client.post("/api/v1/chat", json={"message": "Hello, who are you?"})
    assert response_en.status_code == 200
    data_en = response_en.json()
    # Check that it responded in English
    assert any(word.lower() in data_en["answer"].lower() for word in ["i am", "aqarai", "hello", "real estate", "consultant", "smart"])

