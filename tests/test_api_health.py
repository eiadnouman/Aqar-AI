import httpx
import pytest

from app.main import app


@pytest.mark.anyio
async def test_health_endpoint_returns_200():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "healthy"
