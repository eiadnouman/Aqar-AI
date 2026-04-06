import pytest

from app.api.v1.map_tools import map_live_check_endpoint
from app.models.schemas import MapLiveCheckRequest


class _FakeMapService:
    enabled = True

    def geocode_area_center(self, _location):
        return (30.0444, 31.2357)

    def get_nearby_services(self, _lat, _lon, radius_m=None):
        _ = radius_m
        return ["schools", "hospitals", "transport"]


class _FakeRAGService:
    def __init__(self):
        self.map_intelligence = _FakeMapService()


@pytest.mark.anyio
async def test_map_live_check_endpoint_returns_live_status():
    response = await map_live_check_endpoint(
        MapLiveCheckRequest(location="New Cairo"),
        rag_engine=_FakeRAGService(),
    )

    assert response.enabled is True
    assert response.needs_api_key is False
    assert response.geocode_ok is True
    assert response.nearby_services_ok is True
    assert "schools" in response.nearby_services
