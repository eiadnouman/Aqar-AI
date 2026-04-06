from app.services.map_intelligence_service import MapIntelligenceService


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_geocode_area_center_uses_api_payload(monkeypatch):
    svc = MapIntelligenceService()
    svc.enabled = True

    def _fake_get(_url, params=None, timeout=None):  # noqa: ANN001
        _ = params, timeout
        return _FakeResponse([{"lat": "30.0444", "lon": "31.2357"}])

    monkeypatch.setattr(svc.session, "get", _fake_get)

    center = svc.geocode_area_center("New Cairo")
    assert center == (30.0444, 31.2357)


def test_get_nearby_services_maps_osm_tags(monkeypatch):
    svc = MapIntelligenceService()
    svc.enabled = True

    payload = {
        "elements": [
            {"tags": {"amenity": "school"}},
            {"tags": {"amenity": "hospital"}},
            {"tags": {"shop": "mall"}},
            {"tags": {"public_transport": "platform"}},
            {"tags": {"leisure": "park"}},
        ]
    }

    def _fake_post(_url, data=None, timeout=None):  # noqa: ANN001
        _ = data, timeout
        return _FakeResponse(payload)

    monkeypatch.setattr(svc.session, "post", _fake_post)

    services = svc.get_nearby_services(30.0, 31.0, radius_m=1500)
    assert "schools" in services
    assert "hospitals" in services
    assert "commercial_area" in services
    assert "transport" in services
    assert "green_spaces" in services
