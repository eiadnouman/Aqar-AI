from fastapi import APIRouter, Depends, HTTPException

from app.models.schemas import MapLiveCheckRequest, MapLiveCheckResponse
from app.services.rag_service import RAGService, get_rag_service
import logging

logger = logging.getLogger("aqarai")
router = APIRouter()


@router.post("/map/live-check", response_model=MapLiveCheckResponse)
async def map_live_check_endpoint(
    request: MapLiveCheckRequest,
    rag_engine: RAGService = Depends(get_rag_service),
):
    """
    Executes a live geocode + nearby-services check against configured Map APIs.
    """
    try:
        map_service = rag_engine.map_intelligence
        if not map_service.enabled:
            return MapLiveCheckResponse(
                enabled=False,
                provider="OpenStreetMap (Nominatim + Overpass)",
                needs_api_key=False,
                geocode_ok=False,
                nearby_services_ok=False,
                note="Map API integration is disabled. Set MAP_API_ENABLED=true in .env then retry.",
            )

        location = (request.location or "New Cairo").strip()
        center = map_service.geocode_area_center(location)
        geocode_ok = center is not None

        lat = request.latitude if request.latitude is not None else (center[0] if center else None)
        lon = request.longitude if request.longitude is not None else (center[1] if center else None)

        services = []
        nearby_ok = False
        if lat is not None and lon is not None:
            services = map_service.get_nearby_services(lat, lon, radius_m=request.radius_m)
            nearby_ok = len(services) > 0

        note = (
            "Live check succeeded."
            if geocode_ok
            else "Geocode failed for this location. Try a clearer area name or verify outbound internet access."
        )
        if geocode_ok and not nearby_ok:
            note = (
                "Geocode succeeded, but no nearby services returned in current radius/provider response."
            )

        return MapLiveCheckResponse(
            enabled=True,
            provider="OpenStreetMap (Nominatim + Overpass)",
            needs_api_key=False,
            geocode_ok=geocode_ok,
            nearby_services_ok=nearby_ok,
            resolved_location=location,
            resolved_center={"latitude": lat, "longitude": lon} if lat is not None and lon is not None else None,
            nearby_services=services,
            note=note,
        )
    except Exception as e:
        logger.error(f"Error in map live-check endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))
