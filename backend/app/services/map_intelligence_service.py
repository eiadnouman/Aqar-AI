from typing import Dict, List, Optional, Set, Tuple

import requests

from app.core.config import settings
from app.core.logging import logger


class MapIntelligenceService:
    """
    Optional live map intelligence using external APIs:
    - Geocoding area names to real coordinates
    - Fetching nearby real services/POIs
    """

    def __init__(self):
        self.enabled = settings.map_api_enabled
        self.geocode_url = settings.map_geocode_url
        self.overpass_url = settings.map_overpass_url
        self.user_agent = settings.map_user_agent
        self.contact_email = (settings.map_contact_email or "").strip()
        self.timeout_sec = max(2, int(settings.map_timeout_sec))
        self.radius_m = max(300, int(settings.map_radius_m))
        # Allow disabling live POI calls during ranking by setting MAP_MAX_DOCS_PER_RANK=0
        self.max_docs_per_rank = max(0, int(settings.map_max_docs_per_rank))

        self._geocode_cache: Dict[str, Optional[Tuple[float, float]]] = {}
        self._services_cache: Dict[str, List[str]] = {}

        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.user_agent})

        # Common aliases that improve geocoding for Egypt market terms.
        self._location_aliases = {
            "the 5th settlement": "Fifth Settlement, New Cairo, Cairo, Egypt",
            "the 1st settlement": "First Settlement, New Cairo, Cairo, Egypt",
            "new cairo": "New Cairo, Cairo, Egypt",
            "sheikh zayed": "Sheikh Zayed City, Giza, Egypt",
            "october": "6th of October City, Giza, Egypt",
            "north coast": "North Coast, Matrouh, Egypt",
            "new capital": "New Administrative Capital, Cairo, Egypt",
            "التجمع الخامس": "Fifth Settlement, New Cairo, Cairo, Egypt",
            "التجمع الاول": "First Settlement, New Cairo, Cairo, Egypt",
            "التجمع": "New Cairo, Cairo, Egypt",
            "القاهرة الجديدة": "New Cairo, Cairo, Egypt",
            "الشيخ زايد": "Sheikh Zayed City, Giza, Egypt",
            "زايد": "Sheikh Zayed City, Giza, Egypt",
            "اكتوبر": "6th of October City, Giza, Egypt",
            "٦ اكتوبر": "6th of October City, Giza, Egypt",
            "الساحل الشمالي": "North Coast, Matrouh, Egypt",
            "العاصمة الادارية": "New Administrative Capital, Cairo, Egypt",
        }

    def geocode_area_center(self, location_name: str) -> Optional[Tuple[float, float]]:
        if not self.enabled:
            return None
        if not location_name:
            return None

        key = location_name.strip().lower()
        if key in self._geocode_cache:
            return self._geocode_cache[key]

        query = self._location_aliases.get(key, location_name.strip())
        if "egypt" not in query.lower():
            query = f"{query}, Egypt"

        try:
            response = self.session.get(
                self.geocode_url,
                params=self._build_geocode_params(query),
                timeout=self.timeout_sec,
            )
            response.raise_for_status()
            payload = response.json()
            if not payload:
                self._geocode_cache[key] = None
                return None

            lat = self._safe_float(payload[0].get("lat"))
            lon = self._safe_float(payload[0].get("lon"))
            if lat is None or lon is None:
                self._geocode_cache[key] = None
                return None

            self._geocode_cache[key] = (lat, lon)
            return lat, lon
        except Exception as e:
            logger.warning(f"Map geocoding failed for '{location_name}': {e}")
            self._geocode_cache[key] = None
            return None

    def get_nearby_services(self, lat: float, lon: float, radius_m: Optional[int] = None) -> List[str]:
        if not self.enabled:
            return []
        if lat is None or lon is None:
            return []

        radius = int(radius_m or self.radius_m)
        cache_key = f"{round(lat, 4)}:{round(lon, 4)}:{radius}"
        if cache_key in self._services_cache:
            return self._services_cache[cache_key]

        overpass_query = f"""
        [out:json][timeout:20];
        (
          nwr(around:{radius},{lat},{lon})[amenity~"school|university|hospital|clinic|pharmacy|police"];
          nwr(around:{radius},{lat},{lon})[shop~"mall|supermarket"];
          nwr(around:{radius},{lat},{lon})[public_transport];
          nwr(around:{radius},{lat},{lon})[railway~"station|subway_entrance"];
          nwr(around:{radius},{lat},{lon})[leisure~"park|sports_centre|garden"];
        );
        out tags;
        """

        try:
            response = self.session.post(
                self.overpass_url,
                data={"data": overpass_query},
                timeout=self.timeout_sec,
            )
            response.raise_for_status()
            payload = response.json()
            services: Set[str] = set()
            for element in payload.get("elements", []):
                tags = element.get("tags", {}) or {}
                mapped = self._map_tags_to_services(tags)
                services.update(mapped)

            result = sorted(services)
            self._services_cache[cache_key] = result
            return result
        except Exception as e:
            logger.warning(f"Nearby services API failed for ({lat}, {lon}): {e}")
            self._services_cache[cache_key] = []
            return []

    def _map_tags_to_services(self, tags: Dict[str, str]) -> Set[str]:
        services: Set[str] = set()

        amenity = (tags.get("amenity") or "").lower()
        shop = (tags.get("shop") or "").lower()
        railway = (tags.get("railway") or "").lower()
        leisure = (tags.get("leisure") or "").lower()
        has_transport = "public_transport" in tags

        if amenity in {"school", "university"}:
            services.add("schools")
        if amenity in {"hospital", "clinic", "pharmacy"}:
            services.add("hospitals")
        if amenity in {"police"}:
            services.add("security")

        if shop in {"mall", "supermarket"}:
            services.add("commercial_area")

        if has_transport or railway in {"station", "subway_entrance"}:
            services.add("transport")

        if leisure in {"park", "garden"}:
            services.add("green_spaces")
        if leisure in {"sports_centre"}:
            services.add("club_house")

        return services

    @staticmethod
    def _safe_float(value) -> Optional[float]:
        try:
            return float(value)
        except Exception:
            return None

    def _build_geocode_params(self, query: str) -> Dict[str, str]:
        params = {"q": query, "format": "jsonv2", "limit": 1}
        if self.contact_email:
            params["email"] = self.contact_email
        return params
