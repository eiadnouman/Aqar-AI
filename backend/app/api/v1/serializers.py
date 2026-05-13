import re
from typing import Optional
from urllib.parse import urlparse, urlunparse

from app.core.config import settings
from app.models.schemas import Property


def safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def safe_int(value, default=0):
    try:
        return int(float(value))
    except Exception:
        return default


def resolve_property_id(meta: dict) -> Optional[int]:
    """Resolve the app's property_id while tolerating older FAISS metadata."""
    for key in ("property_id", "id", "apartment_id", "source_id", "source_index"):
        value = meta.get(key)
        if value not in (None, ""):
            resolved = safe_int(value, default=0)
            if resolved > 0:
                return resolved

    url = str(meta.get("url") or "")
    url_match = re.search(r"/property/(\d+)(?:/|$)", url)
    if url_match:
        return int(url_match.group(1))

    image_value = str(meta.get("image") or meta.get("image_url") or "")
    image_match = re.search(r"property[_/-](\d+)", image_value)
    if image_match:
        return int(image_match.group(1))

    listing_match = re.search(r"-(\d+)\.html(?:\?|$)", url)
    if listing_match:
        return int(listing_match.group(1))

    return None


def _external_base_url() -> str:
    return (settings.property_public_base_url or settings.external_api_base_url or "").rstrip("/")


def normalize_external_url(value: str) -> str:
    raw = str(value or "").strip()
    base_url = _external_base_url()
    if not raw:
        return raw

    if raw.startswith(("http://", "https://")):
        parsed = urlparse(raw)
        if parsed.hostname in {"localhost", "127.0.0.1", "0.0.0.0"} and base_url:
            base = urlparse(base_url)
            return urlunparse(parsed._replace(scheme=base.scheme, netloc=base.netloc))
        return raw

    if base_url and raw.startswith(("uploads/", "property/", "api/", "static/")):
        return f"{base_url}/{raw.lstrip('/')}"

    return raw


def doc_to_property(doc) -> Property:
    meta = doc.metadata if isinstance(doc.metadata, dict) else {}
    return Property(
        id=resolve_property_id(meta),
        title=meta.get("title", "Property Listing"),
        location=meta.get("location", "Unknown Location"),
        price=safe_float(meta.get("price", 0)),
        bedrooms=safe_int(meta.get("bedrooms", 0)),
        bathrooms=safe_int(meta.get("bathrooms", 0)),
        size=safe_float(meta.get("size", 0)),
        image_url=normalize_external_url(meta.get("image", "")),
        description=doc.page_content.split("Description: ")[-1][:200] + "...",
        url=normalize_external_url(meta.get("url", "#")),
        latitude=safe_float(meta.get("lat") or meta.get("latitude") or 0) or None,
        longitude=safe_float(meta.get("lon") or meta.get("longitude") or 0) or None,
        distance_km=safe_float(meta.get("distance_km", 0)) or None,
        nearby_services=list(meta.get("nearby_services", []) or []),
        recommendation_score=safe_float(meta.get("recommendation_score", 0)) or None,
    )
