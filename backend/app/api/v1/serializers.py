import json
import re
from typing import Any, List, Optional
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


def safe_bool(value) -> Optional[bool]:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return None


def safe_list(value: Any) -> List[str]:
    if value in (None, ""):
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item not in (None, "")]
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("["):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    return [str(item) for item in parsed if item not in (None, "")]
            except Exception:
                pass
        return [text] if text else []
    return [str(value)]


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
    """Normalizes URLs. For image/upload paths, returns just the relative path.
    The frontend is responsible for building the full URL for images."""
    raw = str(value or "").strip()
    if not raw:
        return raw

    # File resource prefixes — these should return path only (no base URL)
    _resource_prefixes = ("uploads/", "static/")

    if raw.startswith(("http://", "https://")):
        parsed = urlparse(raw)
        path = parsed.path.lstrip("/")
        # If the path looks like a file resource (image/upload), return just the path
        if any(path.startswith(prefix) for prefix in _resource_prefixes):
            return path
        # For localhost URLs, rewrite to external base
        base_url = _external_base_url()
        if parsed.hostname in {"localhost", "127.0.0.1", "0.0.0.0"} and base_url:
            base = urlparse(base_url)
            return urlunparse(parsed._replace(scheme=base.scheme, netloc=base.netloc))
        return raw

    # Already a relative resource path — return as-is
    if any(raw.startswith(prefix) for prefix in _resource_prefixes):
        return raw

    return raw


def doc_to_property(doc) -> Property:
    meta = doc.metadata if isinstance(doc.metadata, dict) else {}
    description = str(meta.get("property_desc") or meta.get("description") or doc.page_content.split("Description: ")[-1])
    images = [normalize_external_url(item) for item in safe_list(meta.get("images"))]
    image_url = normalize_external_url(meta.get("image") or meta.get("image_url") or (images[0] if images else ""))
    if image_url and image_url not in images:
        images.insert(0, image_url)

    property_id = resolve_property_id(meta)
    property_name = meta.get("property_name") or meta.get("title", "Property Listing")
    price_value = safe_float(meta.get("price_value") or meta.get("price") or 0)
    price_per_day = safe_float(meta.get("price_per_day") or meta.get("price") or 0)
    bedrooms_no = safe_int(meta.get("bedrooms_no") or meta.get("bedrooms") or 0)
    bathrooms_no = safe_int(meta.get("bathrooms_no") or meta.get("bathrooms") or 0)

    return Property(
        property_id=property_id,
        owner_id=meta.get("owner_id"),
        property_name=property_name,
        property_desc=description,
        pricing_unit=meta.get("pricing_unit"),
        price_value=price_value,
        price_per_day=price_per_day,
        bedrooms_no=bedrooms_no,
        beds_no=safe_int(meta.get("beds_no") or meta.get("beds") or bedrooms_no or 0),
        bathrooms_no=bathrooms_no,
        images=images,
        ownership_proofs=[normalize_external_url(item) for item in safe_list(meta.get("ownership_proofs"))],
        listing_status=meta.get("listing_status"),
        listing_expiry=str(meta.get("listing_expiry")) if meta.get("listing_expiry") else None,
        is_visible=safe_bool(meta.get("is_visible")),
        is_verified=safe_bool(meta.get("is_verified")),
        is_available=safe_bool(meta.get("is_available")),
        is_furnished=safe_bool(meta.get("is_furnished")),
        is_sponsored=safe_bool(meta.get("is_sponsored")),
        property_type=meta.get("property_type"),
        rate=safe_float(meta.get("rate", 0)) if meta.get("rate") not in (None, "") else None,
        id=property_id,
        title=property_name,
        location=meta.get("location", "Unknown Location"),
        price=price_value,
        bedrooms=bedrooms_no,
        bathrooms=bathrooms_no,
        size=safe_float(meta.get("size", 0)),
        image_url=image_url,
        description=description[:200] + "...",
        url=normalize_external_url(meta.get("url", "#")),
        latitude=safe_float(meta.get("lat") or meta.get("latitude") or 0) or None,
        longitude=safe_float(meta.get("lon") or meta.get("longitude") or 0) or None,
        distance_km=safe_float(meta.get("distance_km", 0)) or None,
        nearby_services=list(meta.get("nearby_services", []) or []),
        recommendation_score=safe_float(meta.get("recommendation_score", 0)) or None,
    )
