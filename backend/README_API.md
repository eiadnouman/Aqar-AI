# Aqar-AI Backend API

## Overview

FastAPI service that powers:
- conversational chat recommendations
- headless property search
- similar property recommendations

Base path for all functional endpoints: `/api/v1`.

## Run

From project root:

```bash
PYTHONPATH=backend uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

or from `backend/` directory:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Environment

`.env` should include one provider at minimum:

```env
GROQ_API_KEY=gsk_xxx
# Optional fallback token (used when Groq is unavailable):
HUGGINGFACEHUB_API_TOKEN=hf_xxx
```

Optional:

```env
FAISS_INDEX_PATH=data/faiss_index_cloud

# Live Map API enrichment
MAP_API_ENABLED=true
MAP_GEOCODE_URL=https://nominatim.openstreetmap.org/search
MAP_OVERPASS_URL=https://overpass-api.de/api/interpreter
MAP_USER_AGENT=AqarAI/2.0
MAP_CONTACT_EMAIL=you@example.com
MAP_TIMEOUT_SEC=8
MAP_RADIUS_M=2000
```

## Endpoints

### `POST /api/v1/chat`

```json
{
  "message": "عايز شقة في التجمع بـ 5 مليون",
  "session_id": "user-123"
}
```

Notes:
- لو الرسالة فيها intent تحليلي (مثال: `أشتري ولا لأ؟` أو `حلل السوق في التجمع`) الشات نفسه دلوقتي بيرجع:
  - قرار شراء عملي (`buy / wait / not now`)
  - أسباب القرار
  - أفضل بديل متاح إن وجد

### `POST /api/v1/search`

```json
{
  "query": "شقة 3 غرف في التجمع"
}
```

### `POST /api/v1/recommend/similar`

```json
{
  "property_description": "Modern apartment in New Cairo"
}
```

### `POST /api/v1/analyze`

```json
{
  "query": "حلل شقق التجمع الخامس تحت 4 مليون قريبة من المدارس والمواصلات"
}
```

Returns:
- market stats (price/size/price-per-sqm)
- segmented breakdown (top locations/types)
- `buy_decision` with `decision` + confidence + reasons
- `better_option` (if available) chosen by affordability + distance + services scoring

### `POST /api/v1/map/live-check`

Use this endpoint to validate live map integration from the backend runtime.

```json
{
  "location": "The 5th Settlement",
  "radius_m": 2000
}
```

Notes:
- Default provider is OpenStreetMap (`Nominatim` + `Overpass`).
- No API key is required by default.
- Add `MAP_CONTACT_EMAIL` for Nominatim policy compliance.

System:
- `GET /health`
- `GET /docs`
