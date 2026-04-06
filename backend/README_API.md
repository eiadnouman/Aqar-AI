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
```

## Endpoints

### `POST /api/v1/chat`

```json
{
  "message": "عايز شقة في التجمع بـ 5 مليون",
  "session_id": "user-123"
}
```

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

System:
- `GET /health`
- `GET /docs`
