# Aqar AI (Egyptian Real Estate Assistant)

RAG-based real estate assistant for the Egyptian market, with:
- FastAPI backend (`/api/v1/*`)
- Streamlit frontend (`src/app.py`)
- Hybrid retrieval (FAISS + BM25)
- Arabic conversational recommendations
- Geo-aware ranking (distance + nearby services)
- Buy/Wait decision with best alternative suggestion (`/api/v1/analyze`)

## Project Structure

- `backend/`: FastAPI app, RAG orchestration, services
- `src/`: Streamlit frontend
- `data/`: FAISS index + property images
- `scripts/interactive_chat.py`: terminal smoke-test client
- `tests/`: automated tests

## Requirements

- Python 3.9+
- `.env` in project root with at least one model provider:

```env
GROQ_API_KEY=gsk_xxx
# Optional unless Groq is unavailable:
HUGGINGFACEHUB_API_TOKEN=hf_xxx

# Optional live map intelligence (geocode + nearby services):
MAP_API_ENABLED=true
MAP_GEOCODE_URL=https://nominatim.openstreetmap.org/search
MAP_OVERPASS_URL=https://overpass-api.de/api/interpreter
MAP_USER_AGENT=AqarAI/2.0
MAP_CONTACT_EMAIL=you@example.com
```

## Installation

```bash
python -m venv venv
source venv/bin/activate   # Linux/macOS
# venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

## Run Backend

From project root:

```bash
PYTHONPATH=backend uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

## Run Frontend

```bash
streamlit run src/app.py
```

## Unified Run Script

```bash
./run.sh
```

Advanced controls:

```bash
./scripts/manage.sh start all
./scripts/manage.sh stop all
./scripts/manage.sh restart backend
./scripts/manage.sh status
./scripts/manage.sh logs backend
```

## API Endpoints

- `POST /api/v1/chat`
- `POST /api/v1/search`
- `POST /api/v1/recommend/similar`
- `POST /api/v1/analyze`
- `POST /api/v1/map/live-check`
- `GET /health`
- `GET /docs`

## Quick Manual Test

```bash
python scripts/interactive_chat.py
```
