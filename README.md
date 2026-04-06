# Aqar AI (Egyptian Real Estate Assistant)

RAG-based real estate assistant for the Egyptian market, with:
- FastAPI backend (`/api/v1/*`)
- Streamlit frontend (`src/app.py`)
- Hybrid retrieval (FAISS + BM25)
- Arabic conversational recommendations

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

## API Endpoints

- `POST /api/v1/chat`
- `POST /api/v1/search`
- `POST /api/v1/recommend/similar`
- `GET /health`
- `GET /docs`

## Quick Manual Test

```bash
python scripts/interactive_chat.py
```
