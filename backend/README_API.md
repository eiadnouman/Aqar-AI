# Aqar-AI Intelligence Microservice (API)

## Overview
This is the core AI service for Aqar-AI. It provides a RESTful API to:
1.  **Understand User Queries**: Uses Llama 3 (via Groq) to extract search filters (Location, Price, etc.) from natural language.
2.  **Retrieve Properties**: Searches the FAISS Vector Database for matching properties.
3.  **Generate Responses**: Returns a friendly, persona-driven Arabic response + structured JSON data for the Frontend.

## Setup & Run

### 1. Requirements
Ensure you have the environment variables set in `.env` (or `backend/.env`):
```bash
HUGGINGFACEHUB_API_TOKEN=...
GROQ_API_KEY=...
FAISS_INDEX_PATH=../data/faiss_index_cloud  # Adjust if needed
```

### 2. Install Dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 3. Run the Server
```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## API Documentation
Once running, visit **http://localhost:8000/docs** for the Swagger UI.

### Endpoint: `POST /v1/chat`

**Request:**
```json
{
  "message": "عايز شقة في التجمع بـ 5 مليون",
  "session_id": "optional-user-id"
}
```

**Response:**
```json
{
  "answer": "اهلا بيك! لقيتلك شقق مميزة في التجمع الخامس...",
  "properties": [
    {
      "title": "Apartment in Fifth Settlement",
      "location": "New Cairo",
      "price": 4500000,
      "bedrooms": 3,
      "image_url": "images/property_10/1.jpg",
      "description": "..."
    }
  ]
}
```
*Note: If the AI is asking for more details (e.g., "What is your budget?"), the `properties` list will be empty `[]`.*

## AI Logic
- **Smart Filtering**: The system tries to extract structured filters (e.g., `min_bedrooms: 3`) using the LLM. If that fails, it falls back to Regex.
- **Human-like Persona**: The AI acts as a consultant. It will not show cards immediately unless the user's intent is clear or they ask for them. It may ask clarifying questions first.
- **Dynamic Location Mapping 🌍**: The AI reads all unique locations from your database at startup. If you add "New Capital" to your CSV/Database, the AI learned it automatically! No code changes needed.
- **RAG**: It retrieves the top k documents relative to the query and filters them strictly based on the extracted criteria.
