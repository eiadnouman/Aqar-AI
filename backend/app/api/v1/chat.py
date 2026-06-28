import json

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from app.api.v1.serializers import doc_to_property
from app.models.schemas import ChatRequest, ChatResponse
from app.services.rag_service import RAGService, get_rag_service
import logging

logger = logging.getLogger("aqarai")
router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, rag_engine: RAGService = Depends(get_rag_service)):
    """Handles natural language conversational queries."""
    try:
        logger.info(f"Processing Chat Request: {request.message} (Session: {request.session_id})")
        response_text, docs = rag_engine.get_recommendation(request.message, session_id=request.session_id)
        
        properties = [doc_to_property(doc) for doc in docs]
            
        return ChatResponse(
            answer=response_text,
            properties=properties,
        )
        
    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest, rag_engine: RAGService = Depends(get_rag_service)):
    """Streams conversational responses via Server-Sent Events (SSE).
    
    Event types:
    - token:      {"text": "chunk of text"}
    - properties: {"properties": [{...}, ...]}
    - done:       {}
    """
    logger.info(f"Processing Stream Chat Request: {request.message} (Session: {request.session_id})")

    def event_generator():
        try:
            for event in rag_engine.get_recommendation_stream(
                request.message, session_id=request.session_id
            ):
                event_type = event.get("event", "token")
                data = event.get("data", {})

                # Convert Document objects to serializable Property dicts
                if event_type == "properties":
                    raw_docs = data.get("docs")
                    if raw_docs is not None:
                        data = {"properties": [
                            doc_to_property(doc).model_dump() for doc in raw_docs
                        ]}
                    # else data already has {"properties": []}

                yield f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
        except Exception as e:
            logger.error(f"SSE stream error: {e}")
            error_data = {"text": "حدث خطأ أثناء المعالجة. حاول مرة تانية."}
            yield f"event: token\ndata: {json.dumps(error_data, ensure_ascii=False)}\n\n"
            yield f"event: properties\ndata: {json.dumps({'properties': []})}\n\n"
            yield f"event: done\ndata: {{}}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
