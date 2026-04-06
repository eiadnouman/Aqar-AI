from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class Property(BaseModel):
    """Schema representing a Real Estate property."""
    title: str
    location: str
    price: float
    bedrooms: int
    bathrooms: int
    size: float
    image_url: str
    description: str
    url: str

class ChatRequest(BaseModel):
    """Payload for conversational queries."""
    message: str = Field(..., description="The user's query about real estate")
    session_id: Optional[str] = Field(None, description="Optional session ID to persist memory context")

class ChatResponse(BaseModel):
    """Response containing AI analysis and properties."""
    answer: str
    properties: List[Property]
    
class SearchRequest(BaseModel):
    """Payload for headless search queries."""
    query: str = Field(..., description="Search query or filters in natural language")

class SearchResponse(BaseModel):
    """Response containing strictly property search results."""
    properties: List[Property]
    filters_used: Optional[Dict[str, Any]] = None

class RecommendRequest(BaseModel):
    """Payload for semantic similarity searches."""
    property_description: str = Field(..., description="Text description of the target property")

class RecommendResponse(BaseModel):
    """Response containing similar properties."""
    properties: List[Property]
