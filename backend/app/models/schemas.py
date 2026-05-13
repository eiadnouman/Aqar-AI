from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class Property(BaseModel):
    """Schema representing a Real Estate property."""
    id: Optional[int] = Field(None, description="Internal property ID, when available")
    title: str
    location: str
    price: float
    bedrooms: int
    bathrooms: int
    size: float
    image_url: str
    description: str
    url: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    distance_km: Optional[float] = None
    nearby_services: List[str] = Field(default_factory=list)
    recommendation_score: Optional[float] = None

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
    query: Optional[str] = Field(None, description="Search query or filters in natural language")
    location: Optional[str] = Field(None, description="Target location")
    min_price: Optional[float] = Field(None, description="Minimum price boundary")
    max_price: Optional[float] = Field(None, description="Maximum price boundary")
    min_bedrooms: Optional[int] = Field(None, description="Minimum bedrooms")
    max_bedrooms: Optional[int] = Field(None, description="Maximum bedrooms")
    property_type: Optional[str] = Field(None, description="Property type filter")
    desired_services: Optional[List[str]] = Field(None, description="Preferred nearby services")

class SearchResponse(BaseModel):
    """Response containing strictly property search results."""
    properties: List[Property]
    filters_used: Optional[Dict[str, Any]] = None

class RecommendRequest(BaseModel):
    """Payload for semantic similarity searches."""
    property_description: str = Field(..., description="Text description of the target property")
    session_id: Optional[str] = Field(None, description="Optional session ID for interaction-aware ranking")
    property_ids: Optional[List[int]] = Field(None, description="Optional clicked/favorited property IDs")

class RecommendResponse(BaseModel):
    """Response containing similar properties."""
    properties: List[Property]


class PropertyInteractionRequest(BaseModel):
    """Payload for storing lightweight user interest signals."""
    session_id: str = Field(..., description="Anonymous or authenticated session ID")
    property_id: int = Field(..., description="Property ID the user interacted with")
    event_type: str = Field("click", description="Interaction type such as click, save, view")


class PropertyInteractionResponse(BaseModel):
    """Response after saving an interaction event."""
    saved: bool
    session_id: str
    property_ids: List[int]


class SessionRecommendRequest(BaseModel):
    """Payload for recommendations from session interaction history."""
    session_id: str = Field(..., description="Session ID with stored property interactions")
    property_ids: Optional[List[int]] = Field(None, description="Optional additional property IDs")
    limit: int = Field(5, ge=1, le=20, description="Maximum properties to return")


class AnalyzeRequest(BaseModel):
    """Payload for market analysis requests."""
    query: Optional[str] = Field(None, description="Optional natural language analysis query")
    location: Optional[str] = Field(None, description="Target location to analyze")
    min_price: Optional[float] = Field(None, description="Minimum price boundary")
    max_price: Optional[float] = Field(None, description="Maximum price boundary")
    min_bedrooms: Optional[int] = Field(None, description="Minimum bedrooms")
    max_bedrooms: Optional[int] = Field(None, description="Maximum bedrooms")
    property_type: Optional[str] = Field(None, description="Property type filter")
    desired_services: Optional[List[str]] = Field(None, description="Preferred nearby services")


class SegmentStat(BaseModel):
    """Grouped aggregation stats for a categorical segment."""
    name: str
    count: int
    avg_price: float
    median_price: float
    avg_price_per_sqm: float


class BuyDecision(BaseModel):
    """Structured buy/no-buy recommendation based on analyzed inventory."""
    decision: str
    headline: str
    confidence: float
    reasons: List[str]


class AnalyzeResponse(BaseModel):
    """Response containing market analysis insights."""
    insight: str
    filters_used: Dict[str, Any]
    match_scope: str
    total_candidates: int
    matched_count: int
    stats: Dict[str, Any]
    top_locations: List[SegmentStat]
    top_property_types: List[SegmentStat]
    buy_decision: BuyDecision
    better_option_found: bool
    better_option_reason: str
    better_option: Optional[Property] = None
    sample_properties: List[Property]


class MapLiveCheckRequest(BaseModel):
    """Payload to verify live map API connectivity and outputs."""
    location: Optional[str] = Field("New Cairo", description="Area name to geocode")
    latitude: Optional[float] = Field(None, description="Optional direct latitude override")
    longitude: Optional[float] = Field(None, description="Optional direct longitude override")
    radius_m: Optional[int] = Field(None, description="Radius for nearby services lookup")


class MapLiveCheckResponse(BaseModel):
    """Result of a live map API verification call."""
    enabled: bool
    provider: str
    needs_api_key: bool
    geocode_ok: bool
    nearby_services_ok: bool
    resolved_location: Optional[str] = None
    resolved_center: Optional[Dict[str, float]] = None
    nearby_services: List[str] = Field(default_factory=list)
    note: str
