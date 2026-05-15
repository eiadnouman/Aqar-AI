from pydantic import BaseModel, ConfigDict, Field
from typing import List, Optional, Any

class ChatRequest(BaseModel):
    message: str = Field(..., example="Show me apartments in New Cairo under 5M")
    session_id: Optional[str] = Field(None, example="user_12345")

class Property(BaseModel):
    title: str
    location: str
    price: float
    bedrooms: int
    bathrooms: int
    size: float
    image_url: str
    description: str
    url: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    properties: List[Property] = []
    filters_used: Optional[dict] = None

class SearchRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "عاوز شقة في التجمع الخامس 3 غرف تحت 5 مليون قريبة من المدارس"
            }
        }
    )

    query: str = Field(..., example="عاوز شقة في التجمع الخامس")
    
class SearchResponse(BaseModel):
    properties: List[Property] = []
    filters_used: Optional[dict] = None

class RecommendRequest(BaseModel):
    property_description: str = Field(..., example="A luxurious villa with a sea view in North Coast")

class RecommendResponse(BaseModel):
    properties: List[Property] = []
