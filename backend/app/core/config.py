import os
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application wide settings managed via Environment Variables.
    Pydantic will automatically validate and map them.
    """
    # Optional API keys (validated at use time)
    huggingfacehub_api_token: Optional[str] = None
    
    # Optional API Keys
    groq_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    
    # Internal Paths
    faiss_index_path: str = "data/faiss_index_cloud"

    # Map API integration (optional but recommended for geospatial precision)
    map_api_enabled: bool = False
    map_geocode_url: str = "https://nominatim.openstreetmap.org/search"
    map_overpass_url: str = "https://overpass-api.de/api/interpreter"
    map_user_agent: str = "AqarAI/2.0 (map-intelligence)"
    map_contact_email: Optional[str] = None
    map_timeout_sec: int = 8
    map_radius_m: int = 2000
    map_max_docs_per_rank: int = 5

    # Optional sync from the graduation_project_api service
    external_api_base_url: Optional[str] = None
    internal_api_key: Optional[str] = None
    external_api_timeout_sec: int = 5
    property_public_base_url: Optional[str] = None
    interaction_cache_ttl_sec: int = 3600
    cors_allowed_origins: str = "*"

    # Runtime performance knobs
    fast_filter_extraction: bool = True
    fast_property_responses: bool = True
    chat_retrieval_k: int = 60
    search_retrieval_k: int = 80
    llm_max_tokens: int = 700
    
    # Application Context
    project_name: str = "Aqar-AI Intelligence API"
    version: str = "2.0.0"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

def get_settings() -> Settings:
    """
    Instantiate settings and ensure fallback paths are resolved
    during local development and production.
    """
    # Resolve default paths relative to root during dev
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    default_faiss = os.path.join(base_dir, "data", "faiss_index_cloud")
    
    if not os.getenv("FAISS_INDEX_PATH"):
        os.environ["FAISS_INDEX_PATH"] = default_faiss
        
    return Settings()

settings = get_settings()
