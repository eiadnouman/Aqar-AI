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
