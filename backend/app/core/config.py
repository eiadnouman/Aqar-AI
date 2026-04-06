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
