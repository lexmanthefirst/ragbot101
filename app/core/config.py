from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    PROJECT_NAME: str = "RAG Chatbot"
    API_V1_STR: str = "/api/v1"
    
    # Database - PostgreSQL
    DATABASE_URL: str
    
    # Database Pool Settings
    DB_POOL_SIZE: int = 5
    DB_MAX_OVERFLOW: int = 10
    DB_POOL_TIMEOUT: int = 30
    DB_POOL_RECYCLE: int = 3600
    DB_ECHO: bool = False
    
    # Security
    SECRET_KEY: str
    
    # CORS
    CORS_ORIGINS: list[str] = ["http://localhost:3000", "http://localhost:8000"]
    
    # Vector DB - Local ChromaDB
    VECTOR_DB_TYPE: str = "chroma"
    CHROMA_DB_DIR: str = "./chroma_db"
    
    # Vector DB - ChromaDB Cloud
    CHROMA_API_KEY: Optional[str] = None
    CHROMA_TENANT: Optional[str] = None
    CHROMA_DATABASE: Optional[str] = None

    # LLM - OpenRouter
    OPENROUTER_API_KEY: str
    OPENROUTER_MODEL: str = "openai/gpt-3.5-turbo"
    EMBEDDING_MODEL: str = "text-embedding-ada-002"
    
    # LLM Fallback Models (optional)
    FALLBACK_MODEL_1: Optional[str] = None
    FALLBACK_MODEL_2: Optional[str] = None
    FALLBACK_MODEL_3: Optional[str] = None
    FALLBACK_MODEL_4: Optional[str] = None
    FALLBACK_MODEL_5: Optional[str] = None
    
    # Logging
    LOG_LEVEL: str = "INFO"

    class Config:
        case_sensitive = True
        env_file = ".env"


settings = Settings()
