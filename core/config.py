"""
Configuration management for the agent system
"""

import os
from typing import Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field


class Config(BaseModel):
    """
    Centralized configuration for the agent system.
    
    Loads from environment variables (.env file).
    """
    
    # Database
    database_url: str = Field(..., description="PostgreSQL connection URL")
    db_pool_min: int = Field(default=1, description="Minimum DB connections")
    db_pool_max: int = Field(default=10, description="Maximum DB connections")
    
    # OpenAI / OpenRouter
    openrouter_api_key: str = Field(..., description="OpenRouter API key")
    openai_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        description="OpenAI API base URL"
    )
    openai_model: str = Field(
        default="openai/gpt-4o-mini",
        description="LLM model for analysis"
    )
    embedding_model: str = Field(
        default="openai/text-embedding-3-small",
        description="Embedding model"
    )
    
    # Agent Configuration
    max_retries: int = Field(default=3, description="Max retry attempts for failed tasks")
    retry_delay_seconds: int = Field(default=2, description="Delay between retries")
    query_timeout_seconds: int = Field(default=60, description="Query execution timeout")
    
    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
    
    @classmethod
    def load(cls) -> 'Config':
        """
        Load configuration from environment variables.
        
        Returns:
            Config instance
        """
        load_dotenv()
        
        return cls(
            database_url=os.getenv("DATABASE_URL", ""),
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY", ""),
            db_pool_min=int(os.getenv("DB_POOL_MIN", "1")),
            db_pool_max=int(os.getenv("DB_POOL_MAX", "10")),
            openai_base_url=os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1"),
            openai_model=os.getenv("OPENAI_MODEL", "openai/gpt-4o-mini"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "openai/text-embedding-3-small"),
            max_retries=int(os.getenv("MAX_RETRIES", "3")),
            retry_delay_seconds=int(os.getenv("RETRY_DELAY_SECONDS", "2")),
            query_timeout_seconds=int(os.getenv("QUERY_TIMEOUT_SECONDS", "60")),
            log_level=os.getenv("LOG_LEVEL", "INFO")
        )
    
    def validate_required(self):
        """
        Validate that required configuration is present.
        
        Raises:
            ValueError: If required config is missing
        """
        if not self.database_url:
            raise ValueError("DATABASE_URL is required in .env file")
        if not self.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY is required in .env file")


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """
    Get the global configuration instance.
    
    Returns:
        Config instance
    """
    global _config
    if _config is None:
        _config = Config.load()
        _config.validate_required()
    return _config
