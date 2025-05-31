# In app/core/config.py
import os
from pydantic import BaseModel
from dotenv import load_dotenv

# Load .env file if it exists (for local development)
load_dotenv()

class Settings(BaseModel):
    """Application settings."""
    # API Settings
    API_KEY: str = os.environ.get("ML_API_KEY", "")
    ENVIRONMENT: str = os.environ.get("ENVIRONMENT", "development")
    
    # Model Settings
    MODEL_PATH: str = os.environ.get("MODEL_PATH", "./models")
    
    # Service Settings
    LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO")
    MAX_REQUESTS_PER_MINUTE: int = int(os.environ.get("MAX_REQUESTS_PER_MINUTE", "100"))
    
    # CORS Settings
    ALLOWED_ORIGINS: list = os.environ.get("ALLOWED_ORIGINS", "https://server-jantra-808666915194.asia-south2.run.app,https://www.jwantra.com,https://jwantra.com,http://localhost:3000").split(",")
    
    # Database Settings (if applicable)
    DB_URL: str = os.environ.get("DB_URL", "")
    
    model_config = {
        "env_file": ".env",
        "case_sensitive": True,
    }

# Create a global settings object
settings = Settings()