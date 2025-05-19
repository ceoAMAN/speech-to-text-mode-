import os
from pydantic_settings import BaseSettings
from typing import Literal

class Settings(BaseSettings):
    """Application settings."""
    # API Settings
    API_V1_STR: str = "/api"
    PROJECT_NAME: str = "Speech-to-Text API"
    
    # Whisper Model Settings
    MODEL_SIZE: Literal["tiny", "base", "small", "medium", "large"] = "base"
    DEVICE: str = None  # Will be auto-detected based on availability
    
    # File Storage
    UPLOAD_DIR: str = "uploads"
    MAX_UPLOAD_SIZE: int = 25 * 1024 * 1024  # 25MB
    
    # Supported audio formats
    ALLOWED_AUDIO_TYPES: list[str] = [
        "audio/mpeg", "audio/mp3", "audio/wav", "audio/x-wav",
        "audio/ogg", "audio/flac", "audio/x-flac"
    ]
    
    # CORS Settings
    CORS_ORIGINS: list[str] = ["*"]
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Create settings instance
settings = Settings()

# Ensure upload directory exists
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

# Auto-detect device (CPU/GPU)
import torch
if settings.DEVICE is None:
    settings.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"