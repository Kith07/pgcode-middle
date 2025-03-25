import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    DEBUG: bool = os.getenv("DEBUG", False)
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", 8080))
    
    ENCODER_SERVICE_URL: str = os.getenv("ENCODER_SERVICE_URL")
    MIDDLE_SERVICE_URL: str = os.getenv("MIDDLE_SERVICE_URL")
    DECODER_SERVICE_URL: str = os.getenv("DECODER_SERVICE_URL")
    
    ALLOWED_ORIGINS: list = ["*"]
    
    ENCODER_TIMEOUT: int = 30
    MIDDLE_TIMEOUT: int = 30
    DECODER_TIMEOUT: int = 60
    
    class Config:
        env_file = ".env"

settings = Settings()