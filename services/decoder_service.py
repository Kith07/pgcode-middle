import requests
import logging
from models.models import DecoderInput, DecoderOutput
from core.config import settings
from fastapi import HTTPException

logger = logging.getLogger(__name__)

class DecoderService:
    """Client for interacting with the decoder service"""
    
    @staticmethod
    async def decode(input_data: DecoderInput) -> dict:
        """Send decode request to decoder service"""
        try:
            response = requests.post(
                f"{settings.DECODER_SERVICE_URL}/predict",
                json=input_data.dict(),
                timeout=settings.DECODER_TIMEOUT
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Decoder service error: {str(e)}")
            raise HTTPException(status_code=502, detail=f"Decoder service error: {str(e)}")
    
    @staticmethod
    async def check_health() -> bool:
        """Check if decoder service is healthy"""
        try:
            response = requests.get(
                f"{settings.DECODER_SERVICE_URL}/health",
                timeout=5
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Decoder health check failed: {str(e)}")
            return False