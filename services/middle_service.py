import requests
import logging
from models.models import MiddleInput, MiddleOutput
from core.config import settings
from fastapi import HTTPException

logger = logging.getLogger(__name__)

class MiddleService:
    """Client for interacting with the middle service"""
    
    @staticmethod
    async def predict(input_data: MiddleInput) -> dict:
        """Send predict request to middle service"""
        try:
            input_dict = input_data.dict() if hasattr(input_data, "dict") else {"inputs_embeds": input_data.inputs_embeds}
            
            response = requests.post(
                f"{settings.MIDDLE_SERVICE_URL}/predict",
                json=input_dict,
                timeout=settings.MIDDLE_TIMEOUT
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Middle service error: {str(e)}")
            raise HTTPException(status_code=502, detail=f"Middle service error: {str(e)}")

    @staticmethod
    async def clear() -> None:
        """Clear middle service state"""
        try:
            response = requests.post(
                f"{settings.MIDDLE_SERVICE_URL}/clear",
                timeout=5
            )
            response.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"Error clearing middle service state: {str(e)}")
            # Don't raise exception here, as this is a non-critical operation
    
    @staticmethod
    async def check_health() -> bool:
        """Check if middle service is healthy"""
        try:
            response = requests.get(
                f"{settings.MIDDLE_SERVICE_URL}/health",
                timeout=5
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Middle health check failed: {str(e)}")
            return False