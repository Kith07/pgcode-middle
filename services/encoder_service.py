import requests
import logging
from models.models import EncoderInput, EncoderOutput
from core.config import settings
from fastapi import HTTPException
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

class EncoderService:
    """Client for interacting with the encoder service"""

    tokenizer = AutoTokenizer.from_pretrained("Qwen/CodeQwen1.5-7B-Chat")
 
    @staticmethod
    async def encode(input_data: EncoderInput) -> dict:
        """Send encode request to encoder service"""
        try:
            tokenized = EncoderService.tokenizer(
                input_data.prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            
            input_ids = tokenized["input_ids"].tolist()
            attention_mask = tokenized["attention_mask"].tolist()
            
            encoder_request = {
                "input_ids": input_ids,
                "attention_mask": attention_mask
            }
            
            response = requests.post(
                f"{settings.ENCODER_SERVICE_URL}/predict",
                json=encoder_request,
                timeout=settings.ENCODER_TIMEOUT
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Encoder service error: {str(e)}")
            raise HTTPException(status_code=502, detail=f"Encoder service error: {str(e)}")

    @staticmethod
    async def check_health() -> bool:
        """Check if encoder service is healthy"""
        try:
            response = requests.get(
                f"{settings.ENCODER_SERVICE_URL}/health",
                timeout=5
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Encoder health check failed: {str(e)}")
            return False