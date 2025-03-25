from fastapi import APIRouter
from fastapi.responses import JSONResponse
from models.models import HealthResponse
from services.encoder_service import EncoderService
from services.middle_service import MiddleService
from services.decoder_service import DecoderService
import logging

router = APIRouter(tags=["health"])
logger = logging.getLogger(__name__)

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint that verifies all component services"""
    
    encoder_healthy = await EncoderService.check_health()
    middle_healthy = await MiddleService.check_health()
    decoder_healthy = await DecoderService.check_health()
    
    status = {
        "status": "healthy",
        "services": {
            "encoder": "healthy" if encoder_healthy else "unhealthy",
            "middle": "healthy" if middle_healthy else "unhealthy",
            "decoder": "healthy" if decoder_healthy else "unhealthy"
        }
    }
    
    all_healthy = encoder_healthy and middle_healthy and decoder_healthy
    if not all_healthy:
        status["status"] = "degraded"
        return JSONResponse(status_code=207, content=status)
    
    return status