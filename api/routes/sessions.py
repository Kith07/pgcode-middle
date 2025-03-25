from fastapi import APIRouter
from models.models import SessionResponse
from core.session import SessionManager
import logging

router = APIRouter(tags=["sessions"])
logger = logging.getLogger(__name__)

@router.post("/clear-session/{session_id}", response_model=SessionResponse)
async def clear_session(session_id: str):
    """Clear a specific session"""
    success = await SessionManager.delete_session(session_id)
    
    if success:
        return SessionResponse(
            status="success", 
            message=f"Session {session_id} cleared",
            session_id=session_id
        )
    else:
        return SessionResponse(
            status="not_found", 
            message=f"Session {session_id} not found",
            session_id=session_id
        )

@router.get("/session-status/{session_id}", response_model=SessionResponse)
async def session_status(session_id: str):
    """Get status of a session"""
    exists = await SessionManager.get_session(session_id)
    
    if exists:
        return SessionResponse(
            status="active", 
            message="Session is active",
            session_id=session_id
        )
    return SessionResponse(
        status="not_found", 
        message="Session not found",
        session_id=session_id
    )