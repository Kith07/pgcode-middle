import logging
from services.middle_service import MiddleService

logger = logging.getLogger(__name__)

sessions = {}

class SessionManager:
    """Handles session management for the XCode pipeline"""
    
    @staticmethod
    async def get_session(session_id: str) -> bool:
        """Check if a session exists"""
        return session_id in sessions
    
    @staticmethod
    async def create_session(session_id: str) -> None:
        """Create a new session"""
        if session_id not in sessions:
            await MiddleService.clear()
            sessions[session_id] = True
            logger.info(f"Created new session: {session_id}")
    
    @staticmethod
    async def delete_session(session_id: str) -> bool:
        """Delete a session"""
        if session_id in sessions:
            await MiddleService.clear()
            del sessions[session_id]
            logger.info(f"Deleted session: {session_id}")
            return True
        return False