"""Session service for centralized session management"""

import os
import json
from pathlib import Path
from typing import Dict, Optional, Any
from datetime import datetime, timedelta
import logging

from ..models.session import Session
from ..utils.logger import get_logger

logger = get_logger(__name__)


class SessionService:
    """Service for managing user sessions with proper data models"""
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize session service
        
        Args:
            storage_path: Path to store session files (default: ~/.ondc-mcp/sessions)
        """
        self.storage_path = Path(storage_path or os.path.expanduser("~/.ondc-mcp/sessions"))
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.sessions_cache: Dict[str, Session] = {}
        self.session_timeout = timedelta(hours=24)  # Sessions expire after 24 hours
        logger.info(f"SessionService initialized with storage at: {self.storage_path}")
    
    def get_or_create(self, session_id: Optional[str] = None) -> Session:
        """
        Get existing session or create new one
        
        Args:
            session_id: Optional session ID to retrieve or create with
            
        Returns:
            Session object
        """
        if session_id:
            session = self.get(session_id)
            if session:
                return session
            # Create new session with the provided ID
            return self.create_with_id(session_id)
        
        # Create new session with auto-generated ID
        return self.create()
    
    def get(self, session_id: str) -> Optional[Session]:
        """
        Get session by ID
        
        Args:
            session_id: Session ID to retrieve
            
        Returns:
            Session object or None if not found/expired
        """
        # Check cache first
        if session_id in self.sessions_cache:
            session = self.sessions_cache[session_id]
            if self._is_valid(session):
                session.update_access_time()
                return session
            else:
                # Remove expired session
                del self.sessions_cache[session_id]
        
        # Try loading from disk
        session = self._load_from_disk(session_id)
        if session and self._is_valid(session):
            session.update_access_time()
            self.sessions_cache[session_id] = session
            return session
        
        return None
    
    def create(self, session_id: Optional[str] = None) -> Session:
        """
        Create new session
        
        Args:
            session_id: Optional external session ID to use
            
        Returns:
            New Session object
        """
        if session_id:
            return self.create_with_id(session_id)
        
        session = Session()
        self.sessions_cache[session.session_id] = session
        self._save_to_disk(session)
        logger.info(f"Created new session: {session.session_id}")
        return session
    
    def create_with_id(self, session_id: str) -> Session:
        """
        Create session with provided external ID (for universal compatibility)
        
        Args:
            session_id: External session ID to use
            
        Returns:
            New Session object with provided ID
        """
        # Check if session already exists
        existing_session = self.get(session_id)
        if existing_session:
            logger.info(f"Session with external ID already exists: {session_id}")
            return existing_session
        
        # Create new session with provided ID
        session = Session(session_id=session_id)
        self.sessions_cache[session_id] = session
        self._save_to_disk(session)
        logger.info(f"Created new session with external ID: {session_id}")
        return session
    
    def update(self, session: Session) -> bool:
        """
        Update session
        
        Args:
            session: Session to update
            
        Returns:
            True if successful
        """
        try:
            session.update_access_time()
            self.sessions_cache[session.session_id] = session
            self._save_to_disk(session)
            return True
        except Exception as e:
            logger.error(f"Failed to update session {session.session_id}: {e}")
            return False
    
    def delete(self, session_id: str) -> bool:
        """
        Delete session
        
        Args:
            session_id: Session ID to delete
            
        Returns:
            True if successful
        """
        try:
            # Remove from cache
            if session_id in self.sessions_cache:
                del self.sessions_cache[session_id]
            
            # Remove from disk
            session_file = self.storage_path / f"{session_id}.json"
            if session_file.exists():
                session_file.unlink()
            
            logger.info(f"Deleted session: {session_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete session {session_id}: {e}")
            return False
    
    def cleanup_expired(self) -> int:
        """
        Clean up expired sessions
        
        Returns:
            Number of sessions cleaned up
        """
        cleaned = 0
        current_time = datetime.utcnow()
        
        # Clean cache
        expired_ids = [
            sid for sid, session in self.sessions_cache.items()
            if not self._is_valid(session)
        ]
        for sid in expired_ids:
            del self.sessions_cache[sid]
            cleaned += 1
        
        # Clean disk storage
        for session_file in self.storage_path.glob("*.json"):
            try:
                with open(session_file, 'r') as f:
                    data = json.load(f)
                
                last_accessed = datetime.fromisoformat(data.get('last_accessed', ''))
                if current_time - last_accessed > self.session_timeout:
                    session_file.unlink()
                    cleaned += 1
            except Exception as e:
                logger.warning(f"Failed to check session file {session_file}: {e}")
        
        if cleaned > 0:
            logger.info(f"Cleaned up {cleaned} expired sessions")
        
        return cleaned
    
    def organize_old_sessions(self) -> int:
        """
        Move old sessions (>1 day) to organized directories by date
        
        Returns:
            Number of sessions organized
        """
        organized = 0
        current_time = datetime.utcnow()
        one_day_ago = current_time - timedelta(days=1)
        
        # Create archive directory
        archive_path = self.storage_path / "archive"
        archive_path.mkdir(exist_ok=True)
        
        for session_file in self.storage_path.glob("*.json"):
            try:
                with open(session_file, 'r') as f:
                    data = json.load(f)
                
                last_accessed = datetime.fromisoformat(data.get('last_accessed', ''))
                if last_accessed < one_day_ago:
                    # Create date-based directory
                    date_dir = archive_path / last_accessed.strftime("%Y-%m-%d")
                    date_dir.mkdir(exist_ok=True)
                    
                    # Move session file
                    new_path = date_dir / session_file.name
                    session_file.rename(new_path)
                    organized += 1
                    
            except Exception as e:
                logger.warning(f"Failed to organize session file {session_file}: {e}")
        
        if organized > 0:
            logger.info(f"Organized {organized} old sessions into archive directories")
        
        return organized
    
    def _is_valid(self, session: Session) -> bool:
        """
        Check if session is still valid
        
        Args:
            session: Session to check
            
        Returns:
            True if valid
        """
        age = datetime.utcnow() - session.last_accessed
        return age < self.session_timeout
    
    def _save_to_disk(self, session: Session) -> bool:
        """
        Save session to disk
        
        Args:
            session: Session to save
            
        Returns:
            True if successful
        """
        try:
            session_file = self.storage_path / f"{session.session_id}.json"
            with open(session_file, 'w') as f:
                json.dump(session.to_dict(), f, indent=2, default=str)
            return True
        except Exception as e:
            logger.error(f"Failed to save session {session.session_id} to disk: {e}")
            return False
    
    def _load_from_disk(self, session_id: str) -> Optional[Session]:
        """
        Load session from disk
        
        Args:
            session_id: Session ID to load
            
        Returns:
            Session object or None if not found
        """
        try:
            session_file = self.storage_path / f"{session_id}.json"
            if not session_file.exists():
                return None
            
            with open(session_file, 'r') as f:
                data = json.load(f)
            
            return Session.from_dict(data)
        except Exception as e:
            logger.error(f"Failed to load session {session_id} from disk: {e}")
            return None
    
    def get_session_summary(self, session: Session) -> Dict[str, Any]:
        """
        Get summary of session state
        
        Args:
            session: Session to summarize
            
        Returns:
            Summary dictionary
        """
        return {
            'session_id': session.session_id,
            'cart': {
                'items_count': session.cart.total_items,
                'total_value': session.cart.total_value,
                'items': [item.to_dict() for item in session.cart.items]
            },
            'checkout': {
                'stage': session.checkout_state.stage.value,
                'has_delivery_info': session.checkout_state.delivery_info is not None,
                'transaction_id': session.checkout_state.transaction_id
            },
            'preferences': session.preferences.to_dict(),
            'age': str(datetime.utcnow() - session.created_at)
        }


# Singleton instance
_session_service: Optional[SessionService] = None


def get_session_service() -> SessionService:
    """Get singleton SessionService instance"""
    global _session_service
    if _session_service is None:
        _session_service = SessionService()
    return _session_service