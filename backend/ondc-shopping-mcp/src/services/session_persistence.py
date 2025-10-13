"""Session persistence service for MCP server

This module provides disk-based session persistence to survive MCP subprocess restarts.
Since Langflow's STDIO MCP client creates new subprocess for each tool call,
we need to persist session state to disk between calls.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging

from ..models.session import Session
from ..utils.logger import get_logger

logger = get_logger(__name__)


class SessionPersistence:
    """Handles disk-based session persistence for MCP server"""
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize session persistence service
        
        Args:
            storage_path: Path to store session files (default: ~/.ondc-mcp/mcp_sessions)
        """
        self.storage_path = Path(storage_path or os.path.expanduser("~/.ondc-mcp/mcp_sessions"))
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.session_timeout = timedelta(hours=24)  # Sessions expire after 24 hours
        logger.info(f"SessionPersistence initialized with storage at: {self.storage_path}")
    
    def get_session(self, session_id: Optional[str] = None) -> Session:
        """
        Get or create session with persistence
        
        Args:
            session_id: Optional session ID to retrieve or create with
            
        Returns:
            Session object (loaded from disk or newly created)
        """
        if session_id:
            # Try to load existing session
            session = self._load_from_disk(session_id)
            if session and self._is_valid(session):
                session.update_access_time()
                self._save_to_disk(session)
                logger.debug(f"[MCP-Session] Loaded existing session: {session_id}")
                return session
            
            # Create new session with provided ID
            session = Session(session_id=session_id)
            self._save_to_disk(session)
            logger.info(f"[MCP-Session] Created new session with ID: {session_id}")
            return session
        
        # Create session with auto-generated ID
        session = Session()
        self._save_to_disk(session)
        logger.info(f"[MCP-Session] Created new session: {session.session_id}")
        return session
    
    def save_session(self, session: Session) -> bool:
        """
        Save session to disk
        
        Args:
            session: Session to save
            
        Returns:
            True if successful
        """
        try:
            session.update_access_time()
            return self._save_to_disk(session)
        except Exception as e:
            logger.error(f"Failed to save session {session.session_id}: {e}")
            return False
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete session from disk
        
        Args:
            session_id: Session ID to delete
            
        Returns:
            True if successful
        """
        try:
            session_file = self.storage_path / f"{session_id}.json"
            if session_file.exists():
                session_file.unlink()
                logger.info(f"[MCP-Session] Deleted session: {session_id}")
                return True
            return False
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
        
        for session_file in self.storage_path.glob("*.json"):
            try:
                with open(session_file, 'r') as f:
                    data = json.load(f)
                
                last_accessed = datetime.fromisoformat(data.get('last_accessed', ''))
                if current_time - last_accessed > self.session_timeout:
                    session_file.unlink()
                    cleaned += 1
                    logger.debug(f"[MCP-Session] Cleaned expired session: {session_file.stem}")
            except Exception as e:
                logger.warning(f"Failed to check session file {session_file}: {e}")
        
        if cleaned > 0:
            logger.info(f"[MCP-Session] Cleaned up {cleaned} expired sessions")
        
        return cleaned
    
    def list_sessions(self) -> List[str]:
        """
        List all session IDs
        
        Returns:
            List of session IDs
        """
        session_ids = []
        for session_file in self.storage_path.glob("*.json"):
            session_ids.append(session_file.stem)
        return sorted(session_ids)
    
    def get_session_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get summary of session without loading full object
        
        Args:
            session_id: Session ID to summarize
            
        Returns:
            Summary dictionary or None if not found
        """
        try:
            session_file = self.storage_path / f"{session_id}.json"
            if not session_file.exists():
                return None
            
            with open(session_file, 'r') as f:
                data = json.load(f)
            
            return {
                'session_id': data.get('session_id'),
                'created_at': data.get('created_at'),
                'last_accessed': data.get('last_accessed'),
                'cart_items': len(data.get('cart', {}).get('items', [])),
                'total_value': data.get('cart', {}).get('total_value', 0),
                'checkout_stage': data.get('checkout_state', {}).get('stage'),
                'transaction_id': data.get('checkout_state', {}).get('transaction_id')
            }
        except Exception as e:
            logger.error(f"Failed to get session summary for {session_id}: {e}")
            return None
    
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
            logger.debug(f"[MCP-Session] Saved session to: {session_file}")
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
                logger.debug(f"[MCP-Session] Session file not found: {session_file}")
                return None
            
            with open(session_file, 'r') as f:
                data = json.load(f)
            
            session = Session.from_dict(data)
            logger.debug(f"[MCP-Session] Loaded session from: {session_file}")
            return session
        except Exception as e:
            logger.error(f"Failed to load session {session_id} from disk: {e}")
            return None


# Singleton instance for MCP server
_session_persistence: Optional[SessionPersistence] = None


def get_session_persistence() -> SessionPersistence:
    """Get singleton SessionPersistence instance for MCP server"""
    global _session_persistence
    if _session_persistence is None:
        _session_persistence = SessionPersistence(os.getenv("SESSION_STORE_PATH"))
    return _session_persistence