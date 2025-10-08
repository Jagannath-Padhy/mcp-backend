"""Enhanced Conversation Session Manager for Langflow Integration

This module solves the Langflow session propagation issue where session_id
is not automatically passed from Agent to MCP tools. It provides:

1. Conversation-level session tracking across all tool calls
2. Auto-detection of conversation context without explicit session_id
3. Temporal session binding based on tool call patterns
4. Environment-based session sharing for subprocess communication

Key Design:
- Creates "conversation sessions" that span multiple tool calls
- Uses heuristics to detect when a new conversation starts
- Maintains conversation state even when session_id is not provided
- Provides fallback session management for Langflow limitations
"""

import os
import json
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging

from ..models.session import Session
from ..utils.logger import get_logger
from .session_persistence import SessionPersistence

logger = get_logger(__name__)


class ConversationSessionManager:
    """Enhanced session manager that works around Langflow's session_id propagation issues"""
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize conversation session manager
        
        Args:
            storage_path: Base path for session storage
        """
        self.base_storage = Path(os.getenv("SESSION_STORE_PATH"))
        self.conversation_storage = self.base_storage / "conversations" 
        self.session_storage = self.base_storage / "mcp_sessions"
        
        # Create storage directories
        self.conversation_storage.mkdir(parents=True, exist_ok=True)
        self.session_storage.mkdir(parents=True, exist_ok=True)
        
        # Session persistence for individual sessions
        self.session_persistence = SessionPersistence(str(self.session_storage))
        
        # Conversation tracking settings
        self.conversation_timeout = timedelta(minutes=30)  # 30 minutes max between calls
        self.new_conversation_triggers = [
            'initialize_shopping', 'phone_login'
        ]
        
        # Current conversation context
        self.current_conversation_id: Optional[str] = None
        self.current_session_id: Optional[str] = None
        self.last_tool_call_time: Optional[datetime] = None
        
        logger.info(f"ConversationSessionManager initialized")
        logger.info(f"Conversation storage: {self.conversation_storage}")
        logger.info(f"Session storage: {self.session_storage}")
    
    def get_session_for_tool_call(self, 
                                 tool_name: str, 
                                 session_id: Optional[str] = None,
                                 **kwargs) -> Tuple[Session, 'ConversationSessionManager']:
        """
        Get session for a tool call with enhanced conversation tracking
        
        This is the main method that solves the Langflow session propagation issue.
        It uses multiple strategies to maintain conversation continuity:
        
        1. If session_id provided explicitly -> use that session
        2. If no session_id but continuing conversation -> use conversation session
        3. If new conversation detected -> create new conversation session
        4. If isolated tool call -> create temporary session
        
        Args:
            tool_name: Name of the tool being called
            session_id: Optional explicit session ID from Langflow Agent
            **kwargs: Additional context from tool call arguments
            
        Returns:
            Tuple of (Session object, this manager instance)
        """
        current_time = datetime.utcnow()
        
        # Strategy 1: Explicit session_id provided
        if session_id:
            logger.info(f"[ConvSession] Using explicit session_id: {session_id}")
            session = self.session_persistence.get_session(session_id)
            self._update_conversation_context(session.session_id, tool_name, current_time)
            return session, self
        
        # Strategy 2: Detect conversation continuity
        conversation_session_id = self._detect_conversation_session(tool_name, current_time, **kwargs)
        
        if conversation_session_id:
            logger.info(f"[ConvSession] Continuing conversation with session: {conversation_session_id}")
            session = self.session_persistence.get_session(conversation_session_id)
            self._update_conversation_context(session.session_id, tool_name, current_time)
            return session, self
        
        # Strategy 3: Create new conversation session
        logger.info(f"[ConvSession] Creating new conversation session for tool: {tool_name}")
        session = self.session_persistence.get_session()  # Auto-generates ID
        self._start_new_conversation(session.session_id, tool_name, current_time)
        return session, self
    
    def _detect_conversation_session(self, 
                                   tool_name: str, 
                                   current_time: datetime,
                                   **kwargs) -> Optional[str]:
        """
        Detect if this tool call is part of an ongoing conversation
        
        Uses multiple heuristics:
        1. Time-based: Recent tool calls within conversation timeout
        2. Pattern-based: Tool call sequence suggests continuation
        3. Context-based: Arguments suggest same user/session
        
        Args:
            tool_name: Name of the tool being called
            current_time: Current timestamp
            **kwargs: Tool call arguments for context analysis
            
        Returns:
            Session ID to continue or None for new conversation
        """
        # Check if we have a recent conversation
        if (self.current_session_id and 
            self.last_tool_call_time and 
            current_time - self.last_tool_call_time < self.conversation_timeout):
            
            # Don't continue if this is a conversation restart trigger
            if tool_name in self.new_conversation_triggers:
                # But allow continuation for search_products if cart exists
                if tool_name == 'search_products':
                    session = self.session_persistence.get_session(self.current_session_id)
                    if session and session.cart and len(session.cart.items) > 0:
                        logger.debug(f"[ConvSession] Continuing search in session with {len(session.cart.items)} cart items")
                        return self.current_session_id
                
                logger.info(f"[ConvSession] Tool '{tool_name}' triggers new conversation")
                return None
            
            # Continue with current session
            logger.debug(f"[ConvSession] Continuing conversation - tool: {tool_name}, last_call: {self.last_tool_call_time}")
            return self.current_session_id
        
        # Check for environment-based session (for subprocess communication)
        env_session_id = os.environ.get('ONDC_SESSION_ID')
        if env_session_id:
            logger.info(f"[ConvSession] Found environment session: {env_session_id}")
            return env_session_id
        
        # Look for recent conversation files
        recent_conversation = self._find_recent_conversation(current_time)
        if recent_conversation:
            logger.info(f"[ConvSession] Found recent conversation: {recent_conversation}")
            return recent_conversation
        
        return None
    
    def _start_new_conversation(self, session_id: str, tool_name: str, current_time: datetime):
        """
        Start a new conversation context
        
        Args:
            session_id: Session ID for the new conversation
            tool_name: Tool that initiated the conversation
            current_time: Conversation start time
        """
        conversation_id = f"conv_{int(current_time.timestamp())}_{session_id[:8]}"
        
        # Update internal state
        self.current_conversation_id = conversation_id
        self.current_session_id = session_id
        self.last_tool_call_time = current_time
        
        # Save conversation metadata
        conversation_data = {
            'conversation_id': conversation_id,
            'session_id': session_id,
            'started_at': current_time.isoformat(),
            'started_by_tool': tool_name,
            'last_activity': current_time.isoformat(),
            'tool_calls': [
                {
                    'tool': tool_name,
                    'timestamp': current_time.isoformat()
                }
            ]
        }
        
        conversation_file = self.conversation_storage / f"{conversation_id}.json"
        with open(conversation_file, 'w') as f:
            json.dump(conversation_data, f, indent=2)
        
        # Set environment variable for subprocess communication
        os.environ['ONDC_SESSION_ID'] = session_id
        os.environ['ONDC_CONVERSATION_ID'] = conversation_id
        
        logger.info(f"[ConvSession] Started conversation {conversation_id} with session {session_id}")
    
    def _update_conversation_context(self, session_id: str, tool_name: str, current_time: datetime):
        """
        Update existing conversation context
        
        Args:
            session_id: Current session ID
            tool_name: Tool being called
            current_time: Call timestamp
        """
        # Update internal state
        self.current_session_id = session_id
        self.last_tool_call_time = current_time
        
        # Update environment variables
        os.environ['ONDC_SESSION_ID'] = session_id
        if self.current_conversation_id:
            os.environ['ONDC_CONVERSATION_ID'] = self.current_conversation_id
        
        # Update conversation file if exists
        if self.current_conversation_id:
            conversation_file = self.conversation_storage / f"{self.current_conversation_id}.json"
            if conversation_file.exists():
                try:
                    with open(conversation_file, 'r') as f:
                        conversation_data = json.load(f)
                    
                    conversation_data['last_activity'] = current_time.isoformat()
                    conversation_data['tool_calls'].append({
                        'tool': tool_name,
                        'timestamp': current_time.isoformat()
                    })
                    
                    with open(conversation_file, 'w') as f:
                        json.dump(conversation_data, f, indent=2)
                        
                except Exception as e:
                    logger.warning(f"Failed to update conversation file: {e}")
        
        logger.debug(f"[ConvSession] Updated conversation context - session: {session_id}, tool: {tool_name}")
    
    def _find_recent_conversation(self, current_time: datetime) -> Optional[str]:
        """
        Find the most recent conversation within timeout window
        
        Args:
            current_time: Current timestamp
            
        Returns:
            Session ID of recent conversation or None
        """
        most_recent = None
        most_recent_time = None
        
        for conversation_file in self.conversation_storage.glob("conv_*.json"):
            try:
                with open(conversation_file, 'r') as f:
                    data = json.load(f)
                
                last_activity = datetime.fromisoformat(data.get('last_activity', ''))
                
                if current_time - last_activity < self.conversation_timeout:
                    if most_recent_time is None or last_activity > most_recent_time:
                        most_recent = data.get('session_id')
                        most_recent_time = last_activity
                        
            except Exception as e:
                logger.warning(f"Failed to check conversation file {conversation_file}: {e}")
        
        return most_recent
    
    def cleanup_conversations(self) -> int:
        """
        Clean up old conversation files
        
        Returns:
            Number of conversations cleaned
        """
        cleaned = 0
        current_time = datetime.utcnow()
        max_age = timedelta(days=7)  # Keep conversations for 7 days
        
        for conversation_file in self.conversation_storage.glob("conv_*.json"):
            try:
                with open(conversation_file, 'r') as f:
                    data = json.load(f)
                
                started_at = datetime.fromisoformat(data.get('started_at', ''))
                if current_time - started_at > max_age:
                    conversation_file.unlink()
                    cleaned += 1
                    
            except Exception as e:
                logger.warning(f"Failed to cleanup conversation file {conversation_file}: {e}")
        
        if cleaned > 0:
            logger.info(f"[ConvSession] Cleaned up {cleaned} old conversations")
        
        return cleaned
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """
        Get summary of current conversation state
        
        Returns:
            Dictionary with conversation information
        """
        return {
            'conversation_id': self.current_conversation_id,
            'session_id': self.current_session_id,
            'last_tool_call_time': self.last_tool_call_time.isoformat() if self.last_tool_call_time else None,
            'environment_session': os.environ.get('ONDC_SESSION_ID'),
            'active_conversations': len(list(self.conversation_storage.glob("conv_*.json")))
        }


# Global instance for MCP server use
_conversation_manager: Optional[ConversationSessionManager] = None


def get_conversation_session_manager() -> ConversationSessionManager:
    """Get singleton ConversationSessionManager instance"""
    global _conversation_manager
    if _conversation_manager is None:
        _conversation_manager = ConversationSessionManager()
    return _conversation_manager


def get_enhanced_session(tool_name: str, session_id: Optional[str] = None, **kwargs) -> Tuple[Session, ConversationSessionManager]:
    """
    Convenience function to get session with enhanced conversation tracking
    
    This is the main function to use in MCP adapters for session management.
    
    Args:
        tool_name: Name of the MCP tool being called
        session_id: Optional session ID from Langflow Agent
        **kwargs: Additional context from tool arguments
        
    Returns:
        Tuple of (Session object, ConversationSessionManager instance)
    """
    manager = get_conversation_session_manager()
    return manager.get_session_for_tool_call(tool_name, session_id, **kwargs)