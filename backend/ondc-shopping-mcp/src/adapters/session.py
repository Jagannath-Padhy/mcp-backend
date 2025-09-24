"""Session management operations for MCP adapters"""

import sys
import os
import uuid
import time
from typing import Dict, Any, Optional

# Ensure Python path is set for tool execution context
current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
from src.adapters.utils import (
    get_persistent_session, 
    save_persistent_session, 
    extract_session_id, 
    format_mcp_response,
    get_services
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Get services
services = get_services()
cart_service = services['cart_service']
session_service = services['session_service']

# Global session cache for preventing session reuse between chats
_LAST_SESSION = None


async def initialize_shopping(userId: str,
                             deviceId: str,
                             session_id: Optional[str] = None, 
                             **kwargs) -> Dict[str, Any]:
    """MCP adapter for initialize_shopping - User credential mode
    
    Collects userId and deviceId from users to ensure backend cart isolation.
    Users should provide their Himira frontend credentials.
    """
    try:
        # Create or get session with enhanced conversation tracking
        session_obj, conversation_manager = get_persistent_session(session_id, tool_name="initialize_shopping", **kwargs)
        
        # Clear any cached last session
        global _LAST_SESSION
        _LAST_SESSION = None
        
        # MANDATORY CREDENTIALS - Validate non-empty values
        if not userId.strip() or not deviceId.strip():
            return format_mcp_response(
                False,
                "🚨 **Invalid Himira Credentials**\n\n"
                "Both userId and deviceId must be non-empty values.\n\n"
                "**Call:** `initialize_shopping(userId='your_user_id', deviceId='your_device_id')`\n\n"
                "**Where to find these:**\n"
                "• Log into your Himira frontend\n"
                "• Open browser Developer Tools (F12)\n"
                "• Check localStorage for your userId and deviceId\n\n"
                "**Provided values:**\n"
                f"• userId: '{userId}' ({'valid' if userId.strip() else 'empty/invalid'})\n"
                f"• deviceId: '{deviceId}' ({'valid' if deviceId.strip() else 'empty/invalid'})",
                session_obj.session_id
            )
        
        logger.info(f"Using provided Himira credentials: userId={userId}, deviceId={deviceId}")
        credentials_status = "✅ **Using Your Himira Credentials**"
        
        # Set user credentials
        session_obj.user_authenticated = True  # Since user provided credentials
        session_obj.demo_mode = False
        session_obj.user_id = userId
        session_obj.device_id = deviceId
        
        logger.info(f"Initialized session {session_obj.session_id[:8]}... with userId: {userId}, deviceId: {deviceId}")
        logger.debug(f"Session object state - user_id: {session_obj.user_id}, device_id: {session_obj.device_id}")
        
        # No need for login - user provided their frontend credentials
        # Backend API calls will use wil-api-key header for authentication
        logger.info(f"Session ready with user credentials - no login needed (using wil-api-key for backend auth)")
        
        # SUCCESS - Real credentials provided
        success_message = (
            f"✅ **Session Ready with Your Himira Credentials!**\n\n"
            f"**Session ID:** `{session_obj.session_id}`\n"
            f"**Your User ID:** `{userId}`\n"
            f"**Your Device ID:** `{deviceId}`\n\n"
            f"{credentials_status}\n\n"
            f"🛍️ **Your cart and order history are now accessible!**\n\n"
            f"**Start Shopping:**\n"
            f"• `search_products query='organic rice'`\n"
            f"• `view_cart` - See your existing cart\n"
            f"• `browse_categories`\n\n"
            f"🚀 **Full Order Journey Available:**\n"
            f"Search → Cart → Checkout → Payment → Delivery\n\n"
            f"Ready to shop! What would you like to find?"
        )
        
        # Save session with configured IDs
        logger.info(f"Saving session {session_obj.session_id[:8]}... with userId={session_obj.user_id}, deviceId={session_obj.device_id[:8]}...")
        save_persistent_session(session_obj, conversation_manager)
        logger.info(f"Session {session_obj.session_id[:8]}... saved successfully")
        
        return format_mcp_response(
            True,
            success_message,
            session_obj.session_id,
            authenticated_mode=True,  # Real user credentials
            user_id=userId,
            device_id=deviceId,
            next_action="start_shopping"
        )
        
    except Exception as e:
        logger.error(f"Failed to initialize shopping: {e}")
        return format_mcp_response(
            False,
            f'❌ Failed to initialize session: {str(e)}',
            'unknown'
        )


async def get_session_info(session_id: Optional[str] = None, **kwargs) -> Dict[str, Any]:
    """MCP adapter for get_session_info"""
    try:
        # Get enhanced session with conversation tracking
        session_obj, conversation_manager = get_persistent_session(session_id, tool_name="get_session_info", **kwargs)
        
        # Get cart summary
        cart_summary = cart_service.get_cart_summary(session_obj)
        
        # Build comprehensive session info
        info = {
            'session_id': session_obj.session_id,
            'user_id': session_obj.user_id,
            'device_id': session_obj.device_id,
            'user_authenticated': session_obj.user_authenticated,
            'demo_mode': session_obj.demo_mode,
            'cart': session_obj.cart.to_dict(),  # Cart class already has to_dict method
            'checkout_state': {
                'stage': session_obj.checkout_state.stage.value if session_obj.checkout_state else 'none',
                'transaction_id': session_obj.checkout_state.transaction_id if session_obj.checkout_state else None,
                'order_id': session_obj.checkout_state.order_id if session_obj.checkout_state else None
            },
            'history': session_obj.history[-5:] if session_obj.history else []  # History is already a list of dicts
        }
        
        message = f" Session {session_obj.session_id[:8]}... | Cart: {cart_summary['total_items']} items (₹{cart_summary['total_value']:.2f})"
        
        # Save session with enhanced persistence
        save_persistent_session(session_obj, conversation_manager)
        
        return format_mcp_response(
            True,
            message,
            session_obj.session_id,
            session_data=info  # Use session_data key for consistency
        )
        
    except Exception as e:
        logger.error(f"Failed to get session info: {e}")
        return format_mcp_response(
            False,
            f' Failed to get session info: {str(e)}',
            session_id or 'unknown'
        )