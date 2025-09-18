"""Session management operations for MCP adapters"""

from typing import Dict, Any, Optional
from .utils import (
    get_persistent_session, 
    save_persistent_session, 
    extract_session_id, 
    format_mcp_response,
    get_services
)
from ..utils.logger import get_logger

logger = get_logger(__name__)

# Get services
services = get_services()
cart_service = services['cart_service']
session_service = services['session_service']

# Global session cache for preventing session reuse between chats
_LAST_SESSION = None


async def initialize_shopping(session_id: Optional[str] = None, **kwargs) -> Dict[str, Any]:
    """MCP adapter for initialize_shopping - Guest-only mode
    
    Creates a new guest session and prompts for deviceId to enable
    complete order placement journey without authentication.
    """
    try:
        # Create or get session with enhanced conversation tracking
        session_obj, conversation_manager = get_persistent_session(session_id, tool_name="initialize_shopping", **kwargs)
        
        # Clear any cached last session
        global _LAST_SESSION
        _LAST_SESSION = None
        
        # Set guest mode explicitly
        session_obj.user_authenticated = False
        session_obj.demo_mode = False
        session_obj.user_id = "guestUser"
        
        # Check if deviceId is provided
        device_id = kwargs.get('deviceId') or kwargs.get('device_id')
        
        if not device_id:
            logger.info(f"Created guest session {session_obj.session_id[:8]}... - requesting deviceId")
            
            welcome_message = (
                "🛒 **Welcome to ONDC Shopping - Guest Mode!**\n\n"
                "Shop as a guest with full order placement capabilities!\n\n"
                "**To continue, I need your device ID:**\n"
                "• This helps track your cart and orders\n"
                "• Use any unique identifier (e.g., 'mobile123', 'laptop456')\n\n"
                "**Call this tool again with deviceId:**\n"
                "`initialize_shopping deviceId='your_device_id'`\n\n"
                "**Example:** `initialize_shopping deviceId='mobile123'`\n\n"
                "🎯 **What you can do:**\n"
                "• Search products\n• Add to cart\n• Place orders\n• Track deliveries\n\n"
                "All without signing up! "
            )
            
            # Save session
            save_persistent_session(session_obj, conversation_manager)
            
            return format_mcp_response(
                False,
                welcome_message,
                session_obj.session_id,
                device_id_required=True,
                next_action="provide_device_id"
            )
        else:
            # DeviceId provided - complete initialization
            session_obj.device_id = device_id
            logger.info(f"Initialized guest session {session_obj.session_id[:8]}... with deviceId: {device_id}")
            
            success_message = (
                f"✅ **Guest Session Ready!**\n\n"
                f"**Session ID:** `{session_obj.session_id}`\n"
                f"**Device ID:** `{device_id}`\n"
                f"**Mode:** Guest (No login required)\n\n"
                f"🛍️ **Start Shopping:**\n"
                f"• `search_products query='organic rice'`\n"
                f"• `browse_categories`\n"
                f"• `view_cart`\n\n"
                f"🚀 **Full Order Journey Available:**\n"
                f"Search → Cart → Checkout → Payment → Delivery\n\n"
                f"Ready to shop! What would you like to find?"
            )
            
            # Save session with deviceId
            save_persistent_session(session_obj, conversation_manager)
            
            return format_mcp_response(
                True,
                success_message,
                session_obj.session_id,
                guest_mode=True,
                device_id=device_id,
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