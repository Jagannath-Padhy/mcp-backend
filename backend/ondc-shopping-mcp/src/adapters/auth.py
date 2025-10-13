"""Authentication operations for MCP adapters"""

from typing import Dict, Any, Optional
from .utils import (
    get_persistent_session, 
    save_persistent_session, 
    extract_session_id, 
    format_mcp_response
)
from ..utils.logger import get_logger

logger = get_logger(__name__)


async def phone_login(phone: str, session_id: Optional[str] = None, **kwargs) -> Dict[str, Any]:
    """
    Quick phone login without OTP - Direct authentication using loginWithPhone endpoint
    
    Args:
        phone: Phone number (10 digits, will be formatted with +91)
        session_id: MCP session ID (optional)
        
    Returns:
        Authentication result with token and user info
    """
    try:
        logger.info(f"[Auth] Phone login initiated for phone: {phone}")
        
        # Get enhanced session with conversation tracking
        session_obj, conversation_manager = get_persistent_session(session_id, tool_name="phone_login", **kwargs)
        
        # Validate phone number
        if not phone:
            return format_mcp_response(
                False,
                " Phone number is required for login",
                session_obj.session_id
            )
        
        # Normalize phone number (add +91 if not present)
        normalized_phone = phone.strip()
        if not normalized_phone.startswith('+'):
            if normalized_phone.startswith('91') and len(normalized_phone) == 12:
                normalized_phone = f"+{normalized_phone}"
            elif len(normalized_phone) == 10:
                normalized_phone = f"+91{normalized_phone}"
        
        logger.info(f"[Auth] Phone login attempt: {normalized_phone}")
        
        # Call backend loginWithPhone endpoint
        from ..buyer_backend_client import BuyerBackendClient
        buyer_app = BuyerBackendClient()
        
        result = await buyer_app.login_with_phone({"phone": normalized_phone})
        
        if result and result.get("success"):
            # Store authentication data in session
            session_obj.auth_token = result.get("token")
            session_obj.user_authenticated = True
            session_obj.user_profile = result.get("user", {})
            session_obj.demo_mode = False
            session_obj.user_id = result.get("user", {}).get("userId")
            
            # Save enhanced session with conversation tracking
            save_persistent_session(session_obj, conversation_manager)
            
            # Get user name for greeting
            user_name = session_obj.user_profile.get("userName", "User")
            if user_name and " " in user_name:
                user_name = user_name.split()[0]  # Use first name only
            
            # Success message
            success_message = (
                f" **Welcome, {user_name}!**\n\n"
                f" Successfully logged in with {normalized_phone}\n\n"
                f" **Ready to shop on ONDC!**\n\n"
                f"What would you like to do?\n"
                f"•  **Search products**: Tell me what you need\n"
                f"•  **Browse categories**: See what's available\n"
                f"•  **View cart**: Check your current cart\n\n"
                f" Try: 'Search for organic vegetables' or 'Show me categories'"
            )
            
            logger.info(f"[Auth] Phone login successful for: {normalized_phone}")
            
            return format_mcp_response(
                True,
                success_message,
                session_obj.session_id,
                authenticated=True,
                user_profile=session_obj.user_profile,
                token=session_obj.auth_token,
                suggestions=["search_products", "browse_categories", "view_cart"]
            )
            
        else:
            error_message = result.get("message", "Login failed") if result else "Connection failed"
            logger.error(f"[Auth] Phone login failed for {normalized_phone}: {error_message}")
            
            return format_mcp_response(
                False,
                f" Login failed: {error_message}\n\n Please check your phone number and try again.",
                session_obj.session_id
            )
            
    except Exception as e:
        logger.error(f"[Auth] Phone login error: {e}")
        return format_mcp_response(
            False,
            f" Authentication error: {str(e)}\n\n Please try again in a moment.",
            session_id or "unknown"
        )