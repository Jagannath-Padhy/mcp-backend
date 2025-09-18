"""User profile management operations for MCP adapters"""

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
user_service = services['user_service']


async def get_user_profile(
    user_id: str,
    device_id: Optional[str] = None,
    session_id: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Get user profile information
    
    Args:
        user_id: User ID (Firebase user ID or "guestUser")
        device_id: Device ID for guest users
        session_id: MCP session ID (optional)
        
    Returns:
        User profile details
    """
    try:
        # Get enhanced session with conversation tracking
        session_obj, conversation_manager = get_persistent_session(session_id, tool_name="get_user_profile", **kwargs)
        
        logger.info(f"[Profile] Get user profile - User: {user_id}, Device: {device_id}")
        
        # Check authentication for profile operations
        if user_id == "guestUser":
            return format_mcp_response(
                False,
                "🔐 Profile access requires login. Please authenticate first.",
                session_obj.session_id
            )
        
        if not session_obj.user_authenticated or not session_obj.auth_token:
            return format_mcp_response(
                False,
                "🔐 Please login to view profile information.",
                session_obj.session_id
            )
        
        # Get profile from backend
        from ..buyer_backend_client import BuyerBackendClient
        buyer_app = BuyerBackendClient()
        
        result = await buyer_app.get_user_profile(session_obj.auth_token)
        
        if result and not result.get('error'):
            profile = result.get('profile', {})
            
            # Format profile information
            name = profile.get('name', 'Not provided')
            email = profile.get('email', 'Not provided')
            phone = profile.get('phone', 'Not provided')
            created_date = profile.get('created_at', 'Unknown')
            
            message = f"👤 User Profile:\n"
            message += f"Name: {name}\n"
            message += f"Email: {email}\n"
            message += f"Phone: {phone}\n"
            message += f"Member since: {created_date}"
            
            # Add additional profile stats if available
            order_count = profile.get('total_orders', 0)
            if order_count > 0:
                message += f"\nTotal Orders: {order_count}"
            
            return format_mcp_response(
                True,
                message,
                session_obj.session_id,
                profile=profile
            )
        else:
            error_msg = result.get('message', 'Failed to fetch profile') if result else 'Backend error'
            logger.error(f"[Profile] Get profile failed: {error_msg}")
            return format_mcp_response(
                False,
                f"❌ Failed to get profile: {error_msg}",
                session_obj.session_id
            )
            
    except Exception as e:
        logger.error(f"[Profile] Get profile error: {e}")
        return format_mcp_response(
            False,
            f"❌ Profile fetch error: {str(e)}",
            session_id or "unknown"
        )


async def update_user_profile(
    profile_data: Dict[str, Any],
    user_id: str,
    device_id: Optional[str] = None,
    session_id: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Update user profile information
    
    Args:
        profile_data: Profile update data (name, email, etc.)
        user_id: User ID (Firebase user ID or "guestUser")
        device_id: Device ID for guest users
        session_id: MCP session ID (optional)
        
    Returns:
        Success/failure with updated profile details
    """
    try:
        # Get enhanced session with conversation tracking
        session_obj, conversation_manager = get_persistent_session(session_id, tool_name="update_user_profile", **kwargs)
        
        logger.info(f"[Profile] Update user profile - User: {user_id}, Device: {device_id}")
        
        # Check authentication for profile operations
        if user_id == "guestUser":
            return format_mcp_response(
                False,
                "🔐 Profile updates require login. Please authenticate first.",
                session_obj.session_id
            )
        
        if not session_obj.user_authenticated or not session_obj.auth_token:
            return format_mcp_response(
                False,
                "🔐 Please login to update profile information.",
                session_obj.session_id
            )
        
        # Update profile via backend
        from ..buyer_backend_client import BuyerBackendClient
        buyer_app = BuyerBackendClient()
        
        result = await buyer_app.update_user_profile(profile_data, session_obj.auth_token)
        
        if result and not result.get('error'):
            updated_fields = list(profile_data.keys())
            field_list = ", ".join(updated_fields)
            
            return format_mcp_response(
                True,
                f"✅ Successfully updated profile: {field_list}",
                session_obj.session_id,
                updated_fields=updated_fields,
                profile=result.get('profile')
            )
        else:
            error_msg = result.get('message', 'Failed to update profile') if result else 'Backend error'
            logger.error(f"[Profile] Update profile failed: {error_msg}")
            return format_mcp_response(
                False,
                f"❌ Failed to update profile: {error_msg}",
                session_obj.session_id
            )
            
    except Exception as e:
        logger.error(f"[Profile] Update profile error: {e}")
        return format_mcp_response(
            False,
            f"❌ Profile update error: {str(e)}",
            session_id or "unknown"
        )