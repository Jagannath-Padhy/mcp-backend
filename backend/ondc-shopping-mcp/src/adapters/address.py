"""Address management operations for MCP adapters"""

from typing import Dict, Any, Optional, List
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


async def get_delivery_addresses(
    user_id: str,
    device_id: Optional[str] = None,
    session_id: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Get user's delivery addresses
    
    Args:
        user_id: User ID (Firebase user ID or "guestUser")
        device_id: Device ID for guest users
        session_id: MCP session ID (optional)
        
    Returns:
        List of delivery addresses
    """
    try:
        # Get enhanced session with conversation tracking
        session_obj, conversation_manager = get_persistent_session(session_id, tool_name="get_delivery_addresses", **kwargs)
        
        logger.info(f"[Address] Get delivery addresses - User: {user_id}, Device: {device_id}")
        
        # Smart address fetching - works for any user with valid userId
        if user_id == "guestUser" or not user_id:
            return format_mcp_response(
                False,
                "📍 User ID required to fetch delivery addresses.",
                session_obj.session_id
            )
        
        # Get addresses from backend using new userId-based endpoint
        from ..buyer_backend_client import BuyerBackendClient
        buyer_app = BuyerBackendClient()
        
        result = await buyer_app.get_delivery_addresses_by_user(user_id)
        
        if result and result.get('success', False):
            addresses = result.get('data', [])
            address_count = len(addresses)
            
            if address_count == 0:
                message = "📍 No delivery addresses found. Add an address to continue with orders."
            else:
                message = f"📍 Found {address_count} delivery address{'es' if address_count != 1 else ''}:"
                for i, addr in enumerate(addresses[:3], 1):  # Show first 3
                    name = addr.get('name', 'Address')
                    area = addr.get('locality', addr.get('area', 'Unknown area'))
                    city = addr.get('city', '')
                    message += f"\n{i}. {name} - {area}, {city}"
                
                if address_count > 3:
                    message += f"\n... and {address_count - 3} more addresses"
            
            return format_mcp_response(
                True,
                message,
                session_obj.session_id,
                addresses=addresses,
                address_count=address_count
            )
        else:
            # Gracefully handle no addresses - return empty result instead of error
            logger.info(f"[Address] No addresses found for user {user_id}")
            return format_mcp_response(
                True,
                "📍 No delivery addresses found for this user.",
                session_obj.session_id,
                addresses=[],
                address_count=0
            )
            
    except Exception as e:
        logger.error(f"[Address] Get addresses error: {e}")
        return format_mcp_response(
            False,
            f"❌ Address fetch error: {str(e)}",
            session_id or "unknown"
        )


async def add_delivery_address(
    address_data: Dict[str, Any],
    user_id: str,
    device_id: Optional[str] = None,
    session_id: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Add new delivery address
    
    Args:
        address_data: Address information (name, building, locality, city, state, pincode, etc.)
        user_id: User ID (Firebase user ID or "guestUser")
        device_id: Device ID for guest users
        session_id: MCP session ID (optional)
        
    Returns:
        Success/failure with address details
    """
    try:
        # Get enhanced session with conversation tracking
        session_obj, conversation_manager = get_persistent_session(session_id, tool_name="add_delivery_address", **kwargs)
        
        logger.info(f"[Address] Add delivery address - User: {user_id}, Device: {device_id}")
        
        # Check authentication for address operations
        if user_id == "guestUser":
            return format_mcp_response(
                False,
                "🔐 Adding addresses requires login. Please authenticate first.",
                session_obj.session_id
            )
        
        if not session_obj.user_authenticated or not session_obj.auth_token:
            return format_mcp_response(
                False,
                "🔐 Please login to add delivery addresses.",
                session_obj.session_id
            )
        
        # Validate required address fields
        required_fields = ['name', 'building', 'locality', 'city', 'state', 'areaCode']
        missing_fields = [field for field in required_fields if not address_data.get(field)]
        
        if missing_fields:
            return format_mcp_response(
                False,
                f"📝 Missing required address fields: {', '.join(missing_fields)}",
                session_obj.session_id
            )
        
        # Add address via backend
        from ..buyer_backend_client import BuyerBackendClient
        buyer_app = BuyerBackendClient()
        
        result = await buyer_app.add_delivery_address(address_data, session_obj.auth_token)
        
        if result and not result.get('error'):
            address_name = address_data.get('name', 'New address')
            city = address_data.get('city', '')
            
            return format_mcp_response(
                True,
                f"✅ Successfully added delivery address: {address_name}, {city}",
                session_obj.session_id,
                address_id=result.get('address_id'),
                address=result.get('address')
            )
        else:
            error_msg = result.get('message', 'Failed to add address') if result else 'Backend error'
            logger.error(f"[Address] Add address failed: {error_msg}")
            return format_mcp_response(
                False,
                f"❌ Failed to add address: {error_msg}",
                session_obj.session_id
            )
            
    except Exception as e:
        logger.error(f"[Address] Add address error: {e}")
        return format_mcp_response(
            False,
            f"❌ Address add error: {str(e)}",
            session_id or "unknown"
        )


async def update_delivery_address(
    address_id: str,
    address_data: Dict[str, Any],
    user_id: str,
    device_id: Optional[str] = None,
    session_id: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Update existing delivery address
    
    Args:
        address_id: ID of address to update
        address_data: Updated address information
        user_id: User ID (Firebase user ID or "guestUser")
        device_id: Device ID for guest users
        session_id: MCP session ID (optional)
        
    Returns:
        Success/failure with updated address details
    """
    try:
        # Get enhanced session with conversation tracking
        session_obj, conversation_manager = get_persistent_session(session_id, tool_name="update_delivery_address", **kwargs)
        
        logger.info(f"[Address] Update delivery address - User: {user_id}, Address ID: {address_id}")
        
        # Check authentication
        if user_id == "guestUser" or not session_obj.user_authenticated or not session_obj.auth_token:
            return format_mcp_response(
                False,
                "🔐 Please login to update delivery addresses.",
                session_obj.session_id
            )
        
        # Update address via backend
        from ..buyer_backend_client import BuyerBackendClient
        buyer_app = BuyerBackendClient()
        
        result = await buyer_app.update_delivery_address(address_id, address_data, session_obj.auth_token)
        
        if result and not result.get('error'):
            return format_mcp_response(
                True,
                f"✅ Successfully updated delivery address",
                session_obj.session_id,
                address=result.get('address')
            )
        else:
            error_msg = result.get('message', 'Failed to update address') if result else 'Backend error'
            logger.error(f"[Address] Update address failed: {error_msg}")
            return format_mcp_response(
                False,
                f"❌ Failed to update address: {error_msg}",
                session_obj.session_id
            )
            
    except Exception as e:
        logger.error(f"[Address] Update address error: {e}")
        return format_mcp_response(
            False,
            f"❌ Address update error: {str(e)}",
            session_id or "unknown"
        )


async def delete_delivery_address(
    address_id: str,
    user_id: str,
    device_id: Optional[str] = None,
    session_id: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Delete delivery address
    
    Args:
        address_id: ID of address to delete
        user_id: User ID (Firebase user ID or "guestUser")
        device_id: Device ID for guest users
        session_id: MCP session ID (optional)
        
    Returns:
        Success/failure message
    """
    try:
        # Get enhanced session with conversation tracking
        session_obj, conversation_manager = get_persistent_session(session_id, tool_name="delete_delivery_address", **kwargs)
        
        logger.info(f"[Address] Delete delivery address - User: {user_id}, Address ID: {address_id}")
        
        # Check authentication
        if user_id == "guestUser" or not session_obj.user_authenticated or not session_obj.auth_token:
            return format_mcp_response(
                False,
                "🔐 Please login to delete delivery addresses.",
                session_obj.session_id
            )
        
        # Delete address via backend
        from ..buyer_backend_client import BuyerBackendClient
        buyer_app = BuyerBackendClient()
        
        result = await buyer_app.delete_delivery_address(address_id, session_obj.auth_token)
        
        if result and not result.get('error'):
            return format_mcp_response(
                True,
                f"✅ Successfully deleted delivery address",
                session_obj.session_id
            )
        else:
            error_msg = result.get('message', 'Failed to delete address') if result else 'Backend error'
            logger.error(f"[Address] Delete address failed: {error_msg}")
            return format_mcp_response(
                False,
                f"❌ Failed to delete address: {error_msg}",
                session_obj.session_id
            )
            
    except Exception as e:
        logger.error(f"[Address] Delete address error: {e}")
        return format_mcp_response(
            False,
            f"❌ Address delete error: {str(e)}",
            session_id or "unknown"
        )