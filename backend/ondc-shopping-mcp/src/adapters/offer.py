"""Offer management operations for MCP adapters"""

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


async def get_active_offers(
    user_id: str,
    device_id: Optional[str] = None,
    session_id: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Get active offers available to user
    
    Args:
        user_id: User ID (Firebase user ID or "guestUser")
        device_id: Device ID for guest users
        session_id: MCP session ID (optional)
        
    Returns:
        List of active offers
    """
    try:
        # Get enhanced session with conversation tracking
        session_obj, conversation_manager = get_persistent_session(session_id, tool_name="get_active_offers", **kwargs)
        
        logger.info(f"[Offer] Get active offers - User: {user_id}, Device: {device_id}")
        
        # Get offers from backend
        from ..buyer_backend_client import BuyerBackendClient
        buyer_app = BuyerBackendClient()
        
        result = await buyer_app.get_active_offers(user_id, device_id)
        
        if result and not result.get('error'):
            offers = result.get('offers', [])
            offer_count = len(offers)
            
            if offer_count == 0:
                message = "🎁 No active offers available at the moment."
            else:
                message = f"🎁 Found {offer_count} active offer{'s' if offer_count != 1 else ''}:"
                for i, offer in enumerate(offers[:3], 1):  # Show first 3
                    title = offer.get('title', 'Untitled Offer')
                    discount = offer.get('discount_amount', offer.get('discount_percent', 'N/A'))
                    message += f"\n{i}. {title} - {discount}"
                
                if offer_count > 3:
                    message += f"\n... and {offer_count - 3} more offers"
            
            return format_mcp_response(
                True,
                message,
                session_obj.session_id,
                offers=offers,
                offer_count=offer_count
            )
        else:
            error_msg = result.get('message', 'Failed to fetch offers') if result else 'Backend error'
            logger.error(f"[Offer] Get offers failed: {error_msg}")
            return format_mcp_response(
                False,
                f"❌ Failed to get offers: {error_msg}",
                session_obj.session_id
            )
            
    except Exception as e:
        logger.error(f"[Offer] Get offers error: {e}")
        return format_mcp_response(
            False,
            f"❌ Offer fetch error: {str(e)}",
            session_id or "unknown"
        )


async def get_applied_offers(
    user_id: str,
    device_id: Optional[str] = None,
    session_id: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Get offers already applied to user's cart/order
    
    Args:
        user_id: User ID (Firebase user ID or "guestUser")
        device_id: Device ID for guest users
        session_id: MCP session ID (optional)
        
    Returns:
        List of applied offers
    """
    try:
        # Get enhanced session with conversation tracking
        session_obj, conversation_manager = get_persistent_session(session_id, tool_name="get_applied_offers", **kwargs)
        
        logger.info(f"[Offer] Get applied offers - User: {user_id}, Device: {device_id}")
        
        # Get applied offers from backend
        from ..buyer_backend_client import BuyerBackendClient
        buyer_app = BuyerBackendClient()
        
        result = await buyer_app.get_applied_offers(user_id, device_id)
        
        if result and not result.get('error'):
            applied_offers = result.get('applied_offers', [])
            total_savings = result.get('total_savings', 0)
            
            if not applied_offers:
                message = "🎁 No offers are currently applied to your cart."
            else:
                offer_count = len(applied_offers)
                message = f"🎁 {offer_count} offer{'s' if offer_count != 1 else ''} applied (₹{total_savings:.2f} savings):"
                for i, offer in enumerate(applied_offers, 1):
                    title = offer.get('title', 'Untitled Offer')
                    savings = offer.get('savings_amount', 0)
                    message += f"\n{i}. {title} - ₹{savings:.2f} off"
            
            return format_mcp_response(
                True,
                message,
                session_obj.session_id,
                applied_offers=applied_offers,
                total_savings=total_savings
            )
        else:
            error_msg = result.get('message', 'Failed to fetch applied offers') if result else 'Backend error'
            logger.error(f"[Offer] Get applied offers failed: {error_msg}")
            return format_mcp_response(
                False,
                f"❌ Failed to get applied offers: {error_msg}",
                session_obj.session_id
            )
            
    except Exception as e:
        logger.error(f"[Offer] Get applied offers error: {e}")
        return format_mcp_response(
            False,
            f"❌ Applied offers fetch error: {str(e)}",
            session_id or "unknown"
        )


async def apply_offer(
    offer_id: str,
    user_id: str,
    device_id: Optional[str] = None,
    session_id: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Apply an offer to user's cart
    
    Args:
        offer_id: ID of the offer to apply
        user_id: User ID (Firebase user ID or "guestUser")
        device_id: Device ID for guest users
        session_id: MCP session ID (optional)
        
    Returns:
        Success/failure with offer application details
    """
    try:
        # Get enhanced session with conversation tracking
        session_obj, conversation_manager = get_persistent_session(session_id, tool_name="apply_offer", **kwargs)
        
        logger.info(f"[Offer] Apply offer - User: {user_id}, Offer ID: {offer_id}")
        
        # Apply offer via backend
        from ..buyer_backend_client import BuyerBackendClient
        buyer_app = BuyerBackendClient()
        
        result = await buyer_app.apply_offer(offer_id, user_id, device_id)
        
        if result and not result.get('error'):
            offer_title = result.get('offer_title', 'Unknown offer')
            savings_amount = result.get('savings_amount', 0)
            
            return format_mcp_response(
                True,
                f"🎉 Successfully applied offer: {offer_title} (₹{savings_amount:.2f} off)",
                session_obj.session_id,
                offer_id=offer_id,
                savings_amount=savings_amount,
                offer_details=result.get('offer_details')
            )
        else:
            error_msg = result.get('message', 'Failed to apply offer') if result else 'Backend error'
            logger.error(f"[Offer] Apply offer failed: {error_msg}")
            return format_mcp_response(
                False,
                f"❌ Failed to apply offer: {error_msg}",
                session_obj.session_id
            )
            
    except Exception as e:
        logger.error(f"[Offer] Apply offer error: {e}")
        return format_mcp_response(
            False,
            f"❌ Offer application error: {str(e)}",
            session_id or "unknown"
        )


async def clear_offers(
    user_id: str,
    device_id: Optional[str] = None,
    session_id: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Clear all applied offers from user's cart
    
    Args:
        user_id: User ID (Firebase user ID or "guestUser")
        device_id: Device ID for guest users
        session_id: MCP session ID (optional)
        
    Returns:
        Success/failure message
    """
    try:
        # Get enhanced session with conversation tracking
        session_obj, conversation_manager = get_persistent_session(session_id, tool_name="clear_offers", **kwargs)
        
        logger.info(f"[Offer] Clear offers - User: {user_id}, Device: {device_id}")
        
        # Clear offers via backend
        from ..buyer_backend_client import BuyerBackendClient
        buyer_app = BuyerBackendClient()
        
        result = await buyer_app.clear_offers(user_id, device_id)
        
        if result and not result.get('error'):
            cleared_count = result.get('cleared_count', 0)
            total_savings_lost = result.get('total_savings_lost', 0)
            
            if cleared_count == 0:
                message = "🎁 No offers were applied to clear."
            else:
                message = f"🧹 Cleared {cleared_count} offer{'s' if cleared_count != 1 else ''} (₹{total_savings_lost:.2f} savings removed)"
            
            return format_mcp_response(
                True,
                message,
                session_obj.session_id,
                cleared_count=cleared_count,
                total_savings_lost=total_savings_lost
            )
        else:
            error_msg = result.get('message', 'Failed to clear offers') if result else 'Backend error'
            logger.error(f"[Offer] Clear offers failed: {error_msg}")
            return format_mcp_response(
                False,
                f"❌ Failed to clear offers: {error_msg}",
                session_obj.session_id
            )
            
    except Exception as e:
        logger.error(f"[Offer] Clear offers error: {e}")
        return format_mcp_response(
            False,
            f"❌ Clear offers error: {str(e)}",
            session_id or "unknown"
        )


async def delete_offer(
    offer_id: str,
    user_id: str,
    device_id: Optional[str] = None,
    session_id: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Remove a specific applied offer from user's cart
    
    Args:
        offer_id: ID of the offer to remove
        user_id: User ID (Firebase user ID or "guestUser")
        device_id: Device ID for guest users
        session_id: MCP session ID (optional)
        
    Returns:
        Success/failure message
    """
    try:
        # Get enhanced session with conversation tracking
        session_obj, conversation_manager = get_persistent_session(session_id, tool_name="delete_offer", **kwargs)
        
        logger.info(f"[Offer] Delete offer - User: {user_id}, Offer ID: {offer_id}")
        
        # Delete offer via backend
        from ..buyer_backend_client import BuyerBackendClient
        buyer_app = BuyerBackendClient()
        
        result = await buyer_app.delete_offer(offer_id, user_id, device_id)
        
        if result and not result.get('error'):
            offer_title = result.get('offer_title', 'Unknown offer')
            savings_lost = result.get('savings_lost', 0)
            
            return format_mcp_response(
                True,
                f"🗑️ Removed offer: {offer_title} (₹{savings_lost:.2f} savings removed)",
                session_obj.session_id,
                offer_id=offer_id,
                savings_lost=savings_lost
            )
        else:
            error_msg = result.get('message', 'Failed to remove offer') if result else 'Backend error'
            logger.error(f"[Offer] Delete offer failed: {error_msg}")
            return format_mcp_response(
                False,
                f"❌ Failed to remove offer: {error_msg}",
                session_obj.session_id
            )
            
    except Exception as e:
        logger.error(f"[Offer] Delete offer error: {e}")
        return format_mcp_response(
            False,
            f"❌ Delete offer error: {str(e)}",
            session_id or "unknown"
        )