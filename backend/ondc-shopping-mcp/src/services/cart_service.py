"""Cart service for managing shopping cart operations"""

from typing import Dict, List, Optional, Any, Tuple
import logging
import json

from ..models.session import Session
from ..buyer_backend_client import BuyerBackendClient, get_buyer_backend_client
from ..utils.logger import get_logger
from ..data_models.ondc_schemas import ONDCDataFactory
from ..utils.field_mapper import enhance_for_backend

logger = get_logger(__name__)


class CartService:
    """Pure backend-driven cart service - no local cart storage"""
    
    def __init__(self, buyer_backend_client: Optional[BuyerBackendClient] = None):
        """
        Initialize cart service
        
        Args:
            buyer_backend_client: Comprehensive client for backend API calls
        """
        self.buyer_app = buyer_backend_client or get_buyer_backend_client()
        logger.info("CartService initialized with comprehensive backend client")
    
    async def add_item(self, session: Session, product: Dict[str, Any], 
                       quantity: int = 1) -> Tuple[bool, str]:
        """
        Add item directly to backend cart
        
        Args:
            session: User session
            product: Product data  
            quantity: Quantity to add
            
        Returns:
            Tuple of (success, message)
        """
        try:
            # Validate quantity
            if quantity is None:
                quantity = 1
            if not isinstance(quantity, int) or quantity < 1 or quantity > 100:
                return False, f" Invalid quantity. Must be between 1 and 100."
            
            # Get user and device IDs
            user_id = session.user_id or "guestUser"  # Guest users supported
            device_id = session.device_id
            
            if not device_id:
                logger.error(f"[Cart] No device_id in session {session.session_id}")
                return False, " Session not properly initialized"
            
            # Prepare cart payload
            cart_payload = self._create_backend_cart_payload(product, quantity)
            
            logger.info(f"[Cart] Adding to backend cart - User: {user_id}, Device: {device_id}, Item: {product.get('name')}")
            
            # Call backend add to cart API
            result = await self.buyer_app.add_to_cart(user_id, device_id, cart_payload)
            
            if result and not result.get('error'):
                product_name = product.get('name', 'Unknown Product')
                logger.info(f"[Cart] Successfully added {product_name} to backend cart")
                return True, f" Added {quantity}x {product_name} to cart"
            else:
                error_msg = result.get('message', 'Failed to add to cart') if result else 'Backend error'
                logger.error(f"[Cart] Backend add to cart failed: {error_msg}")
                return False, f" {error_msg}"
                
        except Exception as e:
            logger.error(f"[Cart] Error adding item to backend cart: {e}")
            return False, f" Failed to add to cart: {str(e)}"
    
    async def view_cart(self, session: Session) -> Tuple[bool, str, List[Dict]]:
        """
        View cart directly from backend
        
        Args:
            session: User session
            
        Returns:
            Tuple of (success, message, cart_items)
        """
        try:
            user_id = session.user_id or "guestUser"
            device_id = session.device_id
            
            if not device_id:
                logger.error(f"[Cart] No device_id in session {session.session_id}")
                return False, " Session not properly initialized", []
            
            logger.info(f"[Cart] Fetching backend cart - User: {user_id}, Device: {device_id}")
            
            # Call backend get cart API
            result = await self.buyer_app.get_cart(user_id, device_id)
            
            # Handle response based on actual Himira backend format
            if result is None:
                return False, " Backend returned no response", []
            
            # Check for explicit error response (dict with error field)
            if isinstance(result, dict) and result.get('error'):
                error_msg = result.get('message', 'Cart operation failed')
                logger.warning(f"[Cart] Backend get cart failed: {error_msg}")
                return False, f" {error_msg}", []
            
            # Handle successful response - backend returns array of cart items directly
            if isinstance(result, list):
                cart_items = result
            elif isinstance(result, dict) and 'data' in result:
                cart_items = result.get('data', [])
            else:
                # Unexpected response format
                logger.warning(f"[Cart] Unexpected response format: {type(result)}")
                return False, " Unexpected response format from backend", []
            
            # Process cart items
            if not cart_items:
                return True, " **Your cart is empty**\n\nStart shopping by searching for products!", []
            
            # Debug: Log raw cart response structure to understand actual data format
            logger.info(f"[Cart] Raw backend response structure (first item): {json.dumps(cart_items[0], indent=2) if cart_items else 'No items'}")
            
            # Format cart display
            display = self._format_backend_cart_display(cart_items)
            logger.info(f"[Cart] Retrieved {len(cart_items)} items from backend cart")
            return True, display, cart_items
                
        except Exception as e:
            logger.error(f"[Cart] Error fetching backend cart: {e}")
            return False, f" Failed to load cart: {str(e)}", []
    
    async def remove_item(self, session: Session, item_id: str) -> Tuple[bool, str]:
        """
        Remove item directly from backend cart
        
        Args:
            session: User session
            item_id: ID of item to remove
            
        Returns:
            Tuple of (success, message)
        """
        try:
            user_id = session.user_id or "guestUser"
            device_id = session.device_id
            
            if not device_id:
                return False, " Session not properly initialized"
            
            logger.info(f"[Cart] Removing from backend cart - User: {user_id}, Device: {device_id}, Item: {item_id}")
            
            # Call backend remove API
            result = await self.buyer_app.remove_multiple_cart_items(user_id, device_id, [item_id])
            
            if result and not result.get('error'):
                logger.info(f"[Cart] Successfully removed item from backend cart")
                return True, f" Item removed from cart"
            else:
                error_msg = result.get('message', 'Failed to remove item') if result else 'Backend error'
                logger.error(f"[Cart] Backend remove failed: {error_msg}")
                return False, f" {error_msg}"
                
        except Exception as e:
            logger.error(f"[Cart] Error removing item from backend cart: {e}")
            return False, f" Failed to remove item: {str(e)}"
    
    async def update_quantity(self, session: Session, item_id: str, 
                            quantity: int) -> Tuple[bool, str]:
        """
        Update item quantity in backend cart
        
        Args:
            session: User session
            item_id: ID of item to update
            quantity: New quantity (0 to remove)
            
        Returns:
            Tuple of (success, message)
        """
        try:
            # Validate quantity
            if quantity is None:
                return False, f" Quantity cannot be empty. Must be between 0 and 100."
            
            if not isinstance(quantity, int) or quantity < 0 or quantity > 100:
                return False, f" Invalid quantity. Must be between 0 and 100."
            
            user_id = session.user_id or "guestUser"
            device_id = session.device_id
            
            if not device_id:
                return False, " Session not properly initialized"
            
            if quantity == 0:
                # Remove item if quantity is 0
                return await self.remove_item(session, item_id)
            
            logger.info(f"[Cart] Updating backend cart quantity - User: {user_id}, Device: {device_id}, Item: {item_id}, Qty: {quantity}")
            
            # Call backend update API
            result = await self.buyer_app.update_cart_item(user_id, device_id, item_id, quantity)
            
            if result and not result.get('error'):
                logger.info(f"[Cart] Successfully updated item quantity in backend cart")
                return True, f" Updated item quantity to {quantity}"
            else:
                error_msg = result.get('message', 'Failed to update quantity') if result else 'Backend error'
                logger.error(f"[Cart] Backend update failed: {error_msg}")
                return False, f" {error_msg}"
                
        except Exception as e:
            logger.error(f"[Cart] Error updating item quantity: {e}")
            return False, f" Failed to update quantity: {str(e)}"
    
    async def clear_cart(self, session: Session) -> Tuple[bool, str]:
        """
        Clear all items from backend cart
        
        Args:
            session: User session
            
        Returns:
            Tuple of (success, message)
        """
        try:
            user_id = session.user_id or "guestUser"
            device_id = session.device_id
            
            if not device_id:
                return False, " Session not properly initialized"
            
            logger.info(f"[Cart] Clearing backend cart - User: {user_id}, Device: {device_id}")
            
            # Call backend clear cart API
            result = await self.buyer_app.clear_cart(user_id, device_id)
            
            if result and not result.get('error'):
                logger.info(f"[Cart] Successfully cleared backend cart")
                return True, " Cart cleared"
            else:
                error_msg = result.get('message', 'Failed to clear cart') if result else 'Backend error'
                logger.error(f"[Cart] Backend clear failed: {error_msg}")
                return False, f" {error_msg}"
                
        except Exception as e:
            logger.error(f"[Cart] Error clearing backend cart: {e}")
            return False, f" Failed to clear cart: {str(e)}"
    
    async def get_cart_summary(self, session: Session) -> Dict[str, Any]:
        """
        Get cart summary from backend
        
        Args:
            session: User session
            
        Returns:
            Cart summary dictionary
        """
        try:
            success, _, cart_items = await self.view_cart(session)
            
            if not success or not cart_items:
                return {
                    'items': [],
                    'total_items': 0,
                    'total_value': 0.0,
                    'is_empty': True
                }
            
            # Calculate totals from backend data
            total_items = 0
            total_value = 0.0
            
            for item in cart_items:
                # Use actual backend format - count is directly on item
                quantity = item.get('count', 1)
                # Use the same price extraction logic as in display formatting
                price = self._extract_item_price(item)
                
                total_items += quantity
                total_value += float(price) * quantity
            
            return {
                'items': cart_items,
                'total_items': total_items,
                'total_value': total_value,
                'is_empty': len(cart_items) == 0
            }
            
        except Exception as e:
            logger.error(f"[Cart] Error getting cart summary: {e}")
            return {
                'items': [],
                'total_items': 0,
                'total_value': 0.0,
                'is_empty': True
            }
    
    def _format_backend_cart_display(self, cart_items: List[Dict]) -> str:
        """
        Format backend cart data for display
        
        Args:
            cart_items: Backend cart items
            
        Returns:
            Formatted cart string
        """
        if not cart_items:
            return " **Your cart is empty**\n\nStart shopping by searching for products!"
        
        lines = [
            f" **Your Cart ({len(cart_items)} items)**",
            ""
        ]
        
        total_items = 0
        total_value = 0.0
        
        for i, item in enumerate(cart_items, 1):
            # Extract item details from backend format
            name = self._extract_item_name(item)
            # Extract quantity from backend format (count field for backend cart items)
            quantity = item.get('count', item.get('quantity', {}).get('count', 1))
            price = self._extract_item_price(item)
            provider = self._extract_provider_name(item)
            
            # Handle missing price data gracefully
            if price > 0:
                subtotal = price * quantity
                total_value += subtotal
                price_display = f"₹{price:.2f} x {quantity} = ₹{subtotal:.2f}"
            else:
                # Price not available in cart data - show quantity only
                price_display = f"Quantity: {quantity} (Price not available in cart)"
            
            total_items += quantity
            
            # Format item line
            provider_info = f" (from {provider})" if provider else ""
            lines.append(
                f"{i}. **{name}**{provider_info}\n"
                f"   {price_display}"
            )
        
        lines.extend([
            "",
            f"**Total Items: {total_items}**",
            f"**Cart Value: {'₹{:.2f}'.format(total_value) if total_value > 0 else 'Price data unavailable in cart'}**",
            "\n Ready to checkout? Use the checkout tools to proceed!"
        ])
        
        return "\n".join(lines)
    
    def _extract_item_name(self, item: Dict) -> str:
        """Extract item name from backend cart item - handles actual Himira format"""
        # Actual backend format analysis from logs:
        # {"_id": "...", "item_id": "2f788d22-cc2b-4f90-9faf-90bfdfae947f", "id": "...", "provider_id": "...", "count": 1}
        
        # Try nested product.descriptor.name first (full product details if available)
        if item.get('item', {}).get('product', {}).get('descriptor', {}).get('name'):
            return item['item']['product']['descriptor']['name']
        # Try item.descriptor.name
        elif item.get('descriptor', {}).get('name'):
            return item['descriptor']['name']
        # Try item.name fallback
        elif item.get('name'):
            return item['name']
        # Handle actual backend format - use item_id as fallback identifier
        elif item.get('item_id'):
            return f"Product {item['item_id'][:8]}..."  # Show first 8 chars of item_id
        elif item.get('id'):
            # Extract product part from ONDC ID if possible
            product_id = item['id'].split('_')[-1] if '_' in item['id'] else item['id']
            return f"Product {product_id[:8]}..."
        else:
            return 'Unknown Product'
    
    def _extract_item_price(self, item: Dict) -> float:
        """Extract item price from backend cart item - handles actual Himira format"""
        # Actual backend format doesn't include price - cart items are just references
        # Backend format: {"item_id": "...", "count": 1, "provider_id": "..."} - NO PRICE DATA
        
        # Try nested product.price first (full product details if available)
        if item.get('item', {}).get('product', {}).get('price', {}).get('value'):
            return float(item['item']['product']['price']['value'])
        # Try item.price
        price = item.get('price', {})
        if isinstance(price, dict):
            return float(price.get('value', 0))
        elif price:
            return float(price)
        
        # Backend cart items don't include price - return 0 for now
        # TODO: Later enhancement could do separate product lookup for price
        return 0.0
    
    def _extract_provider_name(self, item: Dict) -> str:
        """Extract provider name from backend cart item - handles actual Himira format"""
        # Actual backend format has provider_id field like: "hp-seller-preprod.himira.co.in_ONDC:RET18_..."
        
        # Try nested provider structure first (if full data available)
        provider = item.get('provider', {})
        if isinstance(provider, dict):
            descriptor = provider.get('descriptor', {})
            if descriptor.get('name'):
                return descriptor['name']
            elif provider.get('name'):
                return provider['name']
        
        # Handle actual backend format - extract from provider_id
        provider_id = item.get('provider_id', '')
        if provider_id:
            # Extract readable part from ONDC provider ID
            # Format: "hp-seller-preprod.himira.co.in_ONDC:RET18_d871c2ae-bf3f-4d3c-963f-f85f94848e8c"
            if 'himira' in provider_id.lower():
                return 'Himira Store'
            elif '_ONDC:' in provider_id:
                # Extract the part before _ONDC:
                return provider_id.split('_ONDC:')[0].replace('hp-seller-preprod.', '').replace('.himira.co.in', '')
            else:
                return provider_id[:20] + '...' if len(provider_id) > 20 else provider_id
        
        return 'Unknown Provider'
    
    # DEPRECATED: No longer needed with pure backend-driven architecture
    async def sync_with_backend(self, session: Session) -> bool:
        """Deprecated - cart is now purely backend-driven"""
        logger.warning("[Cart] sync_with_backend called but deprecated in pure backend mode")
        return True
    
    def _create_backend_cart_payload(self, product: Dict[str, Any], quantity: int) -> Dict[str, Any]:
        """
        Create cart payload for backend API using ONDC schemas
        
        Args:
            product: Product data from search results
            quantity: Quantity to add
            
        Returns:
            Backend cart payload
        """
        try:
            # Use centralized ONDC factory to create item
            product_data = product.copy()
            product_data['quantity'] = quantity
            
            # Create ONDC-compliant cart item
            ondc_item = ONDCDataFactory.create_cart_item(product_data, auto_enrich=True)
            
            # Enhance for backend (applies field mapping)
            backend_payload = enhance_for_backend(ondc_item)
            
            logger.debug(f"[Cart] Created backend cart payload for {product.get('name')}")
            return backend_payload
            
        except Exception as e:
            logger.error(f"[Cart] Error creating backend cart payload: {e}")
            # Fallback to basic payload
            return {
                "id": product.get('id', ''),
                "local_id": product.get('local_id', product.get('id', '')),
                "quantity": {"count": quantity},
                "provider": product.get('provider', {}),
                "customisations": None,
                "hasCustomisations": False,
                "customisationState": {}
            }
    # ================================
    # LEGACY METHODS - Kept for compatibility but deprecated
    # ================================
    
    async def remove_multiple_items(self, session: Session, item_ids: List[str]) -> Tuple[bool, str]:
        """
        Remove multiple items directly from backend cart
        
        Args:
            session: User session
            item_ids: List of item IDs to remove
            
        Returns:
            Tuple of (success, message)
        """
        try:
            if not item_ids:
                return False, " No items specified to remove"
            
            user_id = session.user_id or "guestUser"
            device_id = session.device_id
            
            if not device_id:
                return False, " Session not properly initialized"
            
            logger.info(f"[Cart] Removing multiple items from backend cart - User: {user_id}, Device: {device_id}, Items: {len(item_ids)}")
            
            # Call backend API for multiple removal
            result = await self.buyer_app.remove_multiple_cart_items(user_id, device_id, item_ids)
            
            if result and not result.get('error'):
                logger.info(f"[Cart] Successfully removed {len(item_ids)} items from backend cart")
                return True, f" Removed {len(item_ids)} items from cart"
            else:
                error_msg = result.get('message', 'Failed to remove items') if result else 'Backend error'
                logger.error(f"[Cart] Backend remove multiple failed: {error_msg}")
                return False, f" {error_msg}"
                
        except Exception as e:
            logger.error(f"[Cart] Error removing multiple items: {e}")
            return False, f" Failed to remove items: {str(e)}"
    
    # DEPRECATED: Wishlist functionality removed in pure backend mode
    async def move_to_wishlist(self, session: Session, item_ids: List[str] = None) -> Tuple[bool, str]:
        """Deprecated - wishlist operations not implemented in pure backend mode"""
        return False, " Wishlist functionality not available"
    
    # DEPRECATED: Recommendations removed in pure backend mode
    async def get_cart_recommendations(self, session: Session) -> Optional[List[Dict]]:
        """Deprecated - recommendations not implemented in pure backend mode"""
        return None


    # DEPRECATED: Analytics removed in pure backend mode
    def get_cart_analytics(self, session: Session) -> Dict[str, Any]:
        """Deprecated - analytics not implemented in pure backend mode"""
        return {
            'total_items': 0,
            'total_value': 0,
            'categories': [],
            'providers': [],
            'average_item_price': 0
        }


    # This method is now the main view_cart method above


# Singleton instance
_cart_service: Optional[CartService] = None


def get_cart_service() -> CartService:
    """Get singleton CartService instance"""
    global _cart_service
    if _cart_service is None:
        _cart_service = CartService()
    return _cart_service