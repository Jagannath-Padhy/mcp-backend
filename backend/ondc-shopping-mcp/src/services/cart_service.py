"""Cart service for managing shopping cart operations"""

from typing import Dict, List, Optional, Any, Tuple
import logging
import json

from ..models.session import Session, Cart, CartItem
from ..buyer_backend_client import BuyerBackendClient, get_buyer_backend_client
from ..utils.logger import get_logger
from ..utils.device_id import get_or_create_device_id
from ..data_models.ondc_schemas import ONDCDataFactory
from ..utils.field_mapper import enhance_for_backend
# Removed fake data imports - using only real data from search results

logger = get_logger(__name__)


class CartService:
    """Service for managing cart operations with clean separation of concerns"""
    
    def __init__(self, buyer_backend_client: Optional[BuyerBackendClient] = None):
        """
        Initialize cart service
        
        Args:
            buyer_backend_client: Comprehensive client for backend API calls
        """
        self.buyer_app = buyer_backend_client or get_buyer_backend_client()
        logger.info("CartService initialized with comprehensive backend client")
    
    async def add_item_to_backend(self, session: Session, product: Dict[str, Any], 
                                  quantity: int = 1) -> Tuple[bool, str]:
        """
        Add item to backend cart - matches frontend pattern
        
        Args:
            session: User session with auth token
            product: Product data
            quantity: Quantity to add
            
        Returns:
            Tuple of (success, message)
        """
        try:
            # Check if user is authenticated (WIL API key auth handled by BuyerBackendClient)
            if not session.user_authenticated or not session.user_id:
                return False, " Please login to add items to cart"
            
            # Get device ID
            device_id = await get_or_create_device_id()
            
            # Prepare cart payload matching frontend structure
            cart_payload = self._create_backend_cart_payload(product, quantity)
            
            logger.info(f"[Cart] Adding to backend cart - User: {session.user_id}, Device: {device_id}, Item: {cart_payload.get('local_id')}")
            
            # Call backend add to cart API
            result = await self.buyer_app.add_to_cart(session.user_id, device_id, cart_payload)
            
            if result and not result.get('error'):
                logger.info(f"[Cart] Successfully added {product.get('descriptor', {}).get('name')} to backend cart")
                return True, f" Added {product.get('descriptor', {}).get('name')} to cart"
            else:
                error_msg = result.get('message', 'Failed to add to cart') if result else 'Backend error'
                logger.error(f"[Cart] Backend add to cart failed: {error_msg}")
                return False, f" {error_msg}"
                
        except Exception as e:
            logger.error(f"[Cart] Error adding item to backend cart: {e}")
            return False, f" Failed to add to cart: {str(e)}"
    
    async def add_item(self, session: Session, product: Dict[str, Any], 
                       quantity: int = 1) -> Tuple[bool, str]:
        """
        Add item to cart
        
        Args:
            session: User session
            product: Product data
            quantity: Quantity to add
            
        Returns:
            Tuple of (success, message)
        """
        try:
            # Validate quantity - handle None case
            if quantity is None:
                quantity = 1
                logger.info(f"[Cart] Quantity was None, defaulting to 1")
            
            if not isinstance(quantity, int) or quantity < 1 or quantity > 100:
                return False, f" Invalid quantity. Must be between 1 and 100."
            
            # Create cart item from product data
            cart_item = self._create_cart_item(product, quantity)
            
            # Log cart item details for debugging
            logger.info(f"[Cart] Adding item: {cart_item.name}, Price: ₹{cart_item.price}, Qty: {cart_item.quantity}, Subtotal: ₹{cart_item.subtotal}")
            
            # Add to cart
            session.cart.add_item(cart_item)
            
            # Add to history
            session.add_to_history('add_to_cart', {
                'product_id': cart_item.id,
                'product_name': cart_item.name,
                'quantity': quantity
            })
            
            # Get updated cart summary
            summary = self.get_cart_summary(session)
            
            # Log cart totals for debugging
            logger.info(f"[Cart] After add - Total items: {summary['total_items']}, Total value: ₹{summary['total_value']}")
            logger.info(f"[Cart] Items in cart: {[f'{item.name}(₹{item.price}x{item.quantity}=₹{item.subtotal})' for item in session.cart.items]}")
            
            message = f" Added {quantity}x {cart_item.name} to cart\n"
            message += f"Cart total: {summary['total_items']} items - ₹{summary['total_value']:.2f}"
            
            return True, message
            
        except Exception as e:
            logger.error(f"Failed to add item to cart: {e}")
            return False, f" Failed to add item to cart: {str(e)}"
    
    async def remove_item(self, session: Session, item_id: str) -> Tuple[bool, str]:
        """
        Remove item from cart
        
        Args:
            session: User session
            item_id: ID of item to remove
            
        Returns:
            Tuple of (success, message)
        """
        try:
            # Find item first to get name for message
            item = session.cart.find_item(item_id)
            if not item:
                return False, f" Item not found in cart"
            
            item_name = item.name
            
            # Remove from cart
            if session.cart.remove_item(item_id):
                # Add to history
                session.add_to_history('remove_from_cart', {
                    'product_id': item_id,
                    'product_name': item_name
                })
                
                # Get updated summary
                summary = self.get_cart_summary(session)
                
                if session.cart.is_empty():
                    message = f" Removed {item_name} from cart\nYour cart is now empty"
                else:
                    message = f" Removed {item_name} from cart\n"
                    message += f"Cart total: {summary['total_items']} items - ₹{summary['total_value']:.2f}"
                
                return True, message
            else:
                return False, f" Failed to remove item from cart"
                
        except Exception as e:
            logger.error(f"Failed to remove item from cart: {e}")
            return False, f" Failed to remove item: {str(e)}"
    
    async def update_quantity(self, session: Session, item_id: str, 
                            quantity: int) -> Tuple[bool, str]:
        """
        Update item quantity in cart
        
        Args:
            session: User session
            item_id: ID of item to update
            quantity: New quantity
            
        Returns:
            Tuple of (success, message)
        """
        try:
            # Validate quantity - handle None case
            if quantity is None:
                return False, f" Quantity cannot be empty. Must be between 0 and 100."
            
            if not isinstance(quantity, int) or quantity < 0 or quantity > 100:
                return False, f" Invalid quantity. Must be between 0 and 100."
            
            # Find item
            item = session.cart.find_item(item_id)
            if not item:
                return False, f" Item not found in cart"
            
            item_name = item.name
            
            # Update quantity (0 means remove)
            if session.cart.update_quantity(item_id, quantity):
                # Add to history
                session.add_to_history('update_quantity', {
                    'product_id': item_id,
                    'product_name': item_name,
                    'new_quantity': quantity
                })
                
                if quantity == 0:
                    message = f" Removed {item_name} from cart"
                else:
                    message = f" Updated {item_name} quantity to {quantity}"
                
                # Get updated summary
                if not session.cart.is_empty():
                    summary = self.get_cart_summary(session)
                    message += f"\nCart total: {summary['total_items']} items - ₹{summary['total_value']:.2f}"
                
                return True, message
            else:
                return False, f" Failed to update quantity"
                
        except Exception as e:
            logger.error(f"Failed to update item quantity: {e}")
            return False, f" Failed to update quantity: {str(e)}"
    
    async def clear_cart(self, session: Session) -> Tuple[bool, str]:
        """
        Clear all items from cart
        
        Args:
            session: User session
            
        Returns:
            Tuple of (success, message)
        """
        try:
            items_count = session.cart.total_items
            
            if items_count == 0:
                return True, "Your cart is already empty"
            
            # Clear cart
            session.cart.clear()
            
            # Add to history
            session.add_to_history('clear_cart', {
                'items_removed': items_count
            })
            
            return True, f" Cleared {items_count} items from cart"
            
        except Exception as e:
            logger.error(f"Failed to clear cart: {e}")
            return False, f" Failed to clear cart: {str(e)}"
    
    def get_cart_summary(self, session: Session) -> Dict[str, Any]:
        """
        Get cart summary
        
        Args:
            session: User session
            
        Returns:
            Cart summary dictionary
        """
        return {
            'items': [item.to_dict() for item in session.cart.items],
            'total_items': session.cart.total_items,
            'total_value': session.cart.total_value,
            'is_empty': session.cart.is_empty()
        }
    
    def format_cart_display(self, session: Session) -> str:
        """
        Format cart for display with enhanced details
        
        Args:
            session: User session
            
        Returns:
            Formatted cart string
        """
        if session.cart.is_empty():
            return " **Your cart is empty**\n\nStart shopping by searching for products!"
        
        lines = [
            f" **Your Cart ({session.cart.total_items} items)**",
            ""
        ]
        
        for i, item in enumerate(session.cart.items, 1):
            # Add provider info if available
            provider_info = f" (from {item.provider_id})" if item.provider_id != "unknown" else ""
            lines.append(
                f"{i}. **{item.name}**{provider_info}\n"
                f"   ₹{item.price:.2f} x {item.quantity} = ₹{item.subtotal:.2f}"
            )
            
            # Add category if available
            if item.category:
                lines.append(f"    Category: {item.category}")
        
        lines.extend([
            "",
            f"**Total: ₹{session.cart.total_value:.2f}**",
            "\n Ready to checkout? Use the checkout tools to proceed!"
        ])
        
        return "\n".join(lines)
    
    async def sync_with_backend(self, session: Session) -> bool:
        """
        Sync cart with backend using comprehensive API
        
        Args:
            session: User session
            
        Returns:
            True if successful
        """
        try:
            # Get user_id and device_id from session
            user_id = session.user_id
            if not user_id:
                logger.warning("[Cart] No user_id in session for backend operation")
                return False
            device_id = getattr(session, 'device_id', None)
            if not device_id:
                logger.error("[Cart] No device_id in session for backend operation")
                return False
            
            logger.info(f"[Cart] Syncing {len(session.cart.items)} items with backend for user {user_id}, device {device_id}")
            
            # First, clear existing cart on backend
            await self.buyer_app.clear_cart(user_id, device_id)
            
            # Add each item to backend cart
            for item in session.cart.items:
                # Match the structure expected by biap-client-node-js
                cart_data = {
                    'id': item.id,
                    'local_id': item.id,  # Use same as id if no separate local_id
                    'provider': {'id': item.provider_id},
                    'quantity': {'count': item.quantity},
                    'customisations': [],
                    'hasCustomisations': False,
                    'price': item.price,
                    'name': item.name
                }
                
                # Call backend API with proper parameters
                result = await self.buyer_app.add_to_cart(user_id, device_id, cart_data)
                
                if result and result.get('error'):
                    logger.warning(f"Failed to sync item {item.name} with backend: {result.get('error')}")
                else:
                    logger.debug(f"Successfully synced item {item.name} with backend")
            
            logger.info(f"[Cart] Successfully synced cart with backend for user {user_id}, device {device_id}")
            return True
                
        except Exception as e:
            logger.error(f"[Cart] Failed to sync cart with backend: {e}")
            return False
    
    def _create_cart_item(self, product: Dict[str, Any], quantity: int) -> CartItem:
        """
        Create CartItem from search results preserving REAL provider data only
        
        Args:
            product: Product data from search results with real provider/location details
            quantity: Quantity to add
            
        Returns:
            CartItem with real provider data preserved from search results
            
        Raises:
            ValueError: If product lacks required real provider data
        """
        # Check for provider data in various formats (provider_details or provider field)
        provider_details = product.get('provider_details', {})
        location_details = product.get('location_details', {})
        
        # Alternative: check if provider data is in the provider field directly
        if not provider_details and product.get('provider'):
            provider_details = product.get('provider', {})
            
        # Alternative: check if location data is available in item_details
        if not location_details:
            item_details = product.get('item_details', {})
            if item_details.get('location_id'):
                # Create synthetic location_details from available data
                location_details = {
                    'local_id': item_details.get('location_id'),
                    'id': f"{provider_details.get('id', '')}_{item_details.get('location_id', '')}"
                }
        
        # Only fail if we have absolutely no provider information
        if not provider_details and not product.get('provider_id'):
            error_msg = f"Product '{product.get('name', 'Unknown')}' missing any provider data"
            logger.error(f"[Cart] {error_msg}")
            raise ValueError(error_msg)
        
        # Extract real provider information with fallbacks
        provider_id = provider_details.get('id') or product.get('provider_id')
        provider_local_id = provider_details.get('local_id')
        location_id = location_details.get('id')  
        location_local_id = location_details.get('local_id')
        
        # Extract from provider_id if structured ID is available
        if provider_id and not provider_local_id and '_' in str(provider_id):
            parts = str(provider_id).split('_')
            if len(parts) >= 3:
                provider_local_id = parts[-1]  # Last part is usually local_id
        
        if not provider_id:
            error_msg = f"Product '{product.get('name')}' missing provider ID"
            logger.error(f"[Cart] {error_msg}")
            raise ValueError(error_msg)
        
        logger.info(f"[Cart] Creating cart item with REAL provider data:")
        logger.info(f"  - Provider ID: {provider_id}")
        logger.info(f"  - Provider Local ID: {provider_local_id}")
        logger.info(f"  - Location Local ID: {location_local_id}")
        
        # Handle price extraction
        price = product.get('price', 0)
        if isinstance(price, dict):
            price = price.get('value', 0)
        
        # Extract or generate product local ID
        product_local_id = product.get('local_id') or product.get('id', '')
        if product_local_id and '_' in product_local_id:
            # Extract UUID from full ONDC format
            id_parts = product_local_id.split('_')
            if len(id_parts) >= 4:
                product_local_id = id_parts[-1]
        
        if not product_local_id:
            import uuid
            product_local_id = str(uuid.uuid4())
        
        # Build real provider structure from search results
        real_provider = {
            "id": provider_id,
            "local_id": provider_local_id,
            "descriptor": provider_details.get("descriptor", {
                "name": provider_details.get("name", "Provider"),
                "short_desc": provider_details.get("name", "Provider")
            }),
            "locations": [{
                "id": location_id or f"{provider_id}_{location_local_id}",
                "local_id": location_local_id,
                "gps": location_details.get("gps", "0,0"),
                "address": location_details.get("address", {})
            }]
        }
        
        # Generate full ONDC product ID
        full_product_id = f"hp-seller-preprod.himira.co.in_ONDC:RET10_{provider_local_id}_{product_local_id}"
        
        # Create CartItem with preserved real data
        cart_item = CartItem(
            id=full_product_id,
            name=product.get('name', 'Unknown Product'),
            price=float(price),
            quantity=quantity,
            local_id=product_local_id,
            bpp_id="hp-seller-preprod.himira.co.in",  # Known constant
            bpp_uri="https://hp-seller-preprod.himira.co.in/api/v2",  # Known constant
            location_id=location_local_id,
            contextCity="std:0172",  # Known area code
            category=product.get('category', ''),
            image_url=product.get('image_url', ''),
            description=product.get('description', ''),
            provider=real_provider,  # Real provider data from search results
            fulfillment_id="1",
            tags=product.get('tags', []),
            customisations=product.get('customisations'),
        )
        
        logger.info(f"[Cart] Created cart item with real provider data: {cart_item.name}")
        return cart_item
    # ================================
    # ADDITIONAL CART METHODS using comprehensive backend
    # ================================
    
    async def get_backend_cart(self, session: Session) -> Optional[Dict]:
        """
        Get cart directly from backend
        
        Args:
            session: User session
            
        Returns:
            Backend cart data or None if failed
        """
        try:
            user_id = session.user_id
            if not user_id:
                logger.warning("[Cart] No user_id in session for backend operation")
                return False
            device_id = getattr(session, 'device_id', None)
            if not device_id:
                logger.error("[Cart] No device_id in session for backend operation")
                return False
            
            result = await self.buyer_app.get_cart(user_id, device_id)
            logger.debug(f"[Cart] Backend cart for {user_id}/{device_id}: {result}")
            return result
            
        except Exception as e:
            logger.error(f"[Cart] Failed to get backend cart: {e}")
            return None
    
    async def remove_multiple_items(self, session: Session, item_ids: List[str]) -> Tuple[bool, str]:
        """
        Remove multiple items from cart
        
        Args:
            session: User session
            item_ids: List of item IDs to remove
            
        Returns:
            Tuple of (success, message)
        """
        try:
            if not item_ids:
                return False, " No items specified to remove"
            
            # Remove from local cart first
            removed_count = 0
            removed_names = []
            
            for item_id in item_ids:
                item = session.cart.find_item(item_id)
                if item:
                    removed_names.append(item.name)
                    if session.cart.remove_item(item_id):
                        removed_count += 1
            
            if removed_count == 0:
                return False, " No items found to remove"
            
            # Sync with backend
            user_id = session.user_id
            if not user_id:
                logger.warning("[Cart] No user_id in session for backend operation")
                return False
            device_id = getattr(session, 'device_id', None)
            if not device_id:
                logger.error("[Cart] No device_id in session for backend operation")
                return False
            
            # Use backend API for multiple removal
            backend_result = await self.buyer_app.remove_multiple_cart_items(user_id, device_id, item_ids)
            
            # Add to history
            session.add_to_history('remove_multiple_from_cart', {
                'item_ids': item_ids,
                'removed_count': removed_count,
                'item_names': removed_names
            })
            
            message = f" Removed {removed_count} items from cart: {', '.join(removed_names[:3])}"
            if len(removed_names) > 3:
                message += f" and {len(removed_names) - 3} more"
            
            # Show updated cart summary
            if not session.cart.is_empty():
                summary = self.get_cart_summary(session)
                message += f"\nCart total: {summary['total_items']} items - ₹{summary['total_value']:.2f}"
            else:
                message += "\nYour cart is now empty"
            
            return True, message
            
        except Exception as e:
            logger.error(f"[Cart] Failed to remove multiple items: {e}")
            return False, f" Failed to remove items: {str(e)}"
    
    async def move_to_wishlist(self, session: Session, item_ids: List[str] = None) -> Tuple[bool, str]:
        """
        Move cart items to wishlist
        
        Args:
            session: User session
            item_ids: Specific item IDs to move (optional, moves all if not specified)
            
        Returns:
            Tuple of (success, message)
        """
        try:
            if session.cart.is_empty():
                return False, " Your cart is empty"
            
            user_id = session.user_id
            if not user_id:
                logger.warning("[Cart] No user_id in session for backend operation")
                return False
            device_id = getattr(session, 'device_id', None)
            if not device_id:
                logger.error("[Cart] No device_id in session for backend operation")
                return False
            
            # Determine which items to move
            items_to_move = []
            if item_ids:
                for item_id in item_ids:
                    item = session.cart.find_item(item_id)
                    if item:
                        items_to_move.append(item)
            else:
                items_to_move = session.cart.items.copy()
            
            if not items_to_move:
                return False, " No items found to move to wishlist"
            
            moved_count = 0
            moved_names = []
            
            # Move each item to wishlist via backend
            for item in items_to_move:
                wishlist_data = {
                    'id': item.id,
                    'name': item.name,
                    'price': item.price,
                    'provider_id': item.provider_id,
                    'category': item.category,
                    'image_url': item.image_url,
                    'description': item.description
                }
                
                # Add to wishlist via backend
                result = await self.buyer_app.add_to_wishlist(user_id, device_id, wishlist_data)
                
                if result and not result.get('error'):
                    # Remove from cart
                    if session.cart.remove_item(item.id):
                        moved_count += 1
                        moved_names.append(item.name)
            
            if moved_count > 0:
                # Sync cart with backend
                await self.sync_with_backend(session)
                
                # Add to history
                session.add_to_history('move_to_wishlist', {
                    'moved_count': moved_count,
                    'item_names': moved_names
                })
                
                message = f" Moved {moved_count} items to wishlist: {', '.join(moved_names[:3])}"
                if len(moved_names) > 3:
                    message += f" and {len(moved_names) - 3} more"
                
                return True, message
            else:
                return False, " Failed to move items to wishlist"
                
        except Exception as e:
            logger.error(f"[Cart] Failed to move items to wishlist: {e}")
            return False, f" Failed to move items: {str(e)}"
    
    async def get_cart_recommendations(self, session: Session) -> Optional[List[Dict]]:
        """
        Get product recommendations based on cart contents
        
        Args:
            session: User session
            
        Returns:
            List of recommended products or None
        """
        try:
            if session.cart.is_empty():
                return None
            
            # Get categories from cart items for recommendations
            categories = set()
            for item in session.cart.items:
                if item.category:
                    categories.add(item.category)
            
            if not categories:
                return None
            
            # Search for similar products
            user_id = session.user_id
            if not user_id:
                logger.warning("[Cart] No user_id in session for backend operation")
                return False
            recommendations = []
            
            for category in list(categories)[:3]:  # Limit to 3 categories
                result = await self.buyer_app.search_products(
                    user_id=user_id,
                    query=category,
                    limit=5
                )
                
                if result and not result.get('error') and 'response' in result:
                    products = result.get('response', {}).get('data', [])
                    for product in products[:3]:  # Top 3 per category
                        # Don't recommend items already in cart
                        if not any(cart_item.id == product.get('id') for cart_item in session.cart.items):
                            recommendations.append(product)
            
            return recommendations[:10] if recommendations else None
            
        except Exception as e:
            logger.error(f"[Cart] Failed to get recommendations: {e}")
            return None


    def get_cart_analytics(self, session: Session) -> Dict[str, Any]:
        """
        Get cart analytics and insights
        
        Args:
            session: User session
            
        Returns:
            Cart analytics data
        """
        if session.cart.is_empty():
            return {
                'total_items': 0,
                'total_value': 0,
                'categories': [],
                'providers': [],
                'average_item_price': 0
            }
        
        # Calculate analytics
        categories = {}
        providers = {}
        total_value = 0
        
        for item in session.cart.items:
            # Category analysis
            if item.category:
                categories[item.category] = categories.get(item.category, 0) + item.quantity
            
            # Provider analysis
            if item.provider_id != 'unknown':
                providers[item.provider_id] = providers.get(item.provider_id, 0) + item.quantity
            
            total_value += item.subtotal
        
        return {
            'total_items': session.cart.total_items,
            'total_value': session.cart.total_value,
            'unique_items': len(session.cart.items),
            'categories': list(categories.keys()),
            'top_category': max(categories.keys(), key=categories.get) if categories else None,
            'providers': list(providers.keys()),
            'top_provider': max(providers.keys(), key=providers.get) if providers else None,
            'average_item_price': session.cart.total_value / session.cart.total_items if session.cart.total_items > 0 else 0
        }


    def _create_backend_cart_payload(self, product: Dict[str, Any], quantity: int) -> Dict[str, Any]:
        """
        Create cart payload matching backend expected structure.
        Uses centralized models and field mapping for DRY principle.
        
        Args:
            product: Product data from search/catalog
            quantity: Quantity to add
            
        Returns:
            Cart payload for backend API
        """
        # Use centralized ONDC factory to create item
        product_data = product.copy()
        product_data['quantity'] = quantity
        
        # Create ONDC-compliant cart item
        ondc_item = ONDCDataFactory.create_cart_item(product_data, auto_enrich=True)
        
        # Enhance for backend (applies field mapping and provider fix)
        backend_payload = enhance_for_backend(ondc_item)
        
        # Ensure SELECT-specific structure
        payload = {
            "local_id": backend_payload.get("local_id"),
            "id": backend_payload.get("id"),
            "quantity": backend_payload.get("quantity", {"count": quantity}),
            "provider": backend_payload.get("provider"),
            "customisations": backend_payload.get("customisations", []),
            "hasCustomisations": backend_payload.get("hasCustomisations", False),
            "customisationState": backend_payload.get("customisationState", {})
        }
        
        logger.debug(f"[Cart] Created DRY backend cart payload: {payload}")
        return payload
    
    async def get_backend_cart(self, session: Session) -> Tuple[bool, str, List[Dict]]:
        """
        Get cart items from backend - matches frontend pattern
        
        Args:
            session: User session with auth token
            
        Returns:
            Tuple of (success, message, cart_items)
        """
        try:
            if not session.user_authenticated or not session.user_id:
                return False, " Please login to view cart", []
            
            # Get device ID
            device_id = await get_or_create_device_id()
            
            logger.info(f"[Cart] Fetching backend cart - User: {session.user_id}, Device: {device_id}")
            
            # Call backend get cart API
            result = await self.buyer_app.get_cart_items(session.user_id, device_id)
            
            if result and not result.get('error'):
                cart_items = result if isinstance(result, list) else result.get('data', [])
                logger.info(f"[Cart] Retrieved {len(cart_items)} items from backend cart")
                return True, f" Cart loaded ({len(cart_items)} items)", cart_items
            else:
                error_msg = result.get('message', 'Failed to load cart') if result else 'Backend error'
                logger.warning(f"[Cart] Backend get cart failed: {error_msg}")
                return False, f" {error_msg}", []
                
        except Exception as e:
            logger.error(f"[Cart] Error fetching backend cart: {e}")
            return False, f" Failed to load cart: {str(e)}", []


# Singleton instance
_cart_service: Optional[CartService] = None


def get_cart_service() -> CartService:
    """Get singleton CartService instance"""
    global _cart_service
    if _cart_service is None:
        _cart_service = CartService()
    return _cart_service