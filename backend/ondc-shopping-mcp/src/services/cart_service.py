"""Cart service for managing shopping cart operations"""

from typing import Dict, List, Optional, Any, Tuple
import logging
import json

from ..models.session import Session, Cart, CartItem
from ..buyer_backend_client import BuyerBackendClient
from ..utils.logger import get_logger
from ..utils.device_id import get_or_create_device_id
from ..utils.himira_provider_constants import (
    enrich_cart_item_with_provider, 
    HIMIRA_BPP_ID, 
    HIMIRA_BPP_URI,
    HIMIRA_LOCATION_LOCAL_ID,
    generate_himira_item_id,
    create_minimal_provider_for_cart
)
from ..data_models.ondc_schemas import ONDCDataFactory, create_cart_item as create_ondc_item
from ..utils.field_mapper import enhance_for_backend, FieldMapper

logger = get_logger(__name__)


class CartService:
    """Service for managing cart operations with clean separation of concerns"""
    
    def __init__(self, buyer_backend_client: Optional[BuyerBackendClient] = None):
        """
        Initialize cart service
        
        Args:
            buyer_backend_client: Comprehensive client for backend API calls
        """
        self.buyer_app = buyer_backend_client or BuyerBackendClient()
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
            # Check if user is authenticated
            if not session.user_authenticated or not session.auth_token or not session.user_id:
                return False, " Please login to add items to cart"
            
            # Use session device ID for consistency  
            device_id = session.device_id
            
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
    
    
    def get_cart_summary(self, session: Session) -> Dict[str, Any]:
        """
        Get cart summary
        
        Args:
            session: User session
            
        Returns:
            Cart summary dictionary
        """
        # Debug logging to identify list vs CartItem issue
        logger.debug(f"[Cart] Debug - session.cart.items type: {type(session.cart.items)}")
        logger.debug(f"[Cart] Debug - session.cart.items length: {len(session.cart.items) if hasattr(session.cart.items, '__len__') else 'No length'}")
        
        cart_items = []
        for i, item in enumerate(session.cart.items):
            logger.debug(f"[Cart] Debug - item {i} type: {type(item)}")
            logger.debug(f"[Cart] Debug - item {i} hasattr to_dict: {hasattr(item, 'to_dict')}")
            if hasattr(item, 'to_dict'):
                cart_items.append(item.to_dict())
            else:
                logger.error(f"[Cart] Error - item {i} is not a CartItem object: {type(item)} - {item}")
                # Try to handle gracefully
                if isinstance(item, dict):
                    cart_items.append(item)
                else:
                    logger.error(f"[Cart] Error - cannot convert item to dict: {item}")
        
        return {
            'items': cart_items,
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
            device_id = session.device_id
            
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
            
            logger.info(f"[Cart] Successfully synced cart with backend for session {session.session_id}")
            return True
                
        except Exception as e:
            logger.error(f"[Cart] Failed to sync cart with backend: {e}")
            return False
    
    def _create_cart_item(self, product: Dict[str, Any], quantity: int) -> CartItem:
        """
        Create BIAP-compatible CartItem from product data with proper Himira provider structure
        
        Args:
            product: Product data dictionary
            quantity: Quantity
            
        Returns:
            CartItem object with BIAP fields and proper ONDC provider data
        """
        # Handle price which might be nested
        price = product.get('price', 0)
        if isinstance(price, dict):
            price = price.get('value', 0)
        
        # Get category for provider serviceability
        category = product.get('category', 'Tinned and Processed Food')
        
        # ENHANCED: Use proper Himira provider data from Postman collection
        # This ensures cart items have the exact structure expected by the backend
        logger.info(f"[Cart] Creating cart item with Himira provider data for: {product.get('name')}")
        
        # Extract local_id from full ONDC ID before enrichment
        product_data = product.copy()
        if product.get('id') and '_' in product['id']:
            # Extract local_id from full ONDC format
            id_parts = product['id'].split('_')
            if len(id_parts) >= 4:
                product_data['local_id'] = id_parts[-1]
                logger.info(f"[Cart] Extracted local_id {id_parts[-1]} from full ID {product['id']}")
        
        # Enrich basic product data with proper ONDC provider structure
        enriched_data = enrich_cart_item_with_provider(product_data, category)
        
        # Extract enriched ONDC fields from Himira constants
        product_id = enriched_data['id']  # Full ONDC ID format from Himira
        local_id = enriched_data['local_id']  # UUID format
        bpp_id = enriched_data['bpp_id']  # hp-seller-preprod.himira.co.in
        bpp_uri = enriched_data['bpp_uri']  # Full BPP URI
        location_id = enriched_data['location_id']  # Provider location ID
        provider_data = enriched_data['provider']  # Full provider structure from Postman
        contextCity = enriched_data['contextCity']  # std:0172 for Kharar/Mohali
        
        logger.info(f"[Cart] Enriched item with Himira ONDC data:")
        logger.info(f"  - Product ID: {product_id}")
        logger.info(f"  - Local ID: {local_id}")
        logger.info(f"  - BPP ID: {bpp_id}")
        logger.info(f"  - Provider ID: {provider_data['id']}")
        logger.info(f"  - Location ID: {location_id}")
        logger.info(f"  - Context City: {contextCity}")
        
        # Create CartItem with enriched Himira ONDC structure
        cart_item = CartItem(
            id=product_id,  # Use enriched ONDC ID format
            name=product.get('name', 'Unknown Product'),
            price=float(price),
            quantity=quantity,
            local_id=local_id,            # UUID from enriched data
            bpp_id=bpp_id,               # Himira BPP ID
            bpp_uri=bpp_uri,             # Himira BPP URI
            location_id=location_id,     # Himira location ID
            contextCity=contextCity,     # std:0172 for area
            category=category,           # Product category
            image_url=product.get('image_url') or enriched_data.get('image_url'),
            description=product.get('description') or enriched_data.get('description'),
            provider=provider_data,      # Full Himira provider structure from Postman
            fulfillment_id="1",         # Default fulfillment
            tags=enriched_data.get('tags', []),
            customisations=enriched_data.get('customisations'),
        )
        
        logger.info(f"[Cart] Successfully created ONDC cart item with Himira provider data: {cart_item.name}")
        logger.debug(f"[Cart] Provider structure: {json.dumps(provider_data, indent=2)}")
        
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
            device_id = session.device_id
            
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
            device_id = session.device_id
            
            # Use backend API for multiple removal
            logger.info(f"[Cart] Calling backend API to remove {len(item_ids)} items for user {user_id}")
            backend_result = await self.buyer_app.remove_multiple_cart_items(user_id, device_id, item_ids)
            
            # Check backend result success
            if backend_result is None:
                logger.error(f"[Cart] Backend API call failed - returned None")
                return False, " Failed to remove items from backend"
            
            # Log backend response for debugging
            logger.info(f"[Cart] Backend API response: {backend_result}")
            
            # Check if backend operation was successful
            backend_success = backend_result.get('success', True)  # Default to True for backward compatibility
            if not backend_success:
                error_msg = backend_result.get('message', 'Unknown backend error')
                logger.error(f"[Cart] Backend removal failed: {error_msg}")
                return False, f" Backend failed to remove items: {error_msg}"
            
            # Only proceed if backend succeeded
            logger.info(f"[Cart] Backend removal successful, updating local state")
            
            # Add to history
            session.add_to_history('remove_multiple_from_cart', {
                'item_ids': item_ids,
                'removed_count': removed_count,
                'item_names': removed_names,
                'backend_success': True
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
            device_id = session.device_id
            
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
            "quantity": {"count": quantity},  # FIXED: Always use the provided quantity parameter
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
            
            # Use session device ID for consistency  
            device_id = session.device_id
            
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

    async def sync_backend_to_local_cart(self, session: Session) -> bool:
        """
        Sync backend cart data to local session.cart
        
        This implements the missing synchronization that was referenced in TODO comments.
        Fetches cart from backend and populates local session.cart for existing operations.
        
        Args:
            session: User session
            
        Returns:
            bool: True if sync successful, False otherwise
        """
        try:
            user_id = session.user_id or "guestUser"
            device_id = session.device_id
            
            logger.info(f"[Cart] Syncing backend cart to local - User: {user_id}, Device: {device_id}")
            
            # Fetch from backend using same API as original working view_cart
            backend_result = await self.buyer_app.get_cart(user_id, device_id)
            
            # Handle response format
            if backend_result is None:
                logger.info(f"[Cart] Backend returned empty cart for {user_id}")
                session.cart.clear()
                return True
                
            # Check for explicit error
            if isinstance(backend_result, dict) and backend_result.get('error'):
                logger.warning(f"[Cart] Backend cart sync failed: {backend_result.get('message', 'Unknown error')}")
                return False
                
            # Get cart items array
            if isinstance(backend_result, list):
                backend_items = backend_result
            elif isinstance(backend_result, dict) and 'data' in backend_result:
                backend_items = backend_result.get('data', [])
            else:
                logger.warning(f"[Cart] Unexpected backend response format: {type(backend_result)}")
                return False
            
            # Clear existing local cart
            session.cart.clear()
            
            # Convert and add each backend item to local cart
            for backend_item in backend_items:
                try:
                    cart_item = self._convert_backend_item_to_cart_item(backend_item)
                    if cart_item:
                        session.cart.add_item(cart_item)
                        logger.debug(f"[Cart] Synced item: {cart_item.name} x{cart_item.quantity}")
                except Exception as e:
                    logger.warning(f"[Cart] Failed to convert backend item: {e}")
                    continue
            
            logger.info(f"[Cart] Sync complete - {len(session.cart.items)} items in local cart")
            return True
            
        except Exception as e:
            logger.error(f"[Cart] Error syncing backend to local cart: {e}")
            return False

    def _convert_backend_item_to_cart_item(self, backend_item: Dict[str, Any]) -> Optional[CartItem]:
        """
        Convert backend cart item to local CartItem object
        
        Args:
            backend_item: Backend cart item response
            
        Returns:
            CartItem object or None if conversion fails
        """
        try:
            # Add type check to prevent "'list' object has no attribute 'get'" error
            if not isinstance(backend_item, dict):
                logger.error(f"[Cart] Expected dict, got {type(backend_item)}: {backend_item}")
                return None
            
            # Extract from backend structure - backend response shows item contains descriptor/price directly
            item_data = backend_item.get('item', {})
            if not isinstance(item_data, dict):
                logger.error(f"[Cart] item_data is not dict: {type(item_data)}")
                return None
                
            # FIXED: descriptor and price are nested in item_data.product
            product_details = item_data.get('product', {})
            if not isinstance(product_details, dict):
                logger.error(f"[Cart] product_details is not dict: {type(product_details)}")
                product_details = {}
                
            descriptor = product_details.get('descriptor', {})
            price_data = product_details.get('price', {})
            
            # Extract essential fields with safe access
            item_id = backend_item.get('id', backend_item.get('item_id', ''))
            name = descriptor.get('name', 'Unknown Product') if isinstance(descriptor, dict) else 'Unknown Product'
            price = float(price_data.get('value', 0)) if isinstance(price_data, dict) else 0.0
            quantity = int(backend_item.get('count', 1))
            
            # Extract provider information safely - provider_id is at backend_item level
            provider_id = backend_item.get('provider_id', '')
            # Provider details are in item_data for structured provider info
            provider_data = item_data.get('provider', {})
            if not isinstance(provider_data, dict):
                provider_data = {}
            
            # Safe image URL extraction (this was the problematic line!)
            image_url = None
            if isinstance(descriptor, dict) and descriptor.get('images'):
                images = descriptor.get('images', [])
                if isinstance(images, list) and len(images) > 0:
                    first_image = images[0]
                    if isinstance(first_image, dict):
                        image_url = first_image.get('url', '')
                    elif isinstance(first_image, str):
                        image_url = first_image
            
            # Safe description extraction
            description = ''
            if isinstance(descriptor, dict):
                description = descriptor.get('short_desc') or descriptor.get('long_desc') or ''
            
            # Create CartItem using required and optional fields
            cart_item = CartItem(
                id=item_id,
                name=name,
                price=price,
                quantity=quantity,
                local_id=item_data.get('local_id', ''),
                bpp_id=item_data.get('bpp_id', ''),
                bpp_uri=item_data.get('bpp_uri', ''),
                contextCity=item_data.get('contextCity'),
                category=item_data.get('category_id', ''),
                image_url=image_url,
                description=description,
                product=item_data,
                provider=provider_data,
                location_id=item_data.get('location_id', ''),
                fulfillment_id=item_data.get('fulfillment_id', ''),
                parent_item_id=item_data.get('parent_item_id', ''),
                tags=item_data.get('tags'),
                customisations=backend_item.get('customisations')
            )
            
            logger.debug(f"[Cart] Successfully converted: {name} x{quantity} - ₹{price}")
            
            return cart_item
            
        except Exception as e:
            logger.error(f"[Cart] Error converting backend item to CartItem: {e}")
            return None
    
    async def get_formatted_cart_view(self, session: Session) -> Dict[str, Any]:
        """
        DRY method: Get formatted cart view with proper backend data parsing
        Used by both view_cart and add_to_cart tools
        
        Args:
            session: User session
            
        Returns:
            Dictionary with cart_display, cart_summary, and raw_backend_data
        """
        try:
            if not session.user_authenticated or not session.user_id:
                logger.info(f"[Cart] Using local cart for unauthenticated session")
                return {
                    'cart_display': self.format_cart_display(session),
                    'cart_summary': self.get_cart_summary(session),
                    'raw_backend_data': None,
                    'source': 'local_session'
                }
            
            # Get fresh backend data with credential logging
            user_id = session.user_id
            device_id = session.device_id
            logger.info(f"[Cart] 🔄 DRY SERVICE: Fetching cart for user={user_id}, device={device_id}")
            
            raw_backend_data = await self.buyer_app.get_cart(user_id, device_id)
            logger.info(f"[Cart] ✅ DRY SERVICE: Backend returned {len(raw_backend_data) if isinstance(raw_backend_data, list) else 'invalid'} items")
            
            if not raw_backend_data or not isinstance(raw_backend_data, list):
                logger.info(f"[Cart] DRY SERVICE: Empty backend response")
                return {
                    'cart_display': "🛒 **Your cart is empty**\n\nStart shopping by searching for products!",
                    'cart_summary': {'total_items': 0, 'total_value': 0.0, 'is_empty': True, 'items_count': 0},
                    'raw_backend_data': [],
                    'source': 'backend_empty'
                }
            
            # Parse backend data with CORRECT structure paths
            total_items = sum(item.get('count', 0) for item in raw_backend_data)
            total_value = sum(
                item.get('count', 0) * 
                float(item.get('item', {}).get('product', {}).get('price', {}).get('value', 0))
                for item in raw_backend_data
            )
            
            cart_summary = {
                'total_items': total_items,
                'total_value': total_value,
                'is_empty': total_items == 0,
                'items_count': len(raw_backend_data)
            }
            
            logger.info(f"[Cart] DRY SERVICE: Parsed {total_items} items, ₹{total_value:.2f} total")
            
            if total_items == 0:
                cart_display = "🛒 **Your cart is empty**\n\nStart shopping by searching for products!"
            else:
                cart_display = f"🛒 **Your Cart ({total_items} items)**\n\n"
                
                for i, item in enumerate(raw_backend_data, 1):
                    # FIXED: Use correct backend structure paths
                    item_details = item.get('item', {})
                    product_details = item_details.get('product', {})
                    descriptor = product_details.get('descriptor', {})
                    price_info = product_details.get('price', {})
                    provider_info = item_details.get('provider', {}).get('descriptor', {})
                    quantity = item.get('count', 0)
                    
                    name = descriptor.get('name', 'Unknown Item')
                    unit_price = float(price_info.get('value', 0))
                    subtotal = quantity * unit_price
                    provider = provider_info.get('name', 'Unknown Store')
                    category = product_details.get('category_id', 'Unknown Category')
                    
                    cart_display += f"{i}. **{name}** (from {provider})\n"
                    cart_display += f"   ₹{unit_price:.2f} x {quantity} = ₹{subtotal:.2f}\n"
                    cart_display += f"   Category: {category}\n\n"
                
                cart_display += f"**Total: ₹{total_value:.2f}**"
            
            return {
                'cart_display': cart_display,
                'cart_summary': cart_summary,
                'raw_backend_data': raw_backend_data,
                'source': 'backend_fresh'
            }
            
        except Exception as e:
            logger.error(f"[Cart] ❌ DRY SERVICE: Failed to get cart view: {e}")
            # Fallback to local session data
            return {
                'cart_display': f"❌ **Cart view failed**: {str(e)}\n\nUsing local session data:\n" + self.format_cart_display(session),
                'cart_summary': self.get_cart_summary(session),
                'raw_backend_data': None,
                'source': 'local_fallback'
            }


# Singleton instance
_cart_service: Optional[CartService] = None


def get_cart_service() -> CartService:
    """Get singleton CartService instance"""
    global _cart_service
    if _cart_service is None:
        _cart_service = CartService()
    return _cart_service