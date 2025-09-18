"""
Comprehensive ONDC Order Service

Implements the complete ONDC order flow:
1. SELECT - Select items and get quotes
2. INIT - Initialize order with delivery details  
3. CONFIRM - Confirm order after payment
4. STATUS - Track order status
5. CANCEL - Cancel orders if needed
6. UPDATE - Update order details

Based on buyer backend APIs from biap-client-node-js
"""

import json
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import logging

from ..buyer_backend_client import BuyerBackendClient
from ..models.session import Session
from ..utils.logger import get_logger

logger = get_logger(__name__)


class OrderService:
    """Comprehensive service for ONDC order management"""
    
    def __init__(self, buyer_backend_client: Optional[BuyerBackendClient] = None):
        """
        Initialize order service
        
        Args:
            buyer_backend_client: Comprehensive client for backend API calls
        """
        self.buyer_app = buyer_backend_client or BuyerBackendClient()
        logger.info("OrderService initialized with comprehensive ONDC flow support")
    
    # ================================
    # ONDC ORDER FLOW - STEP 1: SELECT
    # ================================
    
    async def select_items_for_order(
        self, 
        session: Session, 
        items: List[Dict[str, Any]],
        delivery_location: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, str, Optional[Dict]]:
        """
        ONDC SELECT: Select items for order and get quote
        
        This is the first step in ONDC order flow.
        
        Args:
            session: User session
            items: List of items to select for order
            delivery_location: Delivery location details
            
        Returns:
            Tuple of (success, message, select_response)
        """
        try:
            logger.info(f"[Order] Starting SELECT flow for {len(items)} items")
            
            # Prepare SELECT request data with all required ONDC fields
            select_data = {
                "items": items,
                "delivery_location": delivery_location or {
                    "gps": "30.7455808,76.6537325",  # Default Bangalore coordinates
                    "address": {
                        "city": "Bangalore",
                        "state": "Karnataka",
                        "area_code": "560001",
                        "street": "MG Road",
                        "locality": "Central Bangalore"
                    }
                },
                "city": "std:080",  # Required city code for ONDC
                "domain": "ONDC:RET14",  # Required domain for electronics/retail
                "country": "IND",
                "user_id": session.user_id,
                "device_id": session.device_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Call ONDC SELECT API
            result = await self.buyer_app.select_items(select_data)
            
            if result and not result.get('error'):
                # Wait for response
                await asyncio.sleep(2)  # Allow time for backend processing
                
                # Get SELECT response
                select_response = await self.buyer_app.get_select_response(
                    user_id=session.user_id,
                    message_id=result.get('message_id')
                )
                
                if select_response and not select_response.get('error'):
                    # Update session with order details
                    session.checkout_state.transaction_id = result.get('transaction_id')
                    
                    # Add to session history
                    session.add_to_history('order_select', {
                        'items_count': len(items),
                        'transaction_id': result.get('transaction_id'),
                        'message_id': result.get('message_id')
                    })
                    
                    logger.info(f"[Order] SELECT successful - Transaction ID: {result.get('transaction_id')}")
                    
                    return True, " Items selected and quote received", select_response
                else:
                    logger.error(f"[Order] SELECT response failed: {select_response}")
                    return False, f" Failed to get quote: {select_response.get('error', 'Unknown error')}", None
            else:
                logger.error(f"[Order] SELECT request failed: {result}")
                return False, f" Failed to select items: {result.get('error', 'Unknown error')}", None
                
        except Exception as e:
            logger.error(f"[Order] SELECT flow error: {e}")
            return False, f" Order selection failed: {str(e)}", None
    
    # ================================
    # ONDC ORDER FLOW - STEP 2: INIT
    # ================================
    
    async def initialize_order(
        self, 
        session: Session, 
        delivery_info: Dict[str, Any],
        auth_token: str,
        billing_info: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, str, Optional[Dict]]:
        """
        ONDC INIT: Initialize order with delivery and billing details
        
        This is the second step in ONDC order flow.
        
        Args:
            session: User session
            delivery_info: Delivery address and contact details
            auth_token: User authentication token
            billing_info: Billing information (optional)
            
        Returns:
            Tuple of (success, message, init_response)
        """
        try:
            if not session.checkout_state.transaction_id:
                return False, " No active transaction. Please select items first.", None
            
            logger.info(f"[Order] Starting INIT flow for transaction: {session.checkout_state.transaction_id}")
            
            # Prepare INIT request data
            init_data = {
                "transaction_id": session.checkout_state.transaction_id,
                "delivery_info": delivery_info,
                "billing_info": billing_info or delivery_info,
                "user_id": session.user_id,
                "device_id": session.device_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Call ONDC INIT API
            result = await self.buyer_app.initialize_order(init_data, auth_token)
            
            if result and not result.get('error'):
                # Wait for response
                await asyncio.sleep(2)  # Allow time for backend processing
                
                # Get INIT response
                init_response = await self.buyer_app.get_init_response(
                    auth_token,
                    transaction_id=session.checkout_state.transaction_id
                )
                
                if init_response and not init_response.get('error'):
                    # Update session with delivery info
                    session.checkout_state.delivery_info = delivery_info
                    
                    from ..models.session import CheckoutStage
                    session.checkout_state.stage = CheckoutStage.INIT
                    
                    # Add to session history
                    session.add_to_history('order_init', {
                        'transaction_id': session.checkout_state.transaction_id,
                        'delivery_address': delivery_info.get('address'),
                        'payment_required': init_response.get('payment_required', True)
                    })
                    
                    logger.info(f"[Order] INIT successful for transaction: {session.checkout_state.transaction_id}")
                    
                    return True, " Order initialized with delivery details", init_response
                else:
                    logger.error(f"[Order] INIT response failed: {init_response}")
                    return False, f" Failed to initialize order: {init_response.get('error', 'Unknown error')}", None
            else:
                logger.error(f"[Order] INIT request failed: {result}")
                return False, f" Failed to initialize order: {result.get('error', 'Unknown error')}", None
                
        except Exception as e:
            logger.error(f"[Order] INIT flow error: {e}")
            return False, f" Order initialization failed: {str(e)}", None
    
    # ================================
    # ONDC ORDER FLOW - STEP 3: CONFIRM
    # ================================
    
    async def confirm_order(
        self, 
        session: Session, 
        payment_info: Dict[str, Any],
        auth_token: str
    ) -> Tuple[bool, str, Optional[Dict]]:
        """
        ONDC CONFIRM: Confirm order after payment
        
        This is the final step in ONDC order flow.
        
        Args:
            session: User session
            payment_info: Payment confirmation details
            auth_token: User authentication token
            
        Returns:
            Tuple of (success, message, confirm_response)
        """
        try:
            if not session.checkout_state.transaction_id:
                return False, " No active transaction. Please initialize order first.", None
            
            logger.info(f"[Order] Starting CONFIRM flow for transaction: {session.checkout_state.transaction_id}")
            
            # Prepare CONFIRM request data
            confirm_data = {
                "transaction_id": session.checkout_state.transaction_id,
                "payment_info": payment_info,
                "user_id": session.user_id,
                "device_id": session.device_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Call ONDC CONFIRM API
            result = await self.buyer_app.confirm_order(confirm_data, auth_token)
            
            if result and not result.get('error'):
                # Wait for response
                await asyncio.sleep(3)  # Allow more time for order confirmation
                
                # Get CONFIRM response
                confirm_response = await self.buyer_app.get_confirm_response(
                    auth_token,
                    transaction_id=session.checkout_state.transaction_id
                )
                
                if confirm_response and not confirm_response.get('error'):
                    # Update session with order details
                    from ..models.session import CheckoutStage
                    session.checkout_state.stage = CheckoutStage.CONFIRMED
                    session.checkout_state.order_id = confirm_response.get('order_id')
                    session.checkout_state.payment_method = payment_info.get('method', 'unknown')
                    
                    # Add to session history
                    session.add_to_history('order_confirm', {
                        'transaction_id': session.checkout_state.transaction_id,
                        'order_id': confirm_response.get('order_id'),
                        'payment_method': payment_info.get('method'),
                        'amount': payment_info.get('amount')
                    })
                    
                    # Clear cart after successful order
                    session.cart.clear()
                    
                    logger.info(f"[Order] CONFIRM successful - Order ID: {confirm_response.get('order_id')}")
                    
                    order_id = confirm_response.get('order_id', 'N/A')
                    return True, f" Order confirmed successfully! Order ID: {order_id}", confirm_response
                else:
                    logger.error(f"[Order] CONFIRM response failed: {confirm_response}")
                    return False, f" Order confirmation failed: {confirm_response.get('error', 'Unknown error')}", None
            else:
                logger.error(f"[Order] CONFIRM request failed: {result}")
                return False, f" Failed to confirm order: {result.get('error', 'Unknown error')}", None
                
        except Exception as e:
            logger.error(f"[Order] CONFIRM flow error: {e}")
            return False, f" Order confirmation failed: {str(e)}", None
    
    # ================================
    # ORDER MANAGEMENT
    # ================================
    
    async def get_order_status(
        self, 
        session: Session, 
        order_id: Optional[str] = None,
        auth_token: Optional[str] = None
    ) -> Tuple[bool, str, Optional[Dict]]:
        """
        Get current order status
        
        Args:
            session: User session
            order_id: Specific order ID (optional)
            auth_token: Authentication token for user orders
            
        Returns:
            Tuple of (success, message, status_response)
        """
        try:
            target_order_id = order_id or session.checkout_state.order_id
            
            if not target_order_id:
                return False, " No order ID available", None
            
            logger.info(f"[Order] Getting status for order: {target_order_id}")
            
            # Prepare status request
            status_data = {
                "order_id": target_order_id,
                "user_id": session.user_id,
                "device_id": session.device_id
            }
            
            # Get order status
            if auth_token:
                result = await self.buyer_app.get_order_status(status_data, auth_token)
            else:
                # Try to get order by transaction ID if available
                if session.checkout_state.transaction_id:
                    result = await self.buyer_app.get_order_by_transaction(session.checkout_state.transaction_id)
                else:
                    return False, " Authentication required for order status", None
            
            if result and not result.get('error'):
                logger.info(f"[Order] Status retrieved for order: {target_order_id}")
                return True, f" Order status retrieved for {target_order_id}", result
            else:
                logger.error(f"[Order] Status request failed: {result}")
                return False, f" Failed to get order status: {result.get('error', 'Unknown error')}", None
                
        except Exception as e:
            logger.error(f"[Order] Status check error: {e}")
            return False, f" Order status check failed: {str(e)}", None
    
    async def cancel_order(
        self, 
        session: Session, 
        cancellation_reason: str,
        auth_token: str,
        order_id: Optional[str] = None
    ) -> Tuple[bool, str, Optional[Dict]]:
        """
        Cancel an order
        
        Args:
            session: User session
            cancellation_reason: Reason for cancellation
            auth_token: Authentication token
            order_id: Specific order ID (optional)
            
        Returns:
            Tuple of (success, message, cancel_response)
        """
        try:
            target_order_id = order_id or session.checkout_state.order_id
            
            if not target_order_id:
                return False, " No order ID available for cancellation", None
            
            logger.info(f"[Order] Cancelling order: {target_order_id}")
            
            # Prepare cancel request
            cancel_data = {
                "order_id": target_order_id,
                "cancellation_reason": cancellation_reason,
                "user_id": session.user_id,
                "device_id": session.device_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Call cancel API
            result = await self.buyer_app.cancel_order(cancel_data, auth_token)
            
            if result and not result.get('error'):
                # Add to session history
                session.add_to_history('order_cancel', {
                    'order_id': target_order_id,
                    'reason': cancellation_reason,
                    'timestamp': datetime.utcnow().isoformat()
                })
                
                logger.info(f"[Order] Cancellation requested for order: {target_order_id}")
                return True, f" Order cancellation requested for {target_order_id}", result
            else:
                logger.error(f"[Order] Cancel request failed: {result}")
                return False, f" Failed to cancel order: {result.get('error', 'Unknown error')}", None
                
        except Exception as e:
            logger.error(f"[Order] Cancel order error: {e}")
            return False, f" Order cancellation failed: {str(e)}", None
    
    async def track_order(
        self, 
        session: Session, 
        auth_token: str,
        order_id: Optional[str] = None
    ) -> Tuple[bool, str, Optional[Dict]]:
        """
        Track an order
        
        Args:
            session: User session
            auth_token: Authentication token
            order_id: Specific order ID (optional)
            
        Returns:
            Tuple of (success, message, tracking_response)
        """
        try:
            target_order_id = order_id or session.checkout_state.order_id
            
            if not target_order_id:
                return False, " No order ID available for tracking", None
            
            logger.info(f"[Order] Tracking order: {target_order_id}")
            
            # Prepare track request
            track_data = {
                "order_id": target_order_id,
                "user_id": session.user_id,
                "device_id": session.device_id
            }
            
            # Call track API  
            result = await self.buyer_app.track_order(track_data, auth_token)
            
            if result and not result.get('error'):
                logger.info(f"[Order] Tracking info retrieved for order: {target_order_id}")
                return True, f" Tracking info for order {target_order_id}", result
            else:
                logger.error(f"[Order] Track request failed: {result}")
                return False, f" Failed to track order: {result.get('error', 'Unknown error')}", None
                
        except Exception as e:
            logger.error(f"[Order] Track order error: {e}")
            return False, f" Order tracking failed: {str(e)}", None
    
    async def get_order_history(
        self, 
        session: Session, 
        auth_token: str,
        limit: int = 10,
        offset: int = 0
    ) -> Tuple[bool, str, Optional[List[Dict]]]:
        """
        Get user's order history
        
        Args:
            session: User session
            auth_token: Authentication token
            limit: Number of orders to fetch
            offset: Offset for pagination
            
        Returns:
            Tuple of (success, message, orders_list)
        """
        try:
            logger.info(f"[Order] Getting order history for user: {session.user_id}")
            
            # Call orders API
            result = await self.buyer_app.get_orders(
                auth_token,
                limit=limit,
                offset=offset,
                user_id=session.user_id
            )
            
            if result and not result.get('error'):
                orders = result.get('orders', result.get('data', []))
                logger.info(f"[Order] Retrieved {len(orders) if isinstance(orders, list) else 0} orders")
                return True, f" Order history retrieved ({len(orders) if isinstance(orders, list) else 0} orders)", orders
            else:
                logger.error(f"[Order] Order history request failed: {result}")
                return False, f" Failed to get order history: {result.get('error', 'Unknown error')}", None
                
        except Exception as e:
            logger.error(f"[Order] Order history error: {e}")
            return False, f" Order history retrieval failed: {str(e)}", None
    
    # ================================
    # UTILITY METHODS
    # ================================
    
    def get_ondc_flow_status(self, session: Session) -> Dict[str, Any]:
        """
        Get current ONDC flow status
        
        Args:
            session: User session
            
        Returns:
            ONDC flow status information
        """
        checkout = session.checkout_state
        
        return {
            'current_stage': checkout.stage.value,
            'transaction_id': checkout.transaction_id,
            'order_id': checkout.order_id,
            'has_delivery_info': checkout.delivery_info is not None,
            'payment_method': checkout.payment_method,
            'next_steps': self._get_next_steps(checkout.stage),
            'can_proceed': self._can_proceed_to_next_step(session)
        }
    
    def _get_next_steps(self, current_stage) -> List[str]:
        """
        Get next steps based on current stage
        
        Args:
            current_stage: Current checkout stage
            
        Returns:
            List of next step descriptions
        """
        from ..models.session import CheckoutStage
        
        if current_stage == CheckoutStage.NONE:
            return ["Add items to cart", "Select items for order (get quote)"]
        elif current_stage == CheckoutStage.SELECT:
            return ["Initialize order with delivery details"]
        elif current_stage == CheckoutStage.INIT:
            return ["Complete payment", "Confirm order"]
        elif current_stage == CheckoutStage.CONFIRMED:
            return ["Track order", "Check order status"]
        else:
            return []
    
    def _can_proceed_to_next_step(self, session: Session) -> bool:
        """
        Check if can proceed to next step
        
        Args:
            session: User session
            
        Returns:
            True if can proceed
        """
        from ..models.session import CheckoutStage
        
        checkout = session.checkout_state
        
        if checkout.stage == CheckoutStage.NONE:
            return not session.cart.is_empty()
        elif checkout.stage == CheckoutStage.SELECT:
            return checkout.transaction_id is not None
        elif checkout.stage == CheckoutStage.INIT:
            return checkout.delivery_info is not None
        elif checkout.stage == CheckoutStage.CONFIRMED:
            return checkout.order_id is not None
        else:
            return False
    
    def format_order_summary(self, order_data: Dict[str, Any]) -> str:
        """
        Format order data for display
        
        Args:
            order_data: Order information
            
        Returns:
            Formatted order summary string
        """
        try:
            order_id = order_data.get('order_id', order_data.get('id', 'N/A'))
            status = order_data.get('status', order_data.get('order_status', 'Unknown'))
            total = order_data.get('total_amount', order_data.get('amount', 0))
            items = order_data.get('items', [])
            
            lines = [
                f" **Order #{order_id}**",
                f" Status: {status}",
                f" Total: ₹{total:.2f}" if isinstance(total, (int, float)) else f" Total: {total}",
                ""
            ]
            
            if items and isinstance(items, list):
                lines.append(" **Items:**")
                for item in items[:5]:  # Show first 5 items
                    item_name = item.get('name', item.get('item_name', 'Unknown item'))
                    quantity = item.get('quantity', item.get('qty', 1))
                    lines.append(f"  • {item_name} (Qty: {quantity})")
                
                if len(items) > 5:
                    lines.append(f"  • ... and {len(items) - 5} more items")
            
            return "\n".join(lines)
            
        except Exception as e:
            logger.error(f"[Order] Format order summary error: {e}")
            return f"Order: {order_data.get('order_id', 'Unknown')}"


# Singleton instance
_order_service: Optional[OrderService] = None


def get_order_service() -> OrderService:
    """Get singleton OrderService instance"""
    global _order_service
    if _order_service is None:
        _order_service = OrderService()
    return _order_service