"""Checkout service for managing checkout flow"""

from typing import Dict, List, Optional, Any, Tuple
import logging
import uuid
import json
import asyncio
import time

from ..models.session import Session, CheckoutState, CheckoutStage, DeliveryInfo
from ..buyer_backend_client import BuyerBackendClient
from ..data_models.biap_context_factory import get_biap_context_factory
from ..services.product_enrichment_service import get_product_enrichment_service
from ..services.biap_validation_service import get_biap_validation_service
from ..services.payment_mock_service import mock_payment_service
from ..utils.city_code_mapping import get_city_code_by_pincode
from ..utils.logger import get_logger
from ..config import config

# Import centralized utilities
from ..utils.ondc_constants import (
    get_city_gps,
    DEFAULT_GPS,
    ERROR_MESSAGES
)
from ..utils.location_utils import (
    transform_provider_locations,
    create_provider_for_context,
    extract_location_ids,
    build_location_objects
)

logger = get_logger(__name__)


class CheckoutService:
    """BIAP-compatible consolidated ONDC checkout service with 3 optimized methods"""
    
    def __init__(self, buyer_backend_client: Optional[BuyerBackendClient] = None):
        """
        Initialize checkout service with BIAP-compatible services
        
        Args:
            buyer_backend_client: Client for backend API calls
        """
        self.buyer_app = buyer_backend_client or BuyerBackendClient()
        self.context_factory = get_biap_context_factory()
        self.product_enrichment = get_product_enrichment_service()
        self.validation = get_biap_validation_service()
        logger.info("CheckoutService initialized with BIAP-compatible services")
    
    async def _poll_for_response(
        self,
        poll_function,
        message_id: str,
        operation_name: str,
        max_attempts: int = 15,
        initial_delay: float = 2.0,
        backoff_factor: float = 1.5,
        max_delay: float = 10.0,
        auth_token: Optional[str] = None
    ) -> Optional[Dict]:
        """
        Generic polling mechanism for asynchronous ONDC operations
        
        Args:
            poll_function: The function to call for polling (e.g., get_select_response)
            message_id: The message ID to poll for
            operation_name: Name of the operation (for logging)
            max_attempts: Maximum polling attempts
            initial_delay: Initial delay between polls in seconds
            backoff_factor: Factor to increase delay between attempts
            max_delay: Maximum delay between polls
            auth_token: Optional auth token for authenticated requests
            
        Returns:
            The completed response or None if timeout/error
        """
        logger.info(f"[Polling] Starting {operation_name} polling for messageId: {message_id}")
        
        delay = initial_delay
        attempts = 0
        
        while attempts < max_attempts:
            attempts += 1
            
            try:
                # Call the polling function with message ID
                logger.debug(f"[Polling] Attempt {attempts}/{max_attempts} for {operation_name}")
                
                if auth_token:
                    response = await poll_function(messageIds=message_id, auth_token=auth_token)
                else:
                    response = await poll_function(messageIds=message_id)
                
                logger.debug(f"[Polling] Response: {json.dumps(response, indent=2) if response else 'None'}")
                
                # Check if response is ready
                if response:
                    # Handle different response formats
                    # Some endpoints return array, some return object
                    if isinstance(response, list) and len(response) > 0:
                        response_data = response[0]
                    else:
                        response_data = response
                    
                    # Check status field
                    status = response_data.get('status', '').upper()
                    
                    if status == 'COMPLETED':
                        logger.info(f"[Polling]  {operation_name} completed successfully")
                        return response_data
                    elif status == 'FAILED':
                        logger.error(f"[Polling]  {operation_name} failed: {response_data.get('message', 'Unknown error')}")
                        return None
                    elif status in ['PROCESSING', 'PENDING', '']:
                        # Still processing, continue polling
                        logger.debug(f"[Polling] {operation_name} still processing (status: {status})")
                    else:
                        logger.warning(f"[Polling] Unknown status: {status}")
            
            except Exception as e:
                logger.error(f"[Polling] Error during {operation_name} polling attempt {attempts}: {e}")
            
            # Wait before next attempt
            if attempts < max_attempts:
                logger.debug(f"[Polling] Waiting {delay:.1f} seconds before next attempt")
                await asyncio.sleep(delay)
                
                # Exponential backoff with max delay
                delay = min(delay * backoff_factor, max_delay)
        
        logger.error(f"[Polling]  Timeout: {operation_name} did not complete after {attempts} attempts")
        return None
    
    async def select_items_for_order(
        self, 
        session: Session, 
        delivery_city: str,
        delivery_state: str,
        delivery_pincode: str
    ) -> Dict[str, Any]:
        """
        BIAP-compatible ONDC SELECT stage - Get quotes and fulfillment options
        
        Enhanced with:
        - Product enrichment using BIAP APIs
        - BIAP validation (multiple BPP/Provider checks)
        - Proper city code mapping
        - BIAP context factory
        
        Args:
            session: User session with cart items
            delivery_city: Delivery city
            delivery_state: Delivery state  
            delivery_pincode: Delivery pincode
            
        Returns:
            Quote and fulfillment options from ONDC SELECT
        """
        # Validate cart
        if session.cart.is_empty():
            return {
                'success': False,
                'message': ' Cart is empty. Please add items first.'
            }
        
        # Validate delivery location
        if not all([delivery_city, delivery_state, delivery_pincode]):
            return {
                'success': False,
                'message': ' Missing delivery location. Please provide city, state, and pincode.'
            }
        
        # Generate transaction ID for this checkout session
        session.checkout_state.transaction_id = self._generate_transaction_id()
        
        try:
            logger.info(f"[CheckoutService] Starting SELECT with {len(session.cart.items)} items")
            logger.debug(f"[CheckoutService] Delivery location: {delivery_city}, {delivery_state}, {delivery_pincode}")
            logger.debug(f"[CheckoutService] Cart items: {[{'name': item.name, 'id': item.id, 'local_id': getattr(item, 'local_id', None)} for item in session.cart.items]}")
            
            # Step 1: Enrich cart items with BIAP product data
            logger.info("[CheckoutService] Step 1: Enriching cart items with BIAP product data...")
            enriched_items = await self.product_enrichment.enrich_cart_items(
                session.cart.items, session.session_id
            )
            logger.debug(f"[CheckoutService] Enriched {len(enriched_items)} items")
            
            # Step 2: BIAP validation - check for multiple BPP/Provider items
            logger.info("[CheckoutService] Step 2: Validating order items for BIAP compliance...")
            validation_result = self.validation.validate_order_items(enriched_items, "select")
            logger.debug(f"[CheckoutService] Validation result: {validation_result}")
            if not validation_result.get('success'):
                logger.error(f"[CheckoutService] Validation failed: {validation_result.get('error', {})}")
                return {
                    'success': False,
                    'message': f" {validation_result['error']['message']}"
                }
            
            # Step 3: Get proper city code from pincode
            logger.info(f"[CheckoutService] Step 3: Getting city code for pincode {delivery_pincode}...")
            city_code = get_city_code_by_pincode(delivery_pincode)
            logger.debug(f"[CheckoutService] City code: {city_code}")
            
            # Step 4: Create BIAP-compatible context (match Himira documentation format)
            logger.info("[CheckoutService] Step 4: Creating BIAP-compatible context...")
            context = self.context_factory.create({
                'action': 'select',
                'transaction_id': session.checkout_state.transaction_id,
                'city': delivery_pincode,  #  CRITICAL: Use pincode directly as per Himira docs
                'pincode': delivery_pincode
            })
            logger.debug(f"[CheckoutService] Context created: {json.dumps(context, indent=2)}")
            
            # Step 5: Get BPP info from validated items
            logger.info("[CheckoutService] Step 5: Getting BPP info from validated items...")
            bpp_info = self.validation.get_order_bpp_info(enriched_items)
            logger.debug(f"[CheckoutService] BPP info: {bpp_info}")
            if bpp_info:
                context['bpp_id'] = bpp_info['bpp_id']
                context['bpp_uri'] = bpp_info['bpp_uri']
                logger.info(f"[CheckoutService] Using BPP: {bpp_info['bpp_id']} at {bpp_info['bpp_uri']}")
            
            # Step 6: Transform enriched items to BIAP SELECT format
            logger.info("[CheckoutService] Step 6: Transforming items to BIAP SELECT format...")
            select_items = []
            location_set = set()
            provider_info = None
            
            for item in enriched_items:
                # Create SELECT item structure (Postman collection format)
                select_item = {
                    'id': item.id,                   # Full ONDC ID
                    'local_id': item.local_id,       # UUID local ID
                    'quantity': {'count': item.quantity},  # Wrapped in count object
                    'customisationState': {},        # Required by Postman format
                    'customisations': None,          # Required by Postman format
                    'hasCustomisations': False       # Required by Postman format
                }
                
                # Add product structure for biap backend compatibility
                if item.location_id:
                    # CRITICAL FIX: biap backend expects item.product.location_id, not item.location_id
                    select_item['product'] = {
                        'location_id': item.location_id
                    }
                    # Also keep the direct location_id for backward compatibility
                    select_item['location_id'] = item.location_id
                    location_set.add(item.location_id)
                
                # Add provider at item level using utility
                if item.provider:
                    select_item['provider'] = create_provider_for_context(
                        item.provider, 
                        context="item"
                    )
                
                # Capture provider info from first item
                if not provider_info and item.provider:
                    provider_info = item.provider
                
                select_items.append(select_item)
            
            # Step 7: Build location objects using utility
            location_objs = build_location_objects(location_set, provider_info)
            
            # Step 8: Create SELECT request matching working Postman collection
            # Backend expects message.cart structure with provider at cart level
            
            # Create provider at cart level using utility
            cart_provider = create_provider_for_context(
                provider_info, 
                context="cart"
            ) if provider_info else None
            
            cart_obj = {
                'items': select_items  # Items have provider at item level
            }
            
            # Add provider at cart level if available (for backend transformation)
            if cart_provider:
                cart_obj['provider'] = cart_provider
                logger.debug(f"[CheckoutService] Cart provider: {json.dumps(cart_provider, indent=2)}")
            
            # Get GPS coordinates using centralized utility
            gps_coords = get_city_gps(delivery_city)
            logger.debug(f"[CheckoutService] GPS coordinates for {delivery_city}: {gps_coords}")
            
            #  CRITICAL: Simplify fulfillments to match Postman collection exactly
            fulfillments_obj = [{
                'end': {
                    'location': {
                        'gps': gps_coords,  #  Proper lat,lng coordinates
                        'address': {
                            'area_code': delivery_pincode  #  Only area_code like Postman
                        }
                    }
                }
            }]
            
            select_data = {
                'context': context,
                'message': {
                    'cart': cart_obj,             #  Backend expects 'cart' (confirmed from Postman collection)
                    'fulfillments': fulfillments_obj  #  At message level as backend expects
                },
                'userId': session.session_id,
                'deviceId': getattr(session, 'device_id', config.guest.device_id)
            }
            
            logger.info("[CheckoutService] Step 9: Calling BIAP SELECT API...")
            # Enhanced debug logging for SELECT request
            logger.info(f"[CheckoutService] SELECT request summary:")
            logger.info(f"  - Transaction ID: {context.get('transaction_id')}")
            logger.info(f"  - Items count: {len(select_items)}")
            logger.info(f"  - Delivery pincode: {delivery_pincode}")
            logger.info(f"  - Cart provider ID: {cart_provider['id'] if cart_provider else 'None'}")
            logger.info(f"  - Cart provider locations: {len(cart_provider.get('locations', [])) if cart_provider else 0}")
            
            # CRITICAL VALIDATION: Ensure provider locations match working curl format
            if cart_provider and 'locations' in cart_provider:
                logger.info(f"[CheckoutService] Validating provider location structure:")
                logger.info(f"  Provider ID: {cart_provider.get('id')}")
                logger.info(f"  Provider local_id: {cart_provider.get('local_id')}")
                
                for idx, loc in enumerate(cart_provider['locations']):
                    logger.info(f"  - Location[{idx}]: {json.dumps(loc)}")
                    if isinstance(loc, dict):
                        has_id = 'id' in loc
                        has_local_id = 'local_id' in loc
                        logger.info(f"    Has 'id' field: {has_id}")
                        logger.info(f"    Has 'local_id' field: {has_local_id}")
                        
                        # Validate against working curl format
                        if has_id and has_local_id:
                            # Verify ONDC format for ID (should contain underscores)
                            full_id = loc.get('id', '')
                            local_id = loc.get('local_id', '')
                            if '_' in full_id and full_id.endswith(local_id):
                                logger.info(f"    ✅ Location format matches curl: full_id={full_id}, local_id={local_id}")
                            else:
                                logger.warning(f"    ⚠️ Location ID format may not match ONDC standard")
                        else:
                            # CRITICAL FIX: Ensure both fields are present
                            if not has_id and has_local_id:
                                logger.warning(f"[CheckoutService] Location missing 'id' field, using local_id as fallback")
                                loc['id'] = loc['local_id']
                            elif not has_id:
                                logger.error(f"[CheckoutService] Location missing both 'id' and 'local_id' fields!")
                                # Set default location ID
                                from ..utils.ondc_constants import HIMIRA_LOCATION_LOCAL_ID
                                loc['id'] = HIMIRA_LOCATION_LOCAL_ID
                                logger.info(f"[CheckoutService] Set default location id: {HIMIRA_LOCATION_LOCAL_ID}")
                                
                        # Final validation after any fixes
                        if 'id' in loc and 'local_id' in loc:
                            logger.info(f"    ✅ Location validation passed: {json.dumps(loc)}")
                        else:
                            logger.error(f"    ❌ Location validation failed: missing required fields")
            else:
                logger.warning(f"[CheckoutService] No provider locations found in cart_provider: {cart_provider}")
            
            logger.debug(f"[CheckoutService] Full SELECT request payload:\n{json.dumps(select_data, indent=2)}")
            
            # GUEST MODE: SELECT API call without authentication
            auth_token = getattr(session, 'auth_token', None)
            if not auth_token:
                logger.info("[CheckoutService] GUEST MODE - Calling SELECT without auth token")
                # For guest users, we'll use wil-api-key authentication only
                auth_token = None
            else:
                logger.info("[CheckoutService] Using auth token for SELECT request")
            
            # Call BIAP SELECT API (works with or without auth token for guest)
            # FIXED: Remove double array wrapping - buyer_backend_client will wrap correctly
            select_response = await self.buyer_app.select_items(select_data, auth_token=auth_token)
            
            logger.info(f"[CheckoutService] SELECT API initial response received")
            logger.debug(f"[CheckoutService] SELECT initial response: {json.dumps(select_response, indent=2) if select_response else 'None'}")
            
            # Extract message ID from response for polling
            # FIXED: For array responses, extract messageId from first item's context
            # Matches frontend pattern: data?.map((txn) => txn.context?.message_id)
            message_id = None
            if select_response:
                if isinstance(select_response, list) and len(select_response) > 0:
                    # Expected successful array response format
                    first_item = select_response[0]
                    if isinstance(first_item, dict):
                        context = first_item.get('context', {})
                        if isinstance(context, dict):
                            message_id = context.get('message_id')
                            logger.info(f"[CheckoutService] Extracted messageId from array response: {message_id}")
                elif isinstance(select_response, dict):
                    # Fallback for single object responses (backwards compatibility)
                    context = select_response.get('context', {})
                    if isinstance(context, dict):
                        message_id = context.get('message_id')
                        logger.info(f"[CheckoutService] Extracted messageId from dict response: {message_id}")
                        
            # Log response structure for debugging if no messageId found
            if not message_id:
                logger.error(f"[CheckoutService] No messageId found in response structure:")
                logger.error(f"  Response type: {type(select_response)}")
                if isinstance(select_response, list):
                    logger.error(f"  Array length: {len(select_response)}")
                    if len(select_response) > 0:
                        logger.error(f"  First item type: {type(select_response[0])}")
                        if isinstance(select_response[0], dict):
                            logger.error(f"  First item keys: {list(select_response[0].keys())}")
                elif isinstance(select_response, dict):
                    logger.error(f"  Dict keys: {list(select_response.keys())}")
            
            if not message_id:
                logger.error(f"[CheckoutService] No messageId in SELECT response: {select_response}")
                return {
                    'success': False,
                    'message': ' Failed to initiate SELECT request. No message ID received.'
                }
            
            logger.info(f"[CheckoutService] Polling for SELECT response with messageId: {message_id}")
            
            # Poll for the actual SELECT response
            result = await self._poll_for_response(
                poll_function=self.buyer_app.get_select_response,
                message_id=message_id,
                operation_name="SELECT",
                max_attempts=15,
                initial_delay=2.0,
                auth_token=auth_token
            )
            
            logger.debug(f"[CheckoutService] Final SELECT response after polling: {json.dumps(result, indent=2) if result else 'None'}")
            
            if result and result.get('success', True) and 'error' not in result:
                # Update session to SELECT stage
                session.checkout_state.stage = CheckoutStage.SELECT
                session.add_to_history('select_items_for_order', {
                    'city': delivery_city,
                    'state': delivery_state, 
                    'pincode': delivery_pincode,
                    'items_count': len(session.cart.items)
                })
                
                logger.info("[CheckoutService]  SELECT successful - delivery quotes received")
                return {
                    'success': True,
                    'message': f' Delivery available in {delivery_city}! Quotes ready.',
                    'stage': 'select_completed',
                    'quote_data': result,
                    'next_step': 'provide_complete_delivery_details'
                }
            else:
                error_msg = result.get('message', 'Failed to get delivery quotes') if result else 'SELECT API failed'
                logger.error(f"[CheckoutService] SELECT failed: {error_msg}")
                logger.error(f"[CheckoutService] Full error response: {result}")
                return {
                    'success': False,
                    'message': f' {error_msg}'
                }
                
        except Exception as e:
            logger.error(f"[CheckoutService] SELECT operation failed with exception: {e}", exc_info=True)
            return {
                'success': False,
                'message': ' Failed to get delivery options. Please try again.'
            }
    
    async def initialize_order(
        self,
        session: Session,
        customer_name: str,
        delivery_address: str,
        phone: str,
        email: str,
        payment_method: str = 'cod',
        city: Optional[str] = None,
        state: Optional[str] = None,
        pincode: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        BIAP-compatible ONDC INIT stage - Initialize order with complex delivery structure
        
        Enhanced with:
        - BIAP validation (multiple BPP/Provider checks)
        - Proper city code mapping from pincode
        - BIAP context factory
        - Complex billing/delivery info structure matching BIAP patterns
        
        Args:
            session: User session (must be in SELECT stage)
            customer_name: Customer's full name
            delivery_address: Complete street address
            phone: Contact phone number
            email: Contact email
            payment_method: Payment method (cod, upi, card, netbanking)
            city: Delivery city (optional - extracted from SELECT stage if not provided)
            state: Delivery state (optional - extracted from SELECT stage if not provided)
            pincode: Delivery pincode (optional - extracted from SELECT stage if not provided)
            
        Returns:
            Order initialization status
        """
        # Validate session is in SELECT stage
        if session.checkout_state.stage != CheckoutStage.SELECT:
            return {
                'success': False,
                'message': ' Please select delivery location first.'
            }
        
        # Validate all required information
        missing = []
        if not customer_name: missing.append("customer name")
        if not delivery_address: missing.append("delivery address")
        if not phone: missing.append("phone number")
        if not email: missing.append("email")
        
        if missing:
            return {
                'success': False,
                'message': f' Missing: {", ".join(missing)}'
            }
        
        # Validate payment method - COD NOT SUPPORTED by Himira backend
        if config.payment.enable_cod_payments:
            valid_methods = ['razorpay', 'upi', 'card', 'netbanking', 'cod']  # COD enabled (for testing)
        else:
            valid_methods = ['razorpay', 'upi', 'card', 'netbanking']  # COD disabled (production)
            
        if payment_method.lower() not in valid_methods:
            return {
                'success': False,
                'message': f' Invalid payment method. COD not supported by Himira backend. Choose: {", ".join(valid_methods)}'
            }
            
        # Log payment method selection with mock indicators
        if config.payment.debug_logs:
            logger.info(f"[PAYMENT METHOD] Selected: {payment_method.lower()}")
            logger.info(f"[PAYMENT CONFIG] COD enabled: {config.payment.enable_cod_payments}")
            logger.info(f"[PAYMENT CONFIG] Mock mode: {config.payment.mock_mode}")
        
        try:
            logger.info(f"[CheckoutService] Starting INIT for customer: {customer_name}")
            
            # Step 1: Get enriched items for validation
            enriched_items = await self.product_enrichment.enrich_cart_items(
                session.cart.items, session.session_id
            )
            
            # Step 2: BIAP validation - check for multiple BPP/Provider items  
            validation_result = self.validation.validate_order_items(enriched_items, "init")
            if not validation_result.get('success'):
                return {
                    'success': False,
                    'message': f" {validation_result['error']['message']}"
                }
            
            # Step 3: Get delivery location from SELECT stage or parameters
            # Use provided parameters or fall back to stored delivery info
            final_city = city or session.checkout_state.delivery_info.city if session.checkout_state.delivery_info else "Bangalore"
            final_state = state or session.checkout_state.delivery_info.city if session.checkout_state.delivery_info else "Karnataka" 
            final_pincode = pincode or "560001"  # Should be captured from SELECT stage
            
            # Step 4: Get proper city code from pincode
            city_code = get_city_code_by_pincode(final_pincode)
            
            # Step 5: Create BIAP-compatible context
            context = self.context_factory.create({
                'action': 'init',
                'transaction_id': session.checkout_state.transaction_id,
                'city': city_code,
                'pincode': final_pincode
            })
            
            # Step 6: Get BPP info from validated items
            bpp_info = self.validation.get_order_bpp_info(enriched_items)
            if bpp_info:
                context['bpp_id'] = bpp_info['bpp_id']
                context['bpp_uri'] = bpp_info['bpp_uri']
            
            # Step 7: Create BIAP-compatible billing info structure
            billing_info = {
                'name': customer_name,
                'phone': phone,
                'email': email,
                'address': {
                    'name': customer_name,
                    'building': delivery_address,
                    'locality': "Area",  # Could be extracted from address
                    'city': final_city,
                    'state': final_state,
                    'country': "IND",
                    'area_code': final_pincode
                }
            }
            
            # Step 8: Create delivery info from billing info (BIAP pattern)
            from ..utils.city_code_mapping import create_delivery_info_from_billing_info
            delivery_info = create_delivery_info_from_billing_info(billing_info)
            
            # Step 9: Set session delivery and payment info
            session.checkout_state.delivery_info = DeliveryInfo(
                address=delivery_address,
                phone=phone,
                email=email,
                name=customer_name,
                city=final_city,
                pincode=final_pincode
            )
            session.checkout_state.payment_method = payment_method.lower()
            
            # Step 10: Transform enriched items to BIAP INIT format
            init_items = []
            fulfillment_ids = []
            for item in enriched_items:
                init_item = {
                    'id': item.local_id,
                    'quantity': {'count': item.quantity}
                }
                
                # Add fulfillment_id if available
                if item.fulfillment_id:
                    init_item['fulfillment_id'] = item.fulfillment_id
                    fulfillment_ids.append(item.fulfillment_id)
                
                # Add customisations if available
                if item.customisations:
                    init_item['tags'] = item.customisations
                
                init_items.append(init_item)
            
            # Step 11: Create BIAP-compatible INIT request
            init_data = {
                'context': context,
                'message': {
                    'order': {
                        'billing': billing_info,
                        'items': init_items,
                        'fulfillments': [{
                            'id': fulfillment_ids[0] if fulfillment_ids else "1",
                            'type': 'Delivery',
                            'end': {
                                'contact': {
                                    'phone': phone,
                                    'email': email
                                },
                                'location': {
                                    **delivery_info['location'],
                                    'address': {
                                        **delivery_info['location']['address'],
                                        'name': customer_name,
                                        'area_code': final_pincode
                                    }
                                }
                            },
                            'customer': {
                                'person': {
                                    'name': customer_name
                                }
                            }
                        }],
                        'payment': {
                            'type': 'ON-ORDER',  # BIAP Standard: Always ON-ORDER for online payments
                            'collected_by': 'BAP'  # BIAP Standard: Always BAP for ON-ORDER payments
                        }
                    }
                },
                'userId': session.session_id
            }
            
            # Step 12: Call BIAP INIT API - GUEST MODE
            # Guest mode: No authentication required for order initialization
            auth_token = getattr(session, 'auth_token', None)
            if not auth_token:
                logger.info("[CheckoutService] GUEST MODE - Proceeding without auth token")
                # For guest users, we'll use wil-api-key authentication only
                auth_token = None
            
            init_response = await self.buyer_app.initialize_order(init_data, auth_token=auth_token)
            
            logger.info(f"[CheckoutService] INIT API initial response received")
            logger.debug(f"[CheckoutService] INIT initial response: {json.dumps(init_response, indent=2) if init_response else 'None'}")
            
            # Extract message ID from response for polling
            message_id = None
            if init_response:
                if isinstance(init_response, dict):
                    message_id = init_response.get('messageId') or init_response.get('message_id')
                elif isinstance(init_response, list) and len(init_response) > 0:
                    message_id = init_response[0].get('messageId') or init_response[0].get('message_id')
            
            if not message_id:
                logger.error(f"[CheckoutService] No messageId in INIT response: {init_response}")
                return {
                    'success': False,
                    'message': ' Failed to initiate order initialization. No message ID received.'
                }
            
            logger.info(f"[CheckoutService] Polling for INIT response with messageId: {message_id}")
            
            # Poll for the actual INIT response
            result = await self._poll_for_response(
                poll_function=self.buyer_app.get_init_response,
                message_id=message_id,
                operation_name="INIT",
                max_attempts=15,
                initial_delay=2.0,
                auth_token=auth_token
            )
            
            logger.debug(f"[CheckoutService] Final INIT response after polling: {json.dumps(result, indent=2) if result else 'None'}")
            
            if result and result.get('success', True) and 'error' not in result:
                # Update session to INIT stage  
                session.checkout_state.stage = CheckoutStage.INIT
                session.add_to_history('initialize_order', {
                    'address': delivery_address,
                    'phone': phone,
                    'email': email,
                    'payment_method': payment_method
                })
                
                return {
                    'success': True,
                    'message': f' Order initialized successfully!',
                    'stage': 'init_completed',
                    'order_summary': {
                        'items': session.cart.total_items,
                        'total': session.cart.total_value,
                        'delivery': delivery_address,
                        'payment': payment_method.upper()
                    },
                    'init_data': result,
                    'next_step': 'confirm_order'
                }
            else:
                error_msg = result.get('message', 'Failed to initialize order') if result else 'INIT API failed'
                return {
                    'success': False,
                    'message': f' {error_msg}'
                }
                
        except Exception as e:
            logger.error(f"INIT operation failed: {e}")
            return {
                'success': False,
                'message': ' Failed to initialize order. Please try again.'
            }
    
    async def create_payment(self, session: Session, payment_method: str = 'razorpay') -> Dict[str, Any]:
        """
        MOCK PAYMENT CREATION - Create mock payment between INIT and CONFIRM
        
        This method simulates the Razorpay payment creation step that would normally
        happen between INIT and CONFIRM in the ONDC flow.
        
        Args:
            session: User session (must be in INIT stage)
            payment_method: Payment method (default: razorpay)
            
        Returns:
            Mock payment creation response with payment ID
        """
        # Validate session is in INIT stage
        if session.checkout_state.stage != CheckoutStage.INIT:
            return {
                'success': False,
                'message': f' Cannot create payment. Session is in {session.checkout_state.stage.value} stage, expected INIT.'
            }
        
        try:
            total_amount = session.cart.total_value
            
            # MOCK PAYMENT CREATION - Using values from Himira Order Postman Collection
            if config.payment.mock_mode:
                logger.info(f"[MOCK PAYMENT CREATION] Creating mock payment for amount: {total_amount}")
                
                # Create mock Razorpay order
                mock_order = mock_payment_service.create_mock_razorpay_order(
                    amount=total_amount,
                    currency="INR",
                    transaction_id=session.checkout_state.transaction_id
                )
                
                # Create mock payment
                mock_payment = mock_payment_service.create_mock_payment(
                    amount=total_amount,
                    currency="INR",
                    order_id=mock_order["id"]
                )
                
                if config.payment.debug_logs:
                    logger.info(f"[MOCK PAYMENT CREATION] Order ID: {mock_order['id']}")
                    logger.info(f"[MOCK PAYMENT CREATION] Payment ID: {mock_payment['razorpayPaymentId']}")
                    logger.info(f"[MOCK PAYMENT CREATION] Amount: {total_amount} INR")
                
                # Update session with payment information
                session.checkout_state.payment_id = mock_payment['razorpayPaymentId']
                
                return {
                    'success': True,
                    'message': f' [MOCK] Payment created successfully',
                    'data': {
                        'payment_id': mock_payment['razorpayPaymentId'],  # MOCK: From Postman
                        'order_id': mock_order['id'],  # MOCK: Generated
                        'amount': total_amount,
                        'currency': 'INR',
                        'status': 'created',
                        '_mock_indicators': {
                            'payment_mode': 'MOCK_TESTING',
                            'source': 'himira_postman_collection',
                            'payment_id': mock_payment['razorpayPaymentId']
                        }
                    },
                    'next_step': 'confirm_order'
                }
            else:
                # Real payment implementation would go here
                logger.warning("[REAL PAYMENT CREATION] Real payment mode not implemented yet")
                return {
                    'success': False,
                    'message': ' Real payment mode not implemented. Use mock mode for testing.'
                }
                
        except Exception as e:
            logger.error(f"[CheckoutService] Payment creation failed: {str(e)}")
            return {
                'success': False,
                'message': f' Payment creation failed: {str(e)}'
            }
    
    async def confirm_order(self, session: Session, payment_status: str = 'PENDING') -> Dict[str, Any]:
        """
        BIAP-compatible ONDC CONFIRM stage - Finalize the order with payment validation
        
        Enhanced with:
        - BIAP validation (multiple BPP/Provider checks)
        - Payment status validation
        - Proper BIAP context factory
        - Complete order structure matching BIAP patterns
        
        Args:
            session: User session (must be in INIT stage)
            payment_status: Payment status ('PENDING', 'PAID', 'FAILED')
            
        Returns:
            Final order confirmation with order ID
        """
        # Validate session is in INIT stage
        if session.checkout_state.stage != CheckoutStage.INIT:
            return {
                'success': False,
                'message': ' Please complete delivery and payment details first.'
            }
        
        # Validate required data is present
        if not session.checkout_state.delivery_info or not session.checkout_state.payment_method:
            return {
                'success': False,
                'message': ' Missing delivery or payment information.'
            }
        
        try:
            logger.info(f"[CheckoutService] Starting CONFIRM with payment status: {payment_status}")
            
            # Step 1: Get enriched items for validation
            enriched_items = await self.product_enrichment.enrich_cart_items(
                session.cart.items, session.session_id
            )
            
            # Step 2: BIAP validation - check for multiple BPP/Provider items
            validation_result = self.validation.validate_order_items(enriched_items, "confirm")
            if not validation_result.get('success'):
                return {
                    'success': False,
                    'message': f" {validation_result['error']['message']}"
                }
            
            # Step 3: Payment validation - ALL payments require completion (no COD support)
            payment_method = session.checkout_state.payment_method.lower()
            
            # Log payment validation with mock indicators
            if config.payment.debug_logs:
                logger.info(f"[PAYMENT VALIDATION] Method: {payment_method}")
                logger.info(f"[PAYMENT VALIDATION] Status: {payment_status}")
                logger.info(f"[PAYMENT VALIDATION] Mock mode: {config.payment.mock_mode}")
            
            # For mock mode, always accept PAID/PENDING status
            if config.payment.mock_mode:
                if payment_status.upper() not in ['PAID', 'CAPTURED', 'SUCCESS', 'PENDING']:
                    logger.warning(f"[MOCK PAYMENT] Invalid status: {payment_status}")
                    return {
                        'success': False,
                        'message': f' [MOCK] Payment not completed. Status: {payment_status}'
                    }
                logger.info(f"[MOCK PAYMENT] Status validated: {payment_status}")
            else:
                # For real payments, require completed status
                if payment_status.upper() not in ['PAID', 'CAPTURED', 'SUCCESS']:
                    return {
                        'success': False,
                        'message': f" Payment not completed. Status: {payment_status}. Please complete payment first."
                    }
            
            # Step 4: Create BIAP-compatible context  
            context = self.context_factory.create({
                'action': 'confirm',
                'transaction_id': session.checkout_state.transaction_id
            })
            
            # Step 5: Get BPP info from validated items
            bpp_info = self.validation.get_order_bpp_info(enriched_items)
            if bpp_info:
                context['bpp_id'] = bpp_info['bpp_id']
                context['bpp_uri'] = bpp_info['bpp_uri']
            
            # Step 6: Transform enriched items to BIAP CONFIRM format
            confirm_items = []
            for item in enriched_items:
                confirm_item = {
                    'id': item.local_id,
                    'quantity': {'count': item.quantity}
                }
                
                # Add fulfillment_id if available
                if item.fulfillment_id:
                    confirm_item['fulfillment_id'] = item.fulfillment_id
                
                confirm_items.append(confirm_item)
            
            # Step 7: Create payment object with MOCK values - TESTING ONLY
            total_amount = session.cart.total_value
            
            # MOCK PAYMENT IMPLEMENTATION - Using values from Himira Order Postman Collection
            if config.payment.mock_mode:
                logger.info(f"[MOCK PAYMENT] Creating mock payment for amount: {total_amount}")
                payment_obj = mock_payment_service.create_biap_payment_object(total_amount)
                
                if config.payment.debug_logs:
                    logger.info(f"[MOCK PAYMENT] Payment ID: {payment_obj['razorpayPaymentId']}")
                    logger.info(f"[MOCK PAYMENT] Settlement basis: {payment_obj['@ondc/org/settlement_basis']}")
                    logger.info(f"[MOCK PAYMENT] Settlement window: {payment_obj['@ondc/org/settlement_window']}")
            else:
                # Real payment implementation (disabled for now)
                logger.warning("[REAL PAYMENT] Real payment mode not implemented yet")
                payment_obj = {
                    'type': 'ON-ORDER',
                    'collected_by': 'BAP', 
                    'paid_amount': total_amount,
                    'paymentMethod': payment_method.upper(),
                    'status': payment_status.upper()
                }
            
            # Step 8: Create BIAP-compatible CONFIRM request
            confirm_data = {
                'context': context,
                'message': {
                    'order': {
                        'id': f"ORDER_{session.checkout_state.transaction_id[:8].upper()}",
                        'items': confirm_items,
                        'billing': {
                            'name': session.checkout_state.delivery_info.name,
                            'phone': session.checkout_state.delivery_info.phone,
                            'email': session.checkout_state.delivery_info.email,
                            'address': {
                                'name': session.checkout_state.delivery_info.name,
                                'building': session.checkout_state.delivery_info.address,
                                'locality': "Area",
                                'city': session.checkout_state.delivery_info.city,
                                'state': "Karnataka",  # Should be from delivery info
                                'country': "IND",
                                'area_code': session.checkout_state.delivery_info.pincode
                            }
                        },
                        'fulfillments': [{
                            'id': "1",
                            'type': 'Delivery',
                            'end': {
                                'contact': {
                                    'phone': session.checkout_state.delivery_info.phone,
                                    'email': session.checkout_state.delivery_info.email
                                },
                                'location': {
                                    'address': {
                                        'name': session.checkout_state.delivery_info.name,
                                        'building': session.checkout_state.delivery_info.address,
                                        'locality': "Area",
                                        'city': session.checkout_state.delivery_info.city,
                                        'state': "Karnataka",
                                        'country': "IND",
                                        'area_code': session.checkout_state.delivery_info.pincode
                                    }
                                }
                            },
                            'customer': {
                                'person': {
                                    'name': session.checkout_state.delivery_info.name
                                }
                            }
                        }],
                        'payment': payment_obj
                    }
                },
                'userId': session.session_id
            }
            
            # Step 9: Call BIAP CONFIRM API - MOCK MODE FOR GUEST
            # Guest mode: Allow mock confirmation without authentication
            auth_token = getattr(session, 'auth_token', None)
            if not auth_token:
                if config.payment.mock_mode:
                    logger.info("[CheckoutService] MOCK MODE - Confirming order without auth token")
                    # For mock mode, we'll simulate the confirmation
                    auth_token = None
                else:
                    logger.error("[CheckoutService] Real mode requires authentication for order confirmation")
                    return {
                        'success': False,
                        'message': ' Authentication required for real orders. Please login first.'
                    }
            
            # In mock mode, simulate the confirmation
            if config.payment.mock_mode and not auth_token:
                logger.info("[CheckoutService] MOCK CONFIRM - Simulating order confirmation")
                # Create mock confirmation response
                result = {
                    'success': True,
                    'order_id': f"MOCK_ORDER_{session.checkout_state.transaction_id[:8].upper()}",
                    'message': '[MOCK] Order confirmed successfully'
                }
            else:
                confirm_response = await self.buyer_app.confirm_order(confirm_data, auth_token=auth_token)
                
                logger.info(f"[CheckoutService] CONFIRM API initial response received")
                logger.debug(f"[CheckoutService] CONFIRM initial response: {json.dumps(confirm_response, indent=2) if confirm_response else 'None'}")
                
                # Extract message ID from response for polling
                message_id = None
                if confirm_response:
                    if isinstance(confirm_response, dict):
                        message_id = confirm_response.get('messageId') or confirm_response.get('message_id')
                    elif isinstance(confirm_response, list) and len(confirm_response) > 0:
                        message_id = confirm_response[0].get('messageId') or confirm_response[0].get('message_id')
                
                if not message_id:
                    logger.error(f"[CheckoutService] No messageId in CONFIRM response: {confirm_response}")
                    return {
                        'success': False,
                        'message': ' Failed to confirm order. No message ID received.'
                    }
                
                logger.info(f"[CheckoutService] Polling for CONFIRM response with messageId: {message_id}")
                
                # Poll for the actual CONFIRM response
                result = await self._poll_for_response(
                    poll_function=self.buyer_app.get_confirm_response,
                    message_id=message_id,
                    operation_name="CONFIRM",
                    max_attempts=15,
                    initial_delay=2.0,
                    auth_token=auth_token
                )
                
                logger.debug(f"[CheckoutService] Final CONFIRM response after polling: {json.dumps(result, indent=2) if result else 'None'}")
            
            if result and result.get('success', True) and 'error' not in result:
                # Extract order ID from response
                order_id = result.get('order_id') or self._generate_order_id()
                session.checkout_state.order_id = order_id
                
                # Update session to CONFIRMED stage
                session.checkout_state.stage = CheckoutStage.CONFIRMED
                session.add_to_history('confirm_order', {
                    'order_id': order_id,
                    'total': session.cart.total_value
                })
                
                # Save cart details before clearing
                cart_items = list(session.cart.items)
                total_value = session.cart.total_value
                delivery_address = session.checkout_state.delivery_info.address
                phone = session.checkout_state.delivery_info.phone
                
                # Clear cart after successful order
                session.cart.clear()
                
                return {
                    'success': True,
                    'message': f' Order confirmed successfully!',
                    'order_id': order_id,
                    'order_details': {
                        'order_id': order_id,
                        'items': len(cart_items),
                        'total': total_value,
                        'delivery_address': delivery_address,
                        'payment_method': session.checkout_state.payment_method.upper(),
                        'phone': phone
                    },
                    'confirm_data': result,
                    'next_actions': ['track_order', 'search_products']
                }
            else:
                error_msg = result.get('message', 'Failed to confirm order') if result else 'CONFIRM API failed'
                return {
                    'success': False,
                    'message': f' {error_msg}'
                }
                
        except Exception as e:
            logger.error(f"CONFIRM operation failed: {e}")
            return {
                'success': False,
                'message': ' Failed to confirm order. Please try again.'
            }
    
    async def set_delivery_info(self, session: Session, address: str, 
                               phone: str, email: str, **kwargs) -> Dict[str, Any]:
        """
        Store delivery information in session - Himira compliant
        
        Args:
            session: User session
            address: Full delivery address
            phone: Contact phone number
            email: Contact email
            **kwargs: Additional fields like city, state, pincode
            
        Returns:
            Success response with next steps
        """
        try:
            # Create DeliveryInfo object (note: DeliveryInfo doesn't have state field)
            delivery_info = DeliveryInfo(
                address=address,
                phone=phone,
                email=email,
                city=kwargs.get('city', ''),
                pincode=kwargs.get('pincode', ''),
                name=kwargs.get('name', kwargs.get('customer_name', ''))
            )
            
            # Store in session checkout state
            session.checkout_state.delivery_info = delivery_info
            
            # Also store state separately since DeliveryInfo doesn't have it
            if kwargs.get('state'):
                if not hasattr(session, 'delivery_state'):
                    session.delivery_state = kwargs.get('state')
            
            # Also store customer info if name provided
            if kwargs.get('name'):
                session.customer_info = {
                    'name': kwargs.get('name'),
                    'phone': phone,
                    'email': email,
                    'address': address
                }
            
            logger.info(f"[CheckoutService] Delivery info saved for session {session.session_id}")
            logger.debug(f"[CheckoutService] Delivery details: {delivery_info}")
            
            return {
                'success': True,
                'message': f""" **Delivery Details Saved!**
                
 **Address:** {address}
 **Phone:** {phone}
 **Email:** {email}

 **Next Step:** Getting delivery quotes for your location...""",
                'next_step': 'select_items_for_order',
                'delivery_info': {
                    'address': address,
                    'phone': phone,
                    'email': email,
                    'city': kwargs.get('city'),
                    'state': kwargs.get('state'),
                    'pincode': kwargs.get('pincode')
                }
            }
            
        except Exception as e:
            logger.error(f"[CheckoutService] Failed to set delivery info: {e}")
            return {
                'success': False,
                'message': f' Failed to save delivery details: {str(e)}'
            }
    
    def _generate_transaction_id(self) -> str:
        """Generate unique transaction ID"""
        return f"txn_{uuid.uuid4().hex[:12]}"
    
    def _generate_order_id(self) -> str:
        """Generate unique order ID"""
        return f"ORD{uuid.uuid4().hex[:8].upper()}"


# Singleton instance
_checkout_service: Optional[CheckoutService] = None


def get_checkout_service() -> CheckoutService:
    """Get singleton CheckoutService instance"""
    global _checkout_service
    if _checkout_service is None:
        _checkout_service = CheckoutService()
    return _checkout_service