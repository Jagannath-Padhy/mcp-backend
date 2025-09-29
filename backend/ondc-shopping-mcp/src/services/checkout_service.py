"""Checkout service for managing checkout flow"""

from typing import Dict, List, Optional, Any, Tuple
import logging
import uuid
import json
import asyncio
import time
import os

from ..models.session import Session, CheckoutState, CheckoutStage, DeliveryInfo
from ..models.ondc_payload_models import (
    create_select_payload, 
    create_init_payload,
    validate_select_payload,
    validate_init_payload,
    SELECT_VALIDATION_ERROR,
    INIT_VALIDATION_ERROR
)
from ..buyer_backend_client import BuyerBackendClient, get_buyer_backend_client
from ..data_models.biap_context_factory import get_biap_context_factory
from ..services.payment_mock_service import mock_payment_service
from ..services.cart_service import CartService
from ..services.biap_validation_service import BiapValidationService
from ..services.product_enrichment_service import get_product_enrichment_service
from ..utils.city_code_mapping import get_city_code_by_pincode
from ..utils.logger import get_logger
from ..config import config

# Using real provider data directly from cart items

logger = get_logger(__name__)


class CheckoutService:
    """BIAP-compatible consolidated ONDC checkout service with 3 optimized methods"""
    
    def __init__(self, buyer_backend_client: Optional[BuyerBackendClient] = None, cart_service: Optional[CartService] = None):
        """
        Initialize checkout service with BIAP-compatible services
        
        Args:
            buyer_backend_client: Client for backend API calls
            cart_service: Service for backend cart operations
        """
        self.buyer_app = buyer_backend_client or get_buyer_backend_client()
        self.context_factory = get_biap_context_factory()
        self.cart_service = cart_service
        self.validation = BiapValidationService()  # Initialize validation service
        self.product_enrichment = get_product_enrichment_service()  # Initialize product enrichment service
        logger.info("CheckoutService initialized with backend services")
    
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
                    
                    # ONDC format: check for error field (null = success, non-null = error)
                    error_field = response_data.get('error')
                    
                    if error_field is None and 'message' in response_data:
                        # ONDC success: error is null and message contains the data
                        logger.info(f"[Polling] ✅ {operation_name} completed successfully (ONDC format)")
                        return response_data
                    elif error_field is not None:
                        # ONDC error: error field contains error details
                        logger.error(f"[Polling] ❌ {operation_name} failed: {error_field}")
                        return None
                    elif 'status' in response_data:
                        # Fallback: check status field for backwards compatibility
                        status = response_data.get('status', '').upper()
                        if status == 'COMPLETED':
                            logger.info(f"[Polling] ✅ {operation_name} completed successfully (legacy status format)")
                            return response_data
                        elif status == 'FAILED':
                            logger.error(f"[Polling] ❌ {operation_name} failed: {response_data.get('message', 'Unknown error')}")
                            return None
                    
                    # Still processing - continue polling
                    logger.debug(f"[Polling] {operation_name} still processing (waiting for ONDC response)")
            
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
    
    async def _poll_multiple_message_ids(
        self,
        message_ids: List[str],
        operation_name: str = "MULTI_SELECT", 
        auth_token: Optional[str] = None,
        max_attempts: int = 10,
        initial_delay: float = 1.0
    ) -> List[Dict[str, Any]]:
        """
        Poll multiple message IDs concurrently and return aggregated results
        
        Args:
            message_ids: List of message IDs to poll
            operation_name: Name for logging purposes
            auth_token: Optional auth token
            max_attempts: Maximum polling attempts per message ID
            initial_delay: Initial delay between polls
            
        Returns:
            List of successful polling results
        """
        if not message_ids:
            logger.warning(f"[MultiPoll] No message IDs to poll for {operation_name}")
            return []
        
        logger.info(f"[MultiPoll] Starting parallel polling for {len(message_ids)} message IDs")
        
        # Create polling tasks for all message IDs
        polling_tasks = []
        for idx, message_id in enumerate(message_ids):
            logger.info(f"[MultiPoll] Creating polling task[{idx+1}] for messageId: {message_id}")
            
            # Create individual polling task
            task = self._poll_for_response(
                poll_function=self.buyer_app.get_select_response,
                message_id=message_id,
                operation_name=f"{operation_name}[{idx+1}]",
                max_attempts=max_attempts,
                initial_delay=initial_delay,
                auth_token=auth_token
            )
            polling_tasks.append(task)
        
        # Execute all polling tasks concurrently 
        logger.info(f"[MultiPoll] Executing {len(polling_tasks)} polling tasks concurrently...")
        try:
            # Use asyncio.gather to run all polls in parallel
            results = await asyncio.gather(*polling_tasks, return_exceptions=True)
            
            # Process results and filter successful responses
            successful_results = []
            failed_count = 0
            
            for idx, result in enumerate(results):
                message_id = message_ids[idx]
                
                if isinstance(result, Exception):
                    logger.error(f"[MultiPoll] Task[{idx+1}] failed with exception: {result}")
                    failed_count += 1
                elif result is None:
                    logger.warning(f"[MultiPoll] Task[{idx+1}] returned None (timeout/failed)")
                    failed_count += 1
                elif isinstance(result, dict) and result.get('error') is None:
                    logger.info(f"[MultiPoll] ✅ Task[{idx+1}] successful for messageId: {message_id}")
                    successful_results.append(result)
                else:
                    logger.warning(f"[MultiPoll] Task[{idx+1}] returned error: {result}")
                    failed_count += 1
            
            # Log summary
            total_tasks = len(polling_tasks)
            success_count = len(successful_results)
            logger.info(f"[MultiPoll] 📊 Results: {success_count}/{total_tasks} successful, {failed_count} failed")
            
            if successful_results:
                logger.info(f"[MultiPoll] ✅ {operation_name} completed with {success_count} quote(s)")
            else:
                logger.error(f"[MultiPoll] ❌ All {operation_name} polls failed")
                
            return successful_results
            
        except Exception as e:
            logger.error(f"[MultiPoll] Critical error during parallel polling: {e}", exc_info=True)
            return []
    
    def _aggregate_quote_data(self, quote_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate multiple on_select quote responses into unified quote data
        
        Args:
            quote_results: List of successful on_select responses
            
        Returns:
            Aggregated quote data with providers, items, and totals
        """
        logger.info(f"[QuoteAggregator] Starting aggregation of {len(quote_results)} quote(s)")
        
        aggregated_data = {
            'providers': [],
            'total_value': 0.0,
            'total_delivery': 0.0,
            'items': [],
            'fulfillments': [],
            'raw_quotes': quote_results
        }
        
        for idx, quote_result in enumerate(quote_results):
            try:
                logger.debug(f"[QuoteAggregator] Processing quote[{idx+1}]...")
                
                # Extract quote data from ONDC response format
                message = quote_result.get('message', {})
                quote = message.get('quote', {})
                
                if not quote:
                    logger.warning(f"[QuoteAggregator] Quote[{idx+1}] missing quote data")
                    continue
                
                # Extract provider information
                provider_info = quote.get('provider', {})
                provider_id = provider_info.get('id', f'unknown_provider_{idx+1}')
                provider_locations = provider_info.get('locations', [])
                
                # Extract quote pricing
                quote_price = quote.get('quote', {})
                price_info = quote_price.get('price', {})
                quote_total = float(price_info.get('value', 0))
                currency = price_info.get('currency', 'INR')
                
                # Extract item breakdown
                items_breakdown = quote_price.get('breakup', [])
                provider_items = []
                provider_delivery = 0.0
                
                for item in items_breakdown:
                    item_title = item.get('title', 'Unknown Item')
                    title_type = item.get('@ondc/org/title_type', 'item')
                    item_price = float(item.get('price', {}).get('value', 0))
                    
                    if title_type == 'item':
                        # Regular item
                        item_id = item.get('@ondc/org/item_id', '')
                        item_quantity = item.get('@ondc/org/item_quantity', {}).get('count', 1)
                        
                        provider_items.append({
                            'id': item_id,
                            'title': item_title,
                            'quantity': item_quantity,
                            'price': item_price,
                            'currency': currency
                        })
                    elif title_type == 'delivery':
                        # Delivery charges
                        provider_delivery += item_price
                
                # Extract fulfillments
                fulfillments = quote.get('fulfillments', [])
                for fulfillment in fulfillments:
                    fulfillment_data = {
                        'id': fulfillment.get('id', ''),
                        'provider_name': fulfillment.get('@ondc/org/provider_name', 'Unknown Provider'),
                        'type': fulfillment.get('type', 'Delivery'),
                        'category': fulfillment.get('@ondc/org/category', 'Standard'),
                        'tat': fulfillment.get('@ondc/org/TAT', 'Unknown'),
                        'tracking': fulfillment.get('tracking', False)
                    }
                    aggregated_data['fulfillments'].append(fulfillment_data)
                
                # Create provider summary
                provider_summary = {
                    'id': provider_id,
                    'name': fulfillments[0].get('@ondc/org/provider_name', 'Unknown Provider') if fulfillments else 'Unknown Provider',
                    'locations': provider_locations,
                    'items': provider_items,
                    'total_value': quote_total - provider_delivery,
                    'delivery_charges': provider_delivery,
                    'currency': currency
                }
                
                aggregated_data['providers'].append(provider_summary)
                aggregated_data['total_value'] += (quote_total - provider_delivery)
                aggregated_data['total_delivery'] += provider_delivery
                aggregated_data['items'].extend(provider_items)
                
                logger.info(f"[QuoteAggregator] Quote[{idx+1}] processed: {provider_summary['name']} - ₹{quote_total} ({len(provider_items)} items)")
                
            except Exception as e:
                logger.error(f"[QuoteAggregator] Error processing quote[{idx+1}]: {e}")
                logger.error(f"[QuoteAggregator] Quote data: {json.dumps(quote_result, indent=2)}")
                continue
        
        # Final summary
        total_providers = len(aggregated_data['providers'])
        total_items = len(aggregated_data['items'])
        grand_total = aggregated_data['total_value'] + aggregated_data['total_delivery']
        
        logger.info(f"[QuoteAggregator] ✅ Aggregation complete:")
        logger.info(f"  - Providers: {total_providers}")
        logger.info(f"  - Items: {total_items}")
        logger.info(f"  - Items Total: ₹{aggregated_data['total_value']}")
        logger.info(f"  - Delivery Total: ₹{aggregated_data['total_delivery']}")
        logger.info(f"  - Grand Total: ₹{grand_total}")
        
        return aggregated_data
    
    def _group_cart_items_by_provider(self, cart_items: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Group cart items by provider_id and domain for multi-provider SELECT
        
        Args:
            cart_items: List of cart items from backend
            
        Returns:
            Dictionary mapping group_key to {provider_id, domain, items, readable_name}
        """
        logger.info(f"[ProviderGrouping] Grouping {len(cart_items)} cart items by provider and domain...")
        
        provider_groups = {}
        
        for idx, item in enumerate(cart_items):
            try:
                # Extract provider_id - this is the key field for grouping
                provider_id = item.get('provider_id', '')
                
                if not provider_id:
                    logger.warning(f"[ProviderGrouping] Item[{idx}] missing provider_id: {json.dumps(item, indent=2)}")
                    # Use fallback grouping based on id pattern
                    full_id = item.get('id', '')
                    if '_ONDC:' in full_id:
                        # Extract provider part from full ONDC ID
                        parts = full_id.split('_')
                        if len(parts) >= 3:
                            provider_id = f"{parts[0]}_ONDC:{parts[1].split(':')[1]}"
                    
                    if not provider_id:
                        provider_id = 'unknown_provider'
                        logger.warning(f"[ProviderGrouping] Using fallback provider_id for item[{idx}]")
                
                # Extract domain from real cart item data
                domain = 'ONDC:RET10'  # Default fallback
                
                # Try to get domain from item.domain (real cart data)
                nested_item = item.get('item', {})
                if nested_item and nested_item.get('domain'):
                    domain = nested_item['domain']
                    logger.debug(f"[ProviderGrouping] Using real item.domain: {domain}")
                elif '_ONDC:' in provider_id:
                    # Fallback: extract from provider_id if item.domain not available
                    try:
                        domain_part = provider_id.split('_ONDC:')[1].split('_')[0]
                        domain = f'ONDC:{domain_part}'
                        logger.debug(f"[ProviderGrouping] Fallback domain from provider_id: {domain}")
                    except Exception as domain_error:
                        logger.warning(f"[ProviderGrouping] Failed to extract domain from {provider_id}: {domain_error}")
                
                # Create unique group key (provider_id + domain)
                group_key = f"{provider_id}_{domain}"
                
                # Create provider group if it doesn't exist
                if group_key not in provider_groups:
                    # Extract readable provider name
                    if 'himira' in provider_id.lower():
                        readable_name = f'Himira Store ({domain})'
                    else:
                        # Extract domain from provider_id for name
                        base_name = provider_id.split('_')[0].split('.')[0] if '_' in provider_id else provider_id
                        readable_name = f'{base_name.title()} ({domain})'
                    
                    provider_groups[group_key] = {
                        'provider_id': provider_id,
                        'domain': domain, 
                        'items': [],
                        'readable_name': readable_name
                    }
                    logger.debug(f"[ProviderGrouping] Created new provider group: {readable_name}")
                
                # Add item to provider group
                provider_groups[group_key]['items'].append(item)
                logger.debug(f"[ProviderGrouping] Added item[{idx}] to group: {provider_groups[group_key]['readable_name']}")
                
            except Exception as e:
                logger.error(f"[ProviderGrouping] Error processing item[{idx}]: {e}")
                logger.error(f"[ProviderGrouping] Item data: {json.dumps(item, indent=2)}")
                continue
        
        # Log grouping results
        total_groups = len(provider_groups)
        logger.info(f"[ProviderGrouping] ✅ Grouped items into {total_groups} provider group(s):")
        
        for group_key, group_data in provider_groups.items():
            logger.info(f"  - {group_data['readable_name']}: {len(group_data['items'])} item(s)")
            logger.debug(f"    Provider ID: {group_data['provider_id']}")
            logger.debug(f"    Domain: {group_data['domain']}")
        
        return provider_groups
    
    async def select_items_for_order(
        self, 
        session: Session, 
        delivery_city: str,
        delivery_state: str,
        delivery_pincode: str,
        delivery_gps: str
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
        # DEBUG: Add logging to track SELECT execution
        logger.info(f"[DEBUG SELECT] Step 1: Starting SELECT operation for session {session.session_id}")
        
        # Validate cart using backend cart service
        logger.info(f"[DEBUG SELECT] Step 2: Checking cart summary...")
        cart_summary = await self.cart_service.get_cart_summary(session)
        logger.info(f"[DEBUG SELECT] Step 2 Result: cart_summary = {cart_summary}")
        if cart_summary['is_empty']:
            logger.info(f"[DEBUG SELECT] Step 2 Failed: Cart is empty")
            return {
                'success': False,
                'message': ' Cart is empty. Please add items first.'
            }
        logger.info(f"[DEBUG SELECT] Step 2 Passed: Cart has {cart_summary.get('total_items', 0)} items")
        
        # Validate delivery location
        logger.info(f"[DEBUG SELECT] Step 3: Validating delivery location...")
        logger.info(f"[DEBUG SELECT] Step 3 Params: city={delivery_city}, state={delivery_state}, pincode={delivery_pincode}")
        if not all([delivery_city, delivery_state, delivery_pincode]):
            logger.info(f"[DEBUG SELECT] Step 3 Failed: Missing delivery location")
            return {
                'success': False,
                'message': ' Missing delivery location. Please provide city, state, and pincode.'
            }
        logger.info(f"[DEBUG SELECT] Step 3 Passed: All delivery location params provided")
        
        # Generate transaction ID for this checkout session
        session.checkout_state.transaction_id = self._generate_transaction_id()
        
        try:
            # Get cart items from backend
            logger.info(f"[DEBUG SELECT] Step 4: Getting cart items from backend...")
            success, cart_display, cart_items = await self.cart_service.view_cart(session)
            logger.info(f"[DEBUG SELECT] Step 4 Result: success={success}, items_count={len(cart_items) if cart_items else 0}")
            if not success:
                logger.info(f"[DEBUG SELECT] Step 4 Failed: {cart_display}")
                return {
                    'success': False,
                    'message': f' Failed to get cart items: {cart_display}'
                }
            logger.info(f"[DEBUG SELECT] Step 4 Passed: Retrieved {len(cart_items)} cart items")
            
            # DEBUG: Check what cart data is actually received in checkout service
            logger.error(f"[CHECKOUT DEBUG] Cart items received from cart_service: {len(cart_items)} items")
            for debug_idx, debug_item in enumerate(cart_items):
                logger.error(f"[CHECKOUT DEBUG] Item[{debug_idx}] keys: {list(debug_item.keys()) if isinstance(debug_item, dict) else 'Not a dict'}")
                if isinstance(debug_item, dict) and 'item' in debug_item:
                    nested_item = debug_item.get('item', {})
                    logger.error(f"[CHECKOUT DEBUG] Item[{debug_idx}] nested item keys: {list(nested_item.keys()) if isinstance(nested_item, dict) else 'Nested not a dict'}")
                    logger.error(f"[CHECKOUT DEBUG] Item[{debug_idx}] nested local_id: {nested_item.get('local_id')}")
                else:
                    logger.error(f"[CHECKOUT DEBUG] Item[{debug_idx}] missing 'item' key or not a dict")
            
            logger.info(f"[CheckoutService] Starting SELECT with {len(cart_items)} items")
            logger.debug(f"[CheckoutService] Delivery location: {delivery_city}, {delivery_state}, {delivery_pincode}")
            logger.debug(f"[CheckoutService] Cart items: {[{'name': item.get('name'), 'id': item.get('id')} for item in cart_items]}")
            
            # Step 1: Debug cart service response structure
            logger.error("[EXECUTION CHECKPOINT] Starting Step 1: Debugging cart service response structure...")
            logger.info("[CheckoutService] Step 1: Debugging cart service response structure...")
            logger.info(f"[CheckoutService] Raw cart items count: {len(cart_items)}")
            
            try:
                # CRITICAL DEBUG: Log exact structure of cart items
                for idx, item in enumerate(cart_items):
                    logger.info(f"[CART DEBUG] Item[{idx}] structure:")
                    logger.info(f"  Keys: {list(item.keys()) if isinstance(item, dict) else 'Not a dict'}")
                    logger.info(f"  item_id: {item.get('item_id')}")
                    logger.info(f"  id: {item.get('id')}")
                    logger.info(f"  count: {item.get('count')}")
                    logger.info(f"  Full item: {json.dumps(item, indent=2) if isinstance(item, dict) else str(item)}")
                logger.error("[EXECUTION CHECKPOINT] Step 1 completed successfully")
            except Exception as e:
                logger.error(f"[EXECUTION CHECKPOINT] Step 1 FAILED with exception: {e}")
                raise e
            
            # Step 2: BIAP validation - check for multiple BPP/Provider items using raw data
            logger.error("[EXECUTION CHECKPOINT] Starting Step 2: BIAP validation...")
            logger.info("[CheckoutService] Step 2: Validating order items for BIAP compliance...")
            
            try:
                validation_result = self.validation.validate_order_items(cart_items, "select")
                logger.debug(f"[CheckoutService] Validation result: {validation_result}")
                if not validation_result.get('success'):
                    logger.error(f"[CheckoutService] Validation failed: {validation_result.get('error', {})}")
                    return {
                        'success': False,
                        'message': f" {validation_result['error']['message']}"
                    }
                logger.error("[EXECUTION CHECKPOINT] Step 2 completed successfully")
            except Exception as e:
                logger.error(f"[EXECUTION CHECKPOINT] Step 2 FAILED with exception: {e}")
                raise e
            
            # Step 3: Get proper city code from pincode
            logger.error("[EXECUTION CHECKPOINT] Starting Step 3: Getting city code...")
            logger.info(f"[CheckoutService] Step 3: Getting city code for pincode {delivery_pincode}...")
            
            try:
                city_code = get_city_code_by_pincode(delivery_pincode)
                logger.debug(f"[CheckoutService] City code: {city_code}")
                logger.error("[EXECUTION CHECKPOINT] Step 3 completed successfully")
            except Exception as e:
                logger.error(f"[EXECUTION CHECKPOINT] Step 3 FAILED with exception: {e}")
                raise e
            
            # Step 4: Create SELECT context (automatically simplified for SELECT operations)
            logger.error("[EXECUTION CHECKPOINT] Starting Step 4: Creating SELECT context...")
            logger.info("[CheckoutService] Step 4: Creating SELECT context...")
            
            try:
                context = self.context_factory.create({
                    'action': 'select',
                    'transaction_id': session.checkout_state.transaction_id,
                    'city': delivery_pincode,  #  Use pincode directly as per backend format
                    'pincode': delivery_pincode
                })
                logger.debug(f"[CheckoutService] SELECT context created: {json.dumps(context, indent=2)}")
                logger.error("[EXECUTION CHECKPOINT] Step 4 completed successfully")
            except Exception as e:
                logger.error(f"[EXECUTION CHECKPOINT] Step 4 FAILED with exception: {e}")
                raise e
            
            # Step 5: Get BPP info from validated items (for logging only)
            logger.error("[EXECUTION CHECKPOINT] Starting Step 5: Getting BPP info...")
            logger.info("[CheckoutService] Step 5: Getting BPP info from validated items...")
            
            try:
                bpp_info = self.validation.get_order_bpp_info(cart_items)
                logger.debug(f"[CheckoutService] BPP info: {bpp_info}")
                if bpp_info:
                    # Don't add to context - keep context clean like working frontend
                    logger.info(f"[CheckoutService] Using BPP: {bpp_info['bpp_id']} at {bpp_info['bpp_uri']}")
                logger.error("[EXECUTION CHECKPOINT] Step 5 completed successfully")
            except Exception as e:
                logger.error(f"[EXECUTION CHECKPOINT] Step 5 FAILED with exception: {e}")
                raise e
            
            # Step 6: Group cart items by provider and domain for multi-provider SELECT
            logger.info("[CheckoutService] Step 6: Grouping cart items by provider and domain...")
            provider_groups = self._group_cart_items_by_provider(cart_items)
            
            if not provider_groups:
                return {
                    'success': False,
                    'message': '❌ Failed to group cart items by provider. No valid items found.'
                }
            
            # Step 7: Build separate SELECT requests for each provider group
            logger.info(f"[CheckoutService] Step 7: Building {len(provider_groups)} SELECT request(s)...")
            select_requests = []
            
            for group_key, group_data in provider_groups.items():
                provider_id = group_data['provider_id']
                domain = group_data['domain']
                group_items = group_data['items']
                readable_name = group_data['readable_name']
                
                logger.info(f"[CheckoutService] Building SELECT request for: {readable_name}")
                logger.debug(f"  - Provider ID: {provider_id}")
                logger.debug(f"  - Domain: {domain}")
                logger.debug(f"  - Items: {len(group_items)}")
                
                # Use user's delivery pincode as context.city (not seller's area_code)
                area_code = delivery_pincode  # User's delivery location
                logger.info(f"[CheckoutService] Using user's delivery pincode as context.city: {area_code}")
                
                # Create context for this provider group using user input and real cart data
                group_context = self.context_factory.create({
                    'action': 'select',
                    'transaction_id': session.checkout_state.transaction_id,
                    'city': area_code,  # User's delivery pincode
                    'domain': domain   # Real domain from cart data
                })
                
                # Transform items for this provider group
                select_items = []
                for idx, item in enumerate(group_items):
                    try:
                        # Extract item data from backend cart format
                        nested_item = item.get('item', {})
                        local_id_value = nested_item.get('local_id') or item.get('item_id')
                        full_id_value = nested_item.get('id') or item.get('id')
                        
                        # Use fallback if nested data missing
                        if not local_id_value and not full_id_value:
                            local_id_value = item.get('item_id')
                            full_id_value = item.get('id')
                        
                        if not local_id_value or not full_id_value:
                            logger.error(f"[SELECT] Item missing required IDs: {json.dumps(item, indent=2)}")
                            continue
                            
                        select_item = {
                            'id': full_id_value,
                            'local_id': local_id_value,
                            'customisationState': {},
                            'quantity': {'count': item.get('count', 1)},
                            'customisations': None,
                            'hasCustomisations': False
                        }
                        
                        # Use complete provider object from real cart data
                        provider_data = nested_item.get('provider')
                        if provider_data:
                            # Use the complete provider structure from backend cart data
                            select_item['provider'] = provider_data
                            logger.debug(f"[SELECT] Using real provider data for item[{idx}]: {provider_data.get('id', 'no-id')}")
                        else:
                            # Log warning if provider data is missing
                            logger.warning(f"[SELECT] Item[{idx}] missing provider data in cart response: {json.dumps(nested_item, indent=2)}")
                            
                            # Try fallback to root-level provider_id
                            root_provider_id = item.get('provider_id')
                            if root_provider_id:
                                logger.info(f"[SELECT] Using fallback provider_id: {root_provider_id}")
                                select_item['provider'] = {
                                    'id': root_provider_id,
                                    'local_id': root_provider_id.split('_')[-1] if '_' in root_provider_id else root_provider_id
                                }
                        
                        select_items.append(select_item)
                        logger.debug(f"    Item[{idx}]: {local_id_value}")
                        
                    except Exception as e:
                        logger.error(f"[SELECT] Error processing item for {readable_name}: {e}")
                        continue
                
                if not select_items:
                    logger.warning(f"[SELECT] No valid items for {readable_name}, skipping...")
                    continue
                
                # Create fulfillments for this request using user's delivery location
                fulfillments = [{
                    'end': {
                        'location': {
                            'gps': delivery_gps,  # User's GPS coordinates
                            'address': {
                                'area_code': area_code  # User's delivery pincode
                            }
                        }
                    }
                }]
                
                # Build SELECT request for this provider group using model
                group_request = create_select_payload(
                    context=group_context,
                    cart_items=select_items,
                    fulfillments=fulfillments,
                    user_id=session.user_id,
                    device_id=session.device_id
                )
                
                # Validate SELECT payload structure
                if not validate_select_payload(group_request):
                    logger.error(f"[SELECT] {SELECT_VALIDATION_ERROR}")
                    continue
                
                select_requests.append(group_request)
                logger.info(f"[CheckoutService] ✅ SELECT request built for {readable_name}: {len(select_items)} item(s)")
            
            if not select_requests:
                return {
                    'success': False,
                    'message': '❌ Failed to build any valid SELECT requests. Cart data may be corrupted.'
                }
            
            # Step 8: Log multi-provider SELECT request summary
            logger.info("[CheckoutService] Step 8: Multi-Provider SELECT API Summary:")
            total_items = sum(len(req['message']['cart']['items']) for req in select_requests)
            logger.info(f"  - Provider Groups: {len(select_requests)}")
            logger.info(f"  - Total Items: {total_items}")
            logger.info(f"  - Delivery Location: {delivery_city}, {delivery_state} {delivery_pincode}")
            
            for idx, req in enumerate(select_requests):
                domain = req['context']['domain']
                items_count = len(req['message']['cart']['items'])
                logger.info(f"    Group[{idx+1}]: {domain} - {items_count} item(s)")
            
            # CRITICAL DEBUG: Log multi-provider SELECT request structure
            logger.error("=" * 80)
            logger.error("[CRITICAL DEBUG] MULTI-PROVIDER SELECT REQUESTS:")
            logger.error(f"Sending {len(select_requests)} separate provider requests:")
            for idx, req in enumerate(select_requests):
                logger.error(f"Request[{idx+1}] Domain: {req['context']['domain']}")
                logger.error(f"Request[{idx+1}] Items: {len(req['message']['cart']['items'])}")
            logger.error("=" * 80)
            
            # GUEST MODE: SELECT API call without authentication
            auth_token = getattr(session, 'auth_token', None)
            if not auth_token:
                logger.info("[CheckoutService] GUEST MODE - Calling multi-provider SELECT without auth token")
                auth_token = None
            else:
                logger.info("[CheckoutService] Using auth token for multi-provider SELECT request")
            
            # Call BIAP SELECT API with multi-provider request format
            # This matches your working example: array of separate provider requests
            logger.info(f"[CheckoutService] Sending multi-provider SELECT: {len(select_requests)} provider groups")
            select_response = await self.buyer_app.select_items(select_requests, auth_token=auth_token)
            
            logger.info(f"[CheckoutService] SELECT API initial response received")
            logger.debug(f"[CheckoutService] SELECT initial response: {json.dumps(select_response, indent=2) if select_response else 'None'}")
            
            # Extract ALL message IDs from response for parallel polling
            # ONDC SELECT returns array of responses, each with context.message_id
            # We need to poll ALL message_ids to get complete quote data
            message_ids = []
            if select_response:
                if isinstance(select_response, list):
                    # Expected ONDC array response format
                    for idx, item in enumerate(select_response):
                        if isinstance(item, dict):
                            context = item.get('context', {})
                            if isinstance(context, dict):
                                msg_id = context.get('message_id')
                                if msg_id:
                                    message_ids.append(msg_id)
                                    logger.info(f"[CheckoutService] Extracted messageId[{idx}]: {msg_id}")
                                else:
                                    logger.warning(f"[CheckoutService] Item[{idx}] missing message_id in context")
                            else:
                                logger.warning(f"[CheckoutService] Item[{idx}] missing context")
                        else:
                            logger.warning(f"[CheckoutService] Item[{idx}] is not a dict: {type(item)}")
                elif isinstance(select_response, dict):
                    # Fallback for single object responses (backwards compatibility)
                    context = select_response.get('context', {})
                    if isinstance(context, dict):
                        msg_id = context.get('message_id')
                        if msg_id:
                            message_ids.append(msg_id)
                            logger.info(f"[CheckoutService] Extracted single messageId: {msg_id}")
                        
            # Log results of message ID extraction
            if message_ids:
                logger.info(f"[CheckoutService] ✅ SELECT successful - found {len(message_ids)} message IDs to poll")
                for idx, msg_id in enumerate(message_ids):
                    logger.info(f"  MessageId[{idx+1}]: {msg_id}")
            else:
                logger.error(f"[CheckoutService] ❌ No message IDs found in SELECT response:")
                logger.error(f"  Response type: {type(select_response)}")
                if isinstance(select_response, list):
                    logger.error(f"  Array length: {len(select_response)}")
                    for idx, item in enumerate(select_response):
                        logger.error(f"  Item[{idx}] type: {type(item)}")
                        if isinstance(item, dict):
                            logger.error(f"  Item[{idx}] keys: {list(item.keys())}")
                elif isinstance(select_response, dict):
                    logger.error(f"  Dict keys: {list(select_response.keys())}")
            
            if not message_ids:
                logger.error(f"[CheckoutService] No messageId in SELECT response: {select_response}")
                return {
                    'success': False,
                    'message': '❌ Failed to initiate SELECT request. No message ID received.'
                }
            
            # Poll ALL message IDs concurrently for complete quote data
            logger.info(f"[CheckoutService] Starting parallel polling for {len(message_ids)} message IDs...")
            
            quote_results = await self._poll_multiple_message_ids(
                message_ids=message_ids,
                operation_name="SELECT",
                auth_token=auth_token,
                max_attempts=10,  # 10 seconds polling per message_id as requested
                initial_delay=1.0
            )
            
            logger.info(f"[CheckoutService] Parallel polling completed - received {len(quote_results)} quote(s)")
            if quote_results:
                logger.debug(f"[CheckoutService] All quote results: {json.dumps(quote_results, indent=2)}")
            
            if quote_results:
                # Aggregate quote data from all successful polling results
                aggregated_quotes = self._aggregate_quote_data(quote_results)
                
                # CRITICAL: Extract transaction_id from SELECT response for use in INIT
                # The transaction_id comes from any quote result's context.transaction_id
                if quote_results and len(quote_results) > 0:
                    first_quote = quote_results[0]
                    if isinstance(first_quote, dict) and 'context' in first_quote:
                        select_transaction_id = first_quote['context'].get('transaction_id')
                        if select_transaction_id:
                            logger.info(f"[CheckoutService] 🔍 TRANSACTION ID FLOW: Extracted from SELECT response: {select_transaction_id}")
                            # Update session with the transaction_id from SELECT response (overwrites generated one)
                            session.checkout_state.transaction_id = select_transaction_id
                            logger.info(f"[CheckoutService] ✅ TRANSACTION ID FLOW: Session updated with SELECT transaction_id")
                        else:
                            logger.warning(f"[CheckoutService] ⚠️ TRANSACTION ID: No transaction_id found in SELECT response context")
                    else:
                        logger.warning(f"[CheckoutService] ⚠️ TRANSACTION ID: SELECT response missing context structure")
                
                # Update session to SELECT stage with all quote data
                session.checkout_state.stage = CheckoutStage.SELECT
                session.add_to_history('select_items_for_order', {
                    'city': delivery_city,
                    'state': delivery_state, 
                    'pincode': delivery_pincode,
                    'items_count': len(cart_items),
                    'quotes_received': len(quote_results),
                    'transaction_id': session.checkout_state.transaction_id  # Log the final transaction_id used
                })
                
                # Enhanced user-friendly message with complete quote breakdown
                total_value = aggregated_quotes['total_value']
                delivery_charges = aggregated_quotes['total_delivery']
                grand_total = total_value + delivery_charges
                
                # Create detailed message with all providers and items
                providers_info = []
                for provider_data in aggregated_quotes['providers']:
                    provider_name = provider_data['name']
                    provider_total = provider_data['total_value']
                    item_count = len(provider_data['items'])
                    providers_info.append(f"**{provider_name}**: {item_count} item(s) - ₹{provider_total}")
                
                providers_text = "\n".join(providers_info)
                
                success_message = f"""✅ **Delivery available in {delivery_city}!**

**Quote Summary:**
{providers_text}

**Delivery Charges:** ₹{delivery_charges}
**Grand Total:** ₹{grand_total}

Ready to proceed with order initialization!"""
                
                logger.info(f"[CheckoutService] ✅ SELECT successful - {len(quote_results)} quote(s) received")
                logger.info(f"[CheckoutService] Total value: ₹{total_value}, Delivery: ₹{delivery_charges}, Grand total: ₹{grand_total}")
                
                return {
                    'success': True,
                    'message': success_message,
                    'stage': 'select_completed',
                    'quote_data': aggregated_quotes,
                    'total_value': total_value,
                    'delivery_charges': delivery_charges,
                    'grand_total': grand_total,
                    'providers_count': len(aggregated_quotes['providers']),
                    'next_step': 'initialize_order_with_customer_details'
                }
            else:
                logger.error(f"[CheckoutService] SELECT polling failed - no successful quotes received")
                return {
                    'success': False,
                    'message': f'❌ Failed to get delivery quotes from ONDC network. Please try again.'
                }
                
        except Exception as e:
            logger.error(f"[DEBUG SELECT] EXCEPTION CAUGHT: {type(e).__name__}: {str(e)}", exc_info=True)
            logger.error(f"[DEBUG SELECT] Exception details: {repr(e)}")
            logger.error(f"[CheckoutService] SELECT operation failed with exception: {e}", exc_info=True)
            return {
                'success': False,
                'message': f' Failed to get delivery options. Exception: {str(e)}'
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
        pincode: Optional[str] = None,
        delivery_gps: Optional[str] = None
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
            logger.info(f"[INIT] 🚀 Starting INIT for customer: {customer_name}")
            
            # Step 0: Get GPS coordinates based on pincode (or use provided city/state)
            final_city = city or session.checkout_state.delivery_info.city if session.checkout_state.delivery_info else "Bangalore"
            final_state = state or session.checkout_state.delivery_info.city if session.checkout_state.delivery_info else "Karnataka" 
            final_pincode = pincode or "560001"  # Should be captured from SELECT stage
            
            # Use user-provided GPS coordinates directly (no validation)
            if not delivery_gps:
                return {
                    'success': False,
                    'message': '📍 GPS coordinates required for delivery. Please provide coordinates in "latitude,longitude" format (e.g. "12.9716,77.5946")'
                }
            
            # Parse GPS coordinates from user input
            try:
                lat_str, lng_str = delivery_gps.split(',', 1)
                gps_coordinates = {
                    'lat': lat_str.strip(),
                    'lng': lng_str.strip()
                }
            except ValueError:
                return {
                    'success': False,
                    'message': f'❌ Invalid GPS format: "{delivery_gps}". Please use format: "latitude,longitude" (e.g. "12.9716,77.5946")'
                }
            
            logger.info(f"[INIT] 📍 Location: {final_city}, {final_state} {final_pincode}")
            logger.info(f"[INIT] 🌍 GPS Coordinates from user: {gps_coordinates['lat']}, {gps_coordinates['lng']}")
            
            # Step 1: Get enriched items for validation
            # Get cart items from backend for INIT/CONFIRM
            success, cart_display, cart_items = await self.cart_service.view_cart(session)
            if not success:
                return {
                    'success': False,
                    'message': f' Failed to get cart items: {cart_display}'
                }
            
            enriched_items = await self.product_enrichment.enrich_cart_items(
                cart_items, session.session_id
            )
            
            # Get cart summary for order total calculations
            cart_summary = await self.cart_service.get_cart_summary(session)
            
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
            
            # Step 10: Transform cart items to proper INIT format (matching working curl)
            init_items = []
            fulfillment_ids = []
            
            logger.info(f"[INIT] Mapping {len(cart_items)} cart items to INIT format")
            
            for cart_item in cart_items:
                # Use the actual cart item structure from backend API
                if isinstance(cart_item, dict):
                    # Extract fields from cart item response structure
                    item_data = cart_item.get('item', {})
                    
                    init_item = {
                        'id': cart_item.get('id'),  # Full ONDC item ID  
                        'local_id': cart_item.get('item_id'),  # Local item ID
                        'tags': item_data.get('tags', []),  # ONDC tags from item data
                        'fulfillment_id': 'Fulfillment1',  # Standard fulfillment ID
                        'quantity': {
                            'count': cart_item.get('count', 1)
                        },
                        'provider': {
                            'id': cart_item.get('provider_id'),
                            'local_id': item_data.get('provider', {}).get('local_id'),
                            'locations': item_data.get('provider', {}).get('locations', [])
                        },
                        'customisations': cart_item.get('customisations'),
                        'hasCustomisations': cart_item.get('hasCustomisations', False),
                        'userId': ''  # Empty userId as in working curl
                    }
                    
                    # Log the mapping for debugging
                    logger.debug(f"[INIT] Mapped cart item: {cart_item.get('item_id')} -> {init_item['id']}")
                    
                    init_items.append(init_item)
                    fulfillment_ids.append('Fulfillment1')
                else:
                    # Fallback for legacy item format (should not happen with backend cart data)
                    logger.warning(f"[INIT] Unexpected item format: {type(cart_item)}")
                    init_item = {
                        'id': getattr(cart_item, 'local_id', 'unknown'),
                        'quantity': {'count': getattr(cart_item, 'quantity', 1)}
                    }
                    init_items.append(init_item)
            
            logger.info(f"[INIT] Successfully mapped {len(init_items)} items for INIT request")
            
            # Step 11: Parse address components for proper INIT structure  
            logger.info(f"[INIT] Processing address components from: '{delivery_address}'")
            
            # Enhanced address parsing - split into meaningful components
            address_parts = delivery_address.replace(',', '').split()
            
            # Improved parsing logic with better defaults
            if len(address_parts) >= 3:
                parsed_building = " ".join(address_parts[0:2])  # First 2 parts as building (house no + apt)
                parsed_street = " ".join(address_parts[2:4]) if len(address_parts) > 2 else "Street"  # Next 2 parts as street
                parsed_locality = " ".join(address_parts[4:]) if len(address_parts) > 4 else final_city  # Rest as locality
            elif len(address_parts) == 2:
                parsed_building = address_parts[0]
                parsed_street = address_parts[1]
                parsed_locality = final_city
            elif len(address_parts) == 1:
                parsed_building = address_parts[0]
                parsed_street = final_city + " Street"
                parsed_locality = final_city
            else:
                # Fallback if empty address (user should provide proper structured input)
                logger.warning(f"[INIT] ⚠️ Empty or invalid delivery address provided!")
                return {
                    'success': False,
                    'message': '❌ **Invalid Address**\n\nPlease provide complete address details:\n\n'
                    '**Required Format:**\n'
                    '• Building: House number, apartment (e.g., "123", "Apt 4B")\n'
                    '• Street: Street name (e.g., "Main Street")\n'
                    '• Locality: Area name (e.g., "Downtown", "Sector 5")\n'
                    '• City, State, Pincode: Full location details\n\n'
                    '**Use structured inputs for accurate delivery!**'
                }
            
            logger.info(f"[INIT] ✅ Address components -> Building: '{parsed_building}', Street: '{parsed_street}', Locality: '{parsed_locality}'")
            
            # Step 11.1: Create delivery_info structure matching working curl format
            delivery_info_struct = {
                "type": "Delivery",
                "phone": phone,
                "name": customer_name,
                "email": email,
                "location": {
                    "gps": f"{gps_coordinates['lat']},{gps_coordinates['lng']}",  # User-provided GPS coordinates
                    "address": {
                        "name": customer_name,
                        "building": parsed_building,  # Parsed from delivery_address
                        "street": parsed_street,      # Parsed from delivery_address
                        "locality": parsed_locality,  # Parsed from delivery_address or use city
                        "city": final_city,
                        "state": final_state,
                        "country": "India",
                        "areaCode": final_pincode,
                        "tag": "Home",
                        "lat": str(gps_coordinates['lat']),
                        "lng": str(gps_coordinates['lng']),
                        "email": email
                    }
                }
            }
            
            # Step 11.1: Create ONDC-compliant INIT request (matching exact ONDC specification)
            # ✅ FIXED: Remove 'order' wrapper - ONDC expects message.items directly
            init_data = {
                'context': {
                    'transaction_id': session.checkout_state.transaction_id,
                    'city': final_pincode,  # Use pincode as city for context 
                    'domain': 'ONDC:RET10'  # Standard ONDC retail domain
                },
                'message': {
                    'items': init_items,  # ✅ FIXED: Direct under message, not message.order
                    'billing_info': {
                        'address': {
                            'name': customer_name,
                            'building': parsed_building,   # Parsed from delivery_address
                            'street': parsed_street,       # Parsed from delivery_address
                            'locality': parsed_locality,   # Parsed from delivery_address
                            'city': final_city,
                            'state': final_state,
                            'country': "India",
                            'areaCode': final_pincode,
                            'tag': "Home",
                            'lat': str(gps_coordinates['lat']),
                            'lng': str(gps_coordinates['lng']),
                            'email': email
                        },
                        'phone': phone,
                        'name': customer_name,
                        'email': email
                    },
                    'delivery_info': delivery_info_struct,
                    'payment': {
                        'type': 'ON-ORDER'  # Matching working curl format
                    }
                },
                'deviceId': session.device_id  # deviceId at root level as in working curl
            }
            
            # Step 11.2: Validate INIT payload structure
            if not validate_init_payload(init_data):
                logger.error(f"[INIT] {INIT_VALIDATION_ERROR}")
                return {
                    'success': False,
                    'message': f'❌ **INIT Payload Structure Error**\n\n{INIT_VALIDATION_ERROR}'
                }
            
            logger.info(f"[INIT] ✅ INIT payload structure validated successfully")
            
            # Step 12: Enhanced INIT request logging
            logger.info(f"[INIT] 🚀 Sending INIT request to Himira backend")
            logger.info(f"[INIT] 📋 Request Summary:")
            logger.info(f"  - Transaction ID: {init_data['context']['transaction_id']}")
            logger.info(f"  - Customer: {customer_name}")
            logger.info(f"  - Items Count: {len(init_data['message']['items'])}")
            logger.info(f"  - Device ID: {init_data['deviceId']}")
            logger.info(f"  - City: {init_data['context']['city']}")
            logger.info(f"  - Structure: message.items (ONDC compliant, not message.order)")
            
            # Log the complete INIT payload for debugging
            logger.debug(f"[INIT] 📤 Complete INIT payload:")
            logger.debug(f"{json.dumps(init_data, indent=2)}")
            
            # Step 12.1: Call BIAP INIT API - GUEST MODE
            # Guest mode: No authentication required for order initialization
            auth_token = getattr(session, 'auth_token', None)
            if not auth_token:
                logger.info("[INIT] GUEST MODE - Proceeding without auth token")
                # For guest users, we'll use wil-api-key authentication only
                auth_token = None
            
            logger.info(f"[INIT] 🌐 Calling /v2/initialize_order endpoint...")
            init_response = await self.buyer_app.initialize_order(init_data, auth_token=auth_token)
            
            # Cache INIT response immediately for debugging/recovery
            if init_response:
                message_id_for_cache = session.checkout_state._extract_message_id(init_response)
                session.checkout_state.cache_operation_response('init', init_response, message_id_for_cache)
                logger.info(f"[INIT] 💾 Cached INIT response with message_id: {message_id_for_cache}")
            
            # 🔥 ENHANCED INIT RESPONSE LOGGING - LOG IMMEDIATELY BEFORE POLLING
            if init_response:
                logger.info(f"[INIT] ✅ INIT API response received successfully")
                logger.info(f"[INIT] 📥 Raw INIT response received BEFORE polling:")
                logger.info(f"[INIT] Response URL template: https://hp-buyer-backend-preprod.himira.co.in/clientApis/v2/on_initialize_order?messageIds={{message_id}}")
                logger.info(f"[INIT] 📋 Full INIT response structure:")
                logger.info(f"{json.dumps(init_response, indent=2)}")
                
                # Log response structure analysis with more detail
                if isinstance(init_response, list) and len(init_response) > 0:
                    logger.info(f"[INIT] 📊 Response is array with {len(init_response)} item(s)")
                    first_item = init_response[0]
                    
                    # Check for different message_id locations
                    if 'messageId' in first_item:
                        logger.info(f"[INIT] 🔍 Found messageId at root: {first_item['messageId']}")
                    if 'message_id' in first_item:
                        logger.info(f"[INIT] 🔍 Found message_id at root: {first_item['message_id']}")
                    if 'context' in first_item:
                        context = first_item['context']
                        logger.info(f"[INIT] 📍 Found context with transaction_id: {context.get('transaction_id')}")
                        if 'message_id' in context:
                            logger.info(f"[INIT] ✅ Found message_id in context: {context['message_id']}")
                            logger.info(f"[INIT] 🔗 Full polling URL: https://hp-buyer-backend-preprod.himira.co.in/clientApis/v2/on_initialize_order?messageIds={context['message_id']}")
                        if 'messageId' in context:
                            logger.info(f"[INIT] ✅ Found messageId in context: {context['messageId']}")
                            logger.info(f"[INIT] 🔗 Full polling URL: https://hp-buyer-backend-preprod.himira.co.in/clientApis/v2/on_initialize_order?messageIds={context['messageId']}")
                    
                    # Log all keys for debugging
                    logger.debug(f"[INIT] 🔧 First item keys: {list(first_item.keys())}")
                    if 'context' in first_item:
                        logger.debug(f"[INIT] 🔧 Context keys: {list(first_item['context'].keys())}")
                        
                elif isinstance(init_response, dict):
                    logger.info(f"[INIT] 📊 Response is dict with keys: {list(init_response.keys())}")
                    if 'messageId' in init_response:
                        logger.info(f"[INIT] 🔍 Found messageId: {init_response['messageId']}")
                    if 'message_id' in init_response:
                        logger.info(f"[INIT] 🔍 Found message_id: {init_response['message_id']}")
            else:
                logger.error(f"[INIT] ❌ INIT API returned None/empty response")
            
            logger.info(f"[CheckoutService] INIT API initial response received")
            
            # Extract message ID from ONDC INIT response 
            # ONDC format: response[0]['context']['message_id']
            message_id = None
            if isinstance(init_response, list) and len(init_response) > 0:
                first_item = init_response[0]
                if 'context' in first_item and 'message_id' in first_item['context']:
                    message_id = first_item['context']['message_id']
                    logger.info(f"[INIT] Extracted message_id from context: {message_id}")
            
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
                # INIT step only confirms availability and pricing (ONDC spec)
                # Order ID is generated later during CONFIRM step, not here
                logger.info("[CheckoutService] INIT successful - order ready for confirmation")
                
                # Update session to INIT stage  
                session.checkout_state.stage = CheckoutStage.INIT
                session.add_to_history('initialize_order', {
                    'address': delivery_address,
                    'phone': phone,
                    'email': email,
                    'payment_method': payment_method,
                    'init_successful': True
                })
                
                return {
                    'success': True,
                    'message': f' Order initialized successfully!',
                    'stage': 'init_completed',
                    'order_summary': {
                        'items': len(cart_items),
                        'total': cart_summary.get('total_value', 0.0),
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
            # Get cart summary for payment amount
            cart_summary = await self.cart_service.get_cart_summary(session)
            total_amount = cart_summary['total_value']
            
            # MOCK PAYMENT CREATION - Only if MOCK_AFTER_INIT is enabled
            mock_after_init_env = os.getenv('MOCK_AFTER_INIT', 'false')
            mock_after_init = mock_after_init_env.lower() == 'true'
            
            # 🔥 DEBUG: Log environment variable status
            logger.error(f"[PAYMENT DEBUG] MOCK_AFTER_INIT env var: '{mock_after_init_env}'")
            logger.error(f"[PAYMENT DEBUG] Mock after init enabled: {mock_after_init}")
            logger.error(f"[PAYMENT DEBUG] Config payment mock mode: {config.payment.mock_mode}")
            
            if mock_after_init or config.payment.mock_mode:
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
        # Validate session is in INIT stage (payment creation keeps session in INIT)
        if session.checkout_state.stage != CheckoutStage.INIT:
            return {
                'success': False,
                'message': f' Please complete delivery and payment details first. Current stage: {session.checkout_state.stage.value}'
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
            # Get cart items from backend for INIT/CONFIRM
            success, cart_display, cart_items = await self.cart_service.view_cart(session)
            if not success:
                return {
                    'success': False,
                    'message': f' Failed to get cart items: {cart_display}'
                }
            
            enriched_items = await self.product_enrichment.enrich_cart_items(
                cart_items, session.session_id
            )
            
            # Get cart summary for order total calculations
            cart_summary = await self.cart_service.get_cart_summary(session)
            
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
            total_amount = cart_summary['total_value']
            
            # PAYMENT OBJECT CREATION - Only mock if PAYMENT_MOCK_MODE is enabled
            payment_mock_mode = os.getenv('PAYMENT_MOCK_MODE', 'false').lower() == 'true'
            if config.payment.mock_mode or payment_mock_mode:
                logger.info(f"[MOCK PAYMENT] Creating mock payment for amount: {total_amount}")
                payment_obj = mock_payment_service.create_biap_payment_object(total_amount)
                
                if config.payment.debug_logs:
                    logger.info(f"[MOCK PAYMENT] Payment ID: {payment_obj['razorpayPaymentId']}")
                    logger.info(f"[MOCK PAYMENT] Settlement basis: {payment_obj['@ondc/org/settlement_basis']}")
                    logger.info(f"[MOCK PAYMENT] Settlement window: {payment_obj['@ondc/org/settlement_window']}")
            else:
                # Real payment implementation for CONFIRM step
                logger.info("[REAL PAYMENT] Creating real payment object for CONFIRM")
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
                'userId': session.user_id  # Fixed: Use real Firebase user ID
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
            
            # MOCK AFTER INIT: Mock payment and confirm steps after real INIT
            mock_after_init = os.getenv('MOCK_AFTER_INIT', 'false').lower() == 'true'
            if mock_after_init:
                logger.info(f"[CheckoutService] MOCK CONFIRM ONLY - Simulating final order confirmation step (rest of journey was real)")
                logger.info(f"[CheckoutService] Real ONDC flow completed: SEARCH → CART → SELECT → INIT → [MOCK CONFIRM]")
                
                # Generate order_id for mock confirmation (INIT doesn't provide order_id per ONDC spec)
                mock_order_id = self._generate_order_id()
                
                # Create mock confirmation response
                result = {
                    'success': True,
                    'order_id': mock_order_id,
                    'message': f'🎉 Order confirmed successfully! Order ID: {mock_order_id} (Mock confirmation - real ONDC journey completed through INIT)',
                    'mock_confirm': True,
                    'real_journey_completed': ['SEARCH', 'CART', 'SELECT', 'INIT'],
                    'note': 'Order ID generated for mock CONFIRM since INIT step only confirms pricing per ONDC spec'
                }
                logger.info(f"[CheckoutService] Generated mock order_id for CONFIRM: {mock_order_id}")
                logger.info("[CheckoutService] Note: INIT step correctly does not provide order_id (per ONDC specification)")
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
                    'total': cart_summary['total_value']
                })
                
                # Get final cart details before clearing
                success, cart_display, cart_items = await self.cart_service.view_cart(session)
                
                total_value = cart_summary['total_value']
                delivery_address = session.checkout_state.delivery_info.address
                phone = session.checkout_state.delivery_info.phone
                
                # Clear cart after successful order
                success, message = await self.cart_service.clear_cart(session)
                
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
    
    def _get_gps_from_location(self, pincode: str, city: str, state: str) -> Dict[str, str]:
        """
        Get GPS coordinates based on pincode/city/state
        Returns real coordinates for major cities, fallback for others
        """
        # Major city GPS coordinates database
        city_coordinates = {
            # Karnataka
            'bangalore': {'lat': '12.9716', 'lng': '77.5946'},
            'bengaluru': {'lat': '12.9716', 'lng': '77.5946'},
            'mysore': {'lat': '12.2958', 'lng': '76.6394'},
            'mangalore': {'lat': '12.9141', 'lng': '74.8560'},
            
            # Punjab/Chandigarh  
            'chandigarh': {'lat': '30.7333', 'lng': '76.7794'},
            'ludhiana': {'lat': '30.9010', 'lng': '75.8573'},
            'amritsar': {'lat': '31.6340', 'lng': '74.8723'},
            
            # Delhi NCR
            'delhi': {'lat': '28.7041', 'lng': '77.1025'},
            'new delhi': {'lat': '28.6139', 'lng': '77.2090'},
            'gurgaon': {'lat': '28.4595', 'lng': '77.0266'},
            'noida': {'lat': '28.5355', 'lng': '77.3910'},
            
            # Maharashtra
            'mumbai': {'lat': '19.0760', 'lng': '72.8777'},
            'pune': {'lat': '18.5204', 'lng': '73.8567'},
            'nashik': {'lat': '19.9975', 'lng': '73.7898'},
            'nagpur': {'lat': '21.1458', 'lng': '79.0882'},
            
            # Other major cities
            'hyderabad': {'lat': '17.3850', 'lng': '78.4867'},
            'chennai': {'lat': '13.0827', 'lng': '80.2707'},
            'kolkata': {'lat': '22.5726', 'lng': '88.3639'},
            'ahmedabad': {'lat': '23.0225', 'lng': '72.5714'},
            'jaipur': {'lat': '26.9124', 'lng': '75.7873'},
            'lucknow': {'lat': '26.8467', 'lng': '80.9462'},
            'kanpur': {'lat': '26.4499', 'lng': '80.3319'},
            'kochi': {'lat': '9.9312', 'lng': '76.2673'},
            'coimbatore': {'lat': '11.0168', 'lng': '76.9558'},
            'visakhapatnam': {'lat': '17.6868', 'lng': '83.2185'}
        }
        
        # Pincode-based coordinates for specific areas
        pincode_coordinates = {
            # Bangalore area codes
            '560001': {'lat': '12.9716', 'lng': '77.5946'},  # Bangalore Central
            '560002': {'lat': '12.9698', 'lng': '77.5990'},  # Bangalore City
            '560025': {'lat': '12.9279', 'lng': '77.6271'},  # Bangalore South
            
            # Chandigarh area codes  
            '160001': {'lat': '30.7333', 'lng': '76.7794'},  # Chandigarh
            '140301': {'lat': '30.7456', 'lng': '76.6536'},  # Chandigarh specific
            
            # Delhi area codes
            '110001': {'lat': '28.6139', 'lng': '77.2090'},  # New Delhi
            '110011': {'lat': '28.7041', 'lng': '77.1025'},  # Delhi Central
            
            # Mumbai area codes
            '400001': {'lat': '18.9322', 'lng': '72.8264'},  # Mumbai Fort
            '400070': {'lat': '19.0760', 'lng': '72.8777'},  # Mumbai Andheri
        }
        
        # Try exact pincode match first
        if pincode and pincode in pincode_coordinates:
            coords = pincode_coordinates[pincode]
            logger.info(f"[GPS] 🎯 Found exact pincode match for {pincode}: {coords['lat']}, {coords['lng']}")
            return coords
        
        # Try city name match
        city_lower = city.lower().strip() if city else ""
        if city_lower and city_lower in city_coordinates:
            coords = city_coordinates[city_lower]
            logger.info(f"[GPS] 🏙️ Found city match for {city}: {coords['lat']}, {coords['lng']}")
            return coords
        
        # Fallback based on state/region
        state_lower = state.lower().strip() if state else ""
        state_fallbacks = {
            'karnataka': {'lat': '12.9716', 'lng': '77.5946'},  # Bangalore
            'punjab': {'lat': '30.7333', 'lng': '76.7794'},     # Chandigarh
            'chandigarh': {'lat': '30.7333', 'lng': '76.7794'}, # Chandigarh
            'delhi': {'lat': '28.6139', 'lng': '77.2090'},      # Delhi
            'maharashtra': {'lat': '19.0760', 'lng': '72.8777'}, # Mumbai
            'telangana': {'lat': '17.3850', 'lng': '78.4867'},   # Hyderabad
            'tamil nadu': {'lat': '13.0827', 'lng': '80.2707'},  # Chennai
            'west bengal': {'lat': '22.5726', 'lng': '88.3639'}, # Kolkata
        }
        
        if state_lower and state_lower in state_fallbacks:
            coords = state_fallbacks[state_lower]
            logger.info(f"[GPS] 🗺️ Using state fallback for {state}: {coords['lat']}, {coords['lng']}")
            return coords
        
        # Ultimate fallback - Bangalore coordinates
        fallback_coords = {'lat': '12.9716', 'lng': '77.5946'}
        logger.warning(f"[GPS] ⚠️ No specific match found for {city}, {state} {pincode}. Using Bangalore fallback: {fallback_coords['lat']}, {fallback_coords['lng']}")
        return fallback_coords
    
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
        from .cart_service import get_cart_service
        _checkout_service = CheckoutService(cart_service=get_cart_service())
    return _checkout_service