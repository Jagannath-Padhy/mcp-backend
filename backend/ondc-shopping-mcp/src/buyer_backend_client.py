"""
Comprehensive ONDC Buyer Backend Client

This client implements ALL buyer backend APIs from biap-client-node-js
to provide a complete ONDC buyer experience via MCP.

Based on comprehensive API documentation from /Users/jagannath/Desktop/ondc-genie/biap-client-node-js
"""

import httpx
import json
import logging
import shlex
import os
from typing import Dict, List, Optional, Any, Union
from urllib.parse import urlencode
import asyncio

from .utils.logger import get_logger

logger = get_logger(__name__)


class BuyerBackendClient:
    """Comprehensive client for all ONDC buyer backend APIs"""
    
    def __init__(self, base_url: str = None, api_key: str = None, debug_curl: bool = None):
        """
        Initialize the comprehensive buyer backend client
        
        Args:
            base_url: Base URL of the buyer backend (e.g., http://localhost:3000)
            api_key: WIL API key for authentication
            debug_curl: Enable CURL command logging for debugging
        """
        # Use environment variables - no hardcoded defaults
        self.base_url = base_url or os.getenv("BACKEND_ENDPOINT")
        self.api_key = api_key or os.getenv("WIL_API_KEY")
        
        if not self.base_url:
            raise ValueError("BACKEND_ENDPOINT environment variable or base_url parameter is required")
        if not self.api_key:
            raise ValueError("WIL_API_KEY environment variable or api_key parameter is required")
        self.debug_curl = debug_curl if debug_curl is not None else os.getenv("DEBUG_CURL_LOGGING", "false").lower() == "true"
        
        # Check if base_url already contains /clientApis
        if self.base_url.endswith('/clientApis'):
            self.client_apis_base = self.base_url
        else:
            self.client_apis_base = f"{self.base_url}/clientApis"
        
        # HTTP client configuration
        self.timeout = httpx.Timeout(30.0)
        self.limits = httpx.Limits(max_keepalive_connections=5, max_connections=10)
        
        logger.info(f"BuyerBackendClient initialized with base_url: {self.base_url}")
        if self.debug_curl:
            logger.info("CURL logging enabled for API calls")
    
    def _generate_curl_command(self, method: str, url: str, headers: Dict, 
                               params: Optional[Dict], json_data: Optional[Dict]) -> str:
        """Generate curl command for debugging"""
        curl_parts = ['curl', '-X', method.upper()]
        
        # Add headers (skip auto-generated ones)
        for key, value in headers.items():
            if key.lower() not in ['content-length', 'host', 'user-agent']:
                curl_parts.extend(['-H', shlex.quote(f'{key}: {value}')])
        
        # Add JSON data
        if json_data:
            curl_parts.extend(['-d', shlex.quote(json.dumps(json_data, separators=(',', ':')))])
        
        # Add params to URL
        if params:
            url = f"{url}?{urlencode(params)}"
        
        curl_parts.append(shlex.quote(url))
        return ' '.join(curl_parts)
    
    async def _make_request(
        self, 
        method: str, 
        endpoint: str, 
        params: Optional[Dict] = None,
        json_data: Optional[Dict] = None,
        headers: Optional[Dict] = None,
        auth_token: Optional[str] = None,
        require_auth: bool = False
    ) -> Optional[Dict]:
        """Make HTTP request to buyer backend"""
        
        # Build full URL
        url = f"{self.client_apis_base}{endpoint}"
        
        # Prepare headers
        request_headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "wil-api-key": self.api_key
        }
        
        if headers:
            request_headers.update(headers)
            
        if auth_token:
            request_headers["Authorization"] = f"Bearer {auth_token}"
        elif require_auth:
            logger.warning(f"Auth required for {endpoint} but no token provided")
        
        # Log CURL commands for debugging when enabled
        if self.debug_curl:
            curl_cmd = self._generate_curl_command(method, url, request_headers, params, json_data)
            logger.info(f"CURL: {curl_cmd}")
            
        # Add explicit request logging
        logger.info(f"[REQUEST] {method.upper()} {url}")
        logger.info(f"[REQUEST] Headers: {request_headers}")
        if json_data:
            logger.info(f"[REQUEST] Body: {json.dumps(json_data, indent=2)}")
        if params:
            logger.info(f"[REQUEST] Params: {params}")
        
        # Enhanced logging for cart operations
        if '/cart/' in endpoint:
            logger.info(f"[CART-API] About to execute {method} {endpoint}")
            if json_data:
                logger.info(f"[CART-API] Payload keys: {list(json_data.keys())}")
                logger.info(f"[CART-API] Full payload: {json_data}")
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout, limits=self.limits) as client:
                response = await client.request(
                    method=method.upper(),
                    url=url,
                    params=params,
                    json=json_data,
                    headers=request_headers
                )
                
                logger.debug(f"{method.upper()} {url} -> {response.status_code}")
                
                # FORCE response logging for debugging
                force_response_log = (endpoint in ["/v2/initialize_order", "/v2/select"] or 
                                    '/cart/' in endpoint or  # Force logging for all cart operations
                                    self.debug_curl)
                
                if force_response_log:
                    logger.info(f"[RESPONSE] Status: {response.status_code}")
                    logger.info(f"[RESPONSE] Headers: {dict(response.headers)}")
                    try:
                        response_json = response.json()
                        logger.info(f"[RESPONSE] Body:\n{json.dumps(response_json, indent=2)}")
                    except Exception as e:
                        logger.info(f"[RESPONSE] Body (raw): {response.text}")
                        logger.warning(f"[RESPONSE] Failed to parse JSON: {e}")
                    
                # Legacy response logging for compatibility
                if self.debug_curl or endpoint == "/v2/select":
                    logger.info(f"[CURL RESPONSE] Status: {response.status_code}")
                    try:
                        response_json = response.json()
                        logger.info(f"[CURL RESPONSE] Body:\n{json.dumps(response_json, indent=2)}")
                    except:
                        logger.info(f"[CURL RESPONSE] Body:\n{response.text[:1000]}")  # Limit size
                
                if response.status_code == 200:
                    try:
                        return response.json()
                    except json.JSONDecodeError:
                        return {"success": True, "data": response.text}
                elif response.status_code == 401:
                    logger.error(f"Unauthorized access to {endpoint}. Check WIL_API_KEY validity or auth token for authenticated endpoints.")
                    return None  # Return None for HTTP errors to trigger proper error handling
                elif response.status_code == 400:
                    # Log detailed error for 400 Bad Request
                    error_details = response.text
                    try:
                        error_json = response.json()
                        error_details = json.dumps(error_json, indent=2)
                    except:
                        pass
                    logger.error(f"HTTP 400 Bad Request for {endpoint}. Invalid request format or missing required fields.")
                    logger.error(f"Request data sent: {json.dumps(json_data, indent=2) if json_data else 'None'}")
                    logger.error(f"Backend response: {error_details}")
                    return None  # Return None for HTTP errors to trigger proper error handling
                elif response.status_code == 404:
                    logger.warning(f"HTTP 404 for {endpoint}: Endpoint not found or resource unavailable. {response.text}")
                    return None  # Return None for HTTP errors to trigger proper error handling
                else:
                    # For all other HTTP error status codes (500, etc.), return None
                    logger.error(f"HTTP {response.status_code} for {endpoint}: Server error. {response.text}")
                    # Enhanced logging for cart operations specifically
                    if '/cart/' in endpoint:
                        logger.error(f"[CART-API] HTTP {response.status_code} error details:")
                        try:
                            error_json = response.json()
                            logger.error(f"[CART-API] Error response: {json.dumps(error_json, indent=2)}")
                        except:
                            logger.error(f"[CART-API] Raw error response: {response.text}")
                    return None  # Return None to trigger proper error handling in cart service
                    
        except Exception as e:
            logger.error(f"Network/connection error for {endpoint}: {e}. Check backend availability and network connectivity.")
            return None  # Return None for connection errors to trigger proper error handling
    
    # ================================
    # SEARCH & DISCOVERY APIs
    # ================================
    
    async def search_products(
        self, 
        user_id: str, 
        query: Union[str, Dict] = "", 
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        **filters
    ) -> Optional[Dict]:
        """Search for products - main search API"""
        # Handle both string query and dict params
        if isinstance(query, dict):
            params = query
        else:
            params = {"name": query}
            if latitude and longitude:
                params.update({"latitude": latitude, "longitude": longitude})
            params.update(filters)
        
        return await self._make_request("GET", f"/v2/search/{user_id}", params=params)
    
    async def get_item_details(self, item_params: Dict) -> Optional[Dict]:
        """
        BIAP protocolGetItemDetails: Get detailed item information
        
        Args:
            item_params: Dictionary with 'id' key for item ID
            
        Returns:
            Item details with enriched product data
        """
        item_id = item_params.get('id')
        if not item_id:
            return None
        return await self._make_request("GET", f"/v2/items/{item_id}")
    
    async def get_item_list(self, list_params: Dict) -> Optional[Dict]:
        """
        BIAP protocolGetItemList: Get multiple items information in batch
        
        Args:
            list_params: Dictionary with:
                - itemIds: Comma-separated string of item IDs  
                - providerIds: Comma-separated string of provider IDs
        
        Returns:
            Batch item details with enriched product data
        """
        params = {}
        if list_params.get('itemIds'):
            params['itemIds'] = list_params['itemIds']
        if list_params.get('providerIds'):  
            params['providerIds'] = list_params['providerIds']
            
        return await self._make_request("GET", "/v2/items/batch", params=params)
    
    async def get_multiple_items(self, **filters) -> Optional[Dict]:
        """Get multiple items with filters"""
        return await self._make_request("GET", "/v2/items", params=filters)
    
    async def get_provider_by_item(self, item_id: str) -> Optional[Dict]:
        """Get provider information by item ID"""
        return await self._make_request("GET", f"/v2/providers/{item_id}")
    
    async def get_provider_details(self, **params) -> Optional[Dict]:
        """Get detailed provider information"""
        return await self._make_request("GET", "/v2/provider-details", params=params)
    
    async def get_locations(self, location_id: str = None, **params) -> Optional[Dict]:
        """Get locations or specific location by ID"""
        endpoint = f"/v2/locations/{location_id}" if location_id else "/v2/locations"
        return await self._make_request("GET", endpoint, params=params)
    
    async def get_categories(self, limit: int = 100, page: int = 1) -> Optional[Dict]:
        """Get product categories"""
        return await self._make_request("GET", "/categories", params={"limit": limit, "page": page})
    
    async def get_custom_category(self, name: str) -> Optional[Dict]:
        """Get custom category by name"""
        return await self._make_request("GET", "/category", params={"name": name})
    
    async def get_providers_list(self, **params) -> Optional[Dict]:
        """Get providers list"""
        return await self._make_request("GET", "/v2/providers", params=params)
    
    async def get_custom_menus(self, **params) -> Optional[Dict]:
        """Get custom provider menus"""
        return await self._make_request("GET", "/v2/custom-menus", params=params)
    
    # ================================
    # CART MANAGEMENT APIs
    # ================================
    
    async def add_to_cart(self, user_id: str, device_id: str, cart_data: Dict) -> Optional[Dict]:
        """Add item to cart"""
        return await self._make_request("POST", f"/v2/cart/{user_id}/{device_id}", json_data=cart_data)
    
    async def get_cart(self, user_id: str, device_id: str) -> Optional[Dict]:
        """Get cart contents"""
        return await self._make_request("GET", f"/v2/cart/{user_id}/{device_id}")
    
    async def update_cart_item(self, user_id: str, item_id: str, update_data: Dict) -> Optional[Dict]:
        """Update cart item"""
        return await self._make_request("PUT", f"/v2/cart/{user_id}/{item_id}", json_data=update_data)
    
    async def remove_cart_item(self, user_id: str, device_id: str, item_id: str) -> Optional[Dict]:
        """Remove specific item from cart"""
        return await self._make_request("DELETE", f"/v2/cart/{user_id}/{device_id}/{item_id}")
    
    async def remove_multiple_cart_items(self, user_id: str, device_id: str, item_ids: List[str]) -> Optional[Dict]:
        """Remove multiple items from cart"""
        payload = {
            "itemIds": item_ids,  # Fixed: was "item_ids"  
            "userId": user_id     # Fixed: was missing
        }
        return await self._make_request("DELETE", f"/v2/cart/multiple/{user_id}/{device_id}", json_data=payload)
    
    async def clear_cart(self, user_id: str, device_id: str) -> Optional[Dict]:
        """Clear entire cart"""
        return await self._make_request("DELETE", f"/v2/all/cart/{user_id}/{device_id}")
    
    async def move_cart_to_wishlist(self, user_id: str, device_id: str, item_data: Dict) -> Optional[Dict]:
        """Move cart items to wishlist"""
        return await self._make_request("POST", f"/v2/cart/wishlist/{user_id}/{device_id}", json_data=item_data)
    
    # ================================
    # WISHLIST MANAGEMENT APIs
    # ================================
    
    async def add_to_wishlist(self, user_id: str, device_id: str, item_data: Dict) -> Optional[Dict]:
        """Add item to wishlist"""
        return await self._make_request("POST", f"/v2/wishlist/{user_id}/{device_id}", json_data=item_data)
    
    async def get_wishlist(self, user_id: str, device_id: str) -> Optional[Dict]:
        """Get wishlist items"""
        return await self._make_request("GET", f"/v2/wishlist/{user_id}/{device_id}")
    
    async def update_wishlist_item(self, user_id: str, item_id: str, update_data: Dict) -> Optional[Dict]:
        """Update wishlist item"""
        return await self._make_request("PUT", f"/v2/wishlist/{user_id}/{item_id}", json_data=update_data)
    
    async def remove_wishlist_item(self, user_id: str, device_id: str, item_id: str) -> Optional[Dict]:
        """Remove item from wishlist"""
        return await self._make_request("DELETE", f"/v2/item/wishlist/{user_id}/{device_id}/{item_id}")
    
    async def clear_wishlist(self, user_id: str, device_id: str) -> Optional[Dict]:
        """Clear entire wishlist"""
        return await self._make_request("DELETE", f"/v2/all/wishlist/{user_id}/{device_id}")
    
    # ================================
    # ONDC ORDER FLOW APIs (Critical)
    # ================================
    
    async def select_items(self, select_data: Dict, auth_token: Optional[str] = None) -> Optional[Dict]:
        """ONDC Select: Select items for order (quote request)
        
        Himira backend expects array of requests for v2/select
        Auth token is required for authenticated SELECT operations
        """
        # Wrap in array if not already (Himira backend requirement)
        if not isinstance(select_data, list):
            select_data = [select_data]
        
        result = await self._make_request("POST", "/v2/select", json_data=select_data,
                                        auth_token=auth_token, require_auth=False)
        
        # Unwrap response if it's a single-item array
        if isinstance(result, list) and len(result) == 1:
            return result[0]
        return result
    
    async def get_select_response(self, **params) -> Optional[Dict]:
        """Get select response data"""
        return await self._make_request("GET", "/v2/on_select", params=params)
    
    async def get_select_data_by_message_id(self, message_id: str) -> Optional[Dict]:
        """Get select data by message ID"""
        return await self._make_request("GET", f"/v2/select/Data/{message_id}")
    
    async def initialize_order(self, init_data: Dict, auth_token: Optional[str] = None) -> Optional[Dict]:
        """ONDC Init: Initialize order"""
        return await self._make_request("POST", "/v2/initialize_order", 
                                      json_data=init_data, auth_token=auth_token, require_auth=False)
    
    async def get_init_response(self, auth_token: Optional[str] = None, **params) -> Optional[Dict]:
        """Get init response"""
        return await self._make_request("GET", "/v2/on_initialize_order", 
                                      params=params, auth_token=auth_token, require_auth=False)
    
    async def confirm_order(self, confirm_data: Dict, auth_token: Optional[str] = None) -> Optional[Dict]:
        """ONDC Confirm: Confirm order"""
        return await self._make_request("POST", "/v2/confirm_order", 
                                      json_data=confirm_data, auth_token=auth_token, require_auth=False)
    
    async def get_confirm_response(self, auth_token: Optional[str] = None, **params) -> Optional[Dict]:
        """Get confirm response"""
        return await self._make_request("GET", "/v2/on_confirm_order", 
                                      params=params, auth_token=auth_token, require_auth=False)
    
    # ================================
    # ORDER MANAGEMENT APIs
    # ================================
    
    async def get_orders(self, auth_token: str, **filters) -> Optional[Dict]:
        """Get user's order history"""
        return await self._make_request("GET", "/v2/orders", 
                                      params=filters, auth_token=auth_token, require_auth=True)
    
    async def get_order_by_id(self, order_id: str, auth_token: str) -> Optional[Dict]:
        """Get specific order by ID"""
        return await self._make_request("GET", f"/v2/orders/{order_id}", 
                                      auth_token=auth_token, require_auth=True)
    
    async def get_order_by_transaction(self, transaction_id: str) -> Optional[Dict]:
        """Get order by transaction ID"""
        return await self._make_request("GET", f"/v2/order/{transaction_id}")
    
    async def get_order_status(self, status_data: Dict, auth_token: str) -> Optional[Dict]:
        """Get order status"""
        return await self._make_request("POST", "/v2/order_status", 
                                      json_data=status_data, auth_token=auth_token, require_auth=True)
    
    async def cancel_order(self, cancel_data: Dict, auth_token: str) -> Optional[Dict]:
        """Cancel order"""
        return await self._make_request("POST", "/v2/cancel_order", 
                                      json_data=cancel_data, auth_token=auth_token, require_auth=True)
    
    async def update_order(self, update_data: Dict, auth_token: str) -> Optional[Dict]:
        """Update order"""
        return await self._make_request("POST", "/v2/update", 
                                      json_data=update_data, auth_token=auth_token, require_auth=True)
    
    # ================================
    # PAYMENT APIs
    # ================================
    
    async def create_razorpay_order(self, order_data: Dict, auth_token: Optional[str] = None) -> Optional[Dict]:
        """Create RazorPay order"""
        return await self._make_request("POST", "/v2/razorpay/createOrder", 
                                      json_data=order_data, auth_token=auth_token, require_auth=False)
    
    async def create_razorpay_payment(self, order_id: str, payment_data: Dict, auth_token: str) -> Optional[Dict]:
        """Create RazorPay payment"""
        return await self._make_request("POST", f"/v2/razorpay/{order_id}", 
                                      json_data=payment_data, auth_token=auth_token, require_auth=True)
    
    async def verify_razorpay_payment(self, verification_data: Dict) -> Optional[Dict]:
        """Verify RazorPay payment"""
        return await self._make_request("POST", "/v2/razorpay/verify/process", json_data=verification_data)
    
    async def get_razorpay_keys(self) -> Optional[Dict]:
        """Get RazorPay keys"""
        return await self._make_request("GET", "/v2/razorpay/razorPay/keys")
    
    # ================================
    # USER/ACCOUNT APIs
    # ================================
    
    async def create_guest_user(self, guest_data: Dict) -> Optional[Dict]:
        """Create guest user"""
        return await self._make_request("POST", "/create/guest/user", json_data=guest_data)
    
    async def guest_user_login(self, login_data: Dict) -> Optional[Dict]:
        """Guest user login"""
        return await self._make_request("POST", "/guestUserLogin", json_data=login_data)
    
    async def signup(self, signup_data: Dict) -> Optional[Dict]:
        """Send OTP for phone registration - matches frontend flow"""
        return await self._make_request("POST", "/signup", json_data=signup_data)
    
    async def login_with_phone(self, phone_data: Dict) -> Optional[Dict]:
        """Direct phone login (bypass method - not used in real flow)"""
        return await self._make_request("POST", "/loginWithPhone", json_data=phone_data)
    
    async def verify_otp(self, otp_data: Dict) -> Optional[Dict]:
        """Verify OTP and get backend JWT token - matches frontend flow"""
        return await self._make_request("POST", "/verifyotp", json_data=otp_data)
    
    async def get_user_profile(self, auth_token: str) -> Optional[Dict]:
        """Get user profile"""
        return await self._make_request("GET", "/getUserProfile", 
                                      auth_token=auth_token, require_auth=True)
    
    async def update_user_profile(self, profile_data: Dict, auth_token: str) -> Optional[Dict]:
        """Create/update user profile - matches frontend flow"""
        return await self._make_request("POST", "/userProfile", 
                                      json_data=profile_data, auth_token=auth_token, require_auth=True)
    
    # ================================
    # ADDRESS MANAGEMENT APIs
    # ================================
    
    async def add_delivery_address(self, address_data: Dict, auth_token: str) -> Optional[Dict]:
        """Add delivery address"""
        return await self._make_request("POST", "/v1/delivery_address", 
                                      json_data=address_data, auth_token=auth_token, require_auth=True)
    
    async def get_delivery_addresses(self, auth_token: str) -> Optional[Dict]:
        """Get delivery addresses (legacy method)"""
        return await self._make_request("GET", "/v1/delivery_address", 
                                      auth_token=auth_token, require_auth=True)
    
    async def get_delivery_addresses_by_user(self, user_id: str) -> Optional[Dict]:
        """Get delivery addresses by user ID - only requires wil-api-key"""
        return await self._make_request("GET", f"/v1/delivery_address/{user_id}")
    
    async def update_delivery_address(self, address_id: str, address_data: Dict, auth_token: str) -> Optional[Dict]:
        """Update delivery address"""
        return await self._make_request("POST", f"/v1/update_delivery_address/{address_id}", 
                                      json_data=address_data, auth_token=auth_token, require_auth=True)
    
    async def delete_delivery_address(self, address_id: str, auth_token: str) -> Optional[Dict]:
        """Delete delivery address"""
        return await self._make_request("DELETE", f"/v1/delete_delivery_address/{address_id}", 
                                      auth_token=auth_token, require_auth=True)
    
    # ================================
    # SUPPORT & TRACKING APIs
    # ================================
    
    async def get_support(self, support_data: Dict, auth_token: str) -> Optional[Dict]:
        """Get support"""
        return await self._make_request("POST", "/v2/get_support", 
                                      json_data=support_data, auth_token=auth_token, require_auth=True)
    
    async def track_order(self, track_data: Dict, auth_token: str) -> Optional[Dict]:
        """Track order"""
        return await self._make_request("POST", "/v2/track", 
                                      json_data=track_data, auth_token=auth_token, require_auth=True)
    
    async def submit_complaint(self, complaint_data: Dict) -> Optional[Dict]:
        """Submit complaint"""
        return await self._make_request("POST", "/v2/complaint", json_data=complaint_data)
    
    async def submit_feedback(self, order_id: str, feedback_data: Dict) -> Optional[Dict]:
        """Submit feedback"""
        return await self._make_request("POST", f"/v2/feedback/{order_id}", json_data=feedback_data)
    
    async def contact_us(self, contact_data: Dict) -> Optional[Dict]:
        """Contact us form"""
        return await self._make_request("POST", "/v2/contact", json_data=contact_data)
    
    # ================================
    # REVIEWS & RATINGS APIs
    # ================================
    
    async def get_item_reviews(self, item_id: str) -> Optional[Dict]:
        """Get item reviews"""
        return await self._make_request("GET", f"/v2/reviews/{item_id}")
    
    async def create_order_review(self, order_id: str, review_data: Dict, auth_token: str) -> Optional[Dict]:
        """Create order review"""
        return await self._make_request("POST", f"/v2/review/{order_id}", 
                                      json_data=review_data, auth_token=auth_token, require_auth=True)
    
    async def get_product_reviews(self, item_id: str, user_id: str) -> Optional[Dict]:
        """Get product reviews for user"""
        return await self._make_request("GET", f"/reviewsV2/{item_id}/{user_id}")
    
    async def create_product_review(self, item_id: str, review_data: Dict, auth_token: str) -> Optional[Dict]:
        """Create product review"""
        return await self._make_request("POST", f"/reviewsV2/{item_id}/", 
                                      json_data=review_data, auth_token=auth_token, require_auth=True)
    

    # ================================
    # OFFERS & COUPONS APIs
    # ================================
    
    async def get_active_offers(self, user_id: str, device_id: str) -> Optional[Dict]:
        """Get active offers for user"""
        return await self._make_request("GET", f"/v2/active_offers/{user_id}/{device_id}")
    
    async def apply_offer(self, offer_id: str, user_id: str, device_id: str) -> Optional[Dict]:
        """Apply offer to user's cart"""
        return await self._make_request("POST", f"/v2/apply_offer/{user_id}/{device_id}", json_data={"offer_id": offer_id})
    
    async def get_applied_offers(self, user_id: str, device_id: str) -> Optional[Dict]:
        """Get applied offers for user"""
        return await self._make_request("GET", f"/v2/getAppliedOffer/{user_id}/{device_id}")
    
    async def clear_offers(self, user_id: str, device_id: str) -> Optional[Dict]:
        """Clear all applied offers"""
        return await self._make_request("DELETE", f"/v2/clearOffer/{user_id}/{device_id}")
    
    async def delete_offer(self, offer_id: str, user_id: str, device_id: str) -> Optional[Dict]:
        """Remove specific applied offer"""
        return await self._make_request("DELETE", f"/v2/deleteOffer/{user_id}/{device_id}/{offer_id}")
    
    # ================================
    # UTILITY METHODS
    # ================================
    
    async def health_check(self) -> bool:
        """Check if the buyer backend is accessible"""
        try:
            response = await self._make_request("GET", "/categories", params={"limit": 1})
            return response is not None and "error" not in response
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    def get_full_order_flow_sequence(self) -> List[str]:
        """Get the complete ONDC order flow sequence"""
        return [
            "1. search_products() - Find products",
            "2. add_to_cart() - Add items to cart",
            "3. select_items() - Create quote request",
            "4. get_select_response() - Get quote response",
            "5. initialize_order() - Initialize order",
            "6. get_init_response() - Get init response",
            "7. create_razorpay_order() - Create payment",
            "8. confirm_order() - Confirm order",
            "9. get_confirm_response() - Get confirmation",
            "10. track_order() - Track order",
            "11. get_order_status() - Check status"
        ]


# Singleton instance for global use
_buyer_client: Optional[BuyerBackendClient] = None


def get_buyer_backend_client(base_url: str = None, api_key: str = None) -> BuyerBackendClient:
    """Get singleton BuyerBackendClient instance"""
    global _buyer_client
    if _buyer_client is None:
        _buyer_client = BuyerBackendClient(base_url, api_key)
    return _buyer_client