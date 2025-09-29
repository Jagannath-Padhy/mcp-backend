#!/usr/bin/env python3
"""
ONDC Shopping Backend API Server
FastAPI server with MCP-Agent integration for frontend applications
"""

import os
import re
import uuid
import json
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager, contextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm import RequestParams

# Import GoogleCompletionTasks AFTER applying monkey patch (done below)
GoogleAugmentedLLM = None
GoogleCompletionTasks = None

# Import Google GenAI for timeout configuration
try:
    from google.genai import Client, types
    GOOGLE_GENAI_AVAILABLE = True
except ImportError:
    GOOGLE_GENAI_AVAILABLE = False
    print("Warning: google.genai not available for timeout configuration")

# Google Client timeout configuration
GOOGLE_CLIENT_TIMEOUT_MS = int(os.getenv('GOOGLE_CLIENT_TIMEOUT_MS', '45000'))  # 45 seconds default

def create_google_client_with_timeout(config):
    """Create Google Client with maximum timeout configuration"""
    if not GOOGLE_GENAI_AVAILABLE:
        raise ImportError("google.genai not available")
    
    # Maximum timeout configuration (60 seconds for all operations)
    timeout_ms = 60000
    
    # Enhanced timeout configuration for all operations
    http_options = types.HttpOptions(
        timeout=timeout_ms
    )
    
    if config and config.vertexai:
        return Client(
            vertexai=config.vertexai,
            project=config.project,
            location=config.location,
            http_options=http_options
        )
    else:
        return Client(
            api_key=config.api_key,
            http_options=http_options
        )

# Google classes will be imported and patched during startup
GoogleAugmentedLLM = None
GoogleCompletionTasks = None

def apply_google_timeout_fix():
    """Apply Google API timeout fix during startup"""
    global GoogleAugmentedLLM, GoogleCompletionTasks
    
    logger.info("🔍 [DEBUG] Applying Google timeout fix during startup...")
    
    if not GOOGLE_GENAI_AVAILABLE:
        logger.error("❌ [TIMEOUT FIX] Google GenAI not available - timeout fix not applied")
        return False
        
    try:
        # Import GoogleCompletionTasks during startup
        from mcp_agent.workflows.llm.augmented_llm_google import GoogleAugmentedLLM as GaLLM, GoogleCompletionTasks as GCT
        GoogleAugmentedLLM = GaLLM
        GoogleCompletionTasks = GCT
        
        logger.info("🔍 [DEBUG] Imported GoogleCompletionTasks successfully")
        
        # Store original method
        original_request_completion_task = GoogleCompletionTasks.request_completion_task
        logger.info(f"🔍 [DEBUG] Stored original method: {original_request_completion_task.__name__}")
        
        @staticmethod
        async def patched_request_completion_task(request):
            """Patched version with timeout configuration"""
            import logging
            logger = logging.getLogger(__name__)
            logger.info("[TIMEOUT FIX] Using patched Google API method with timeout")
            try:
                google_client = create_google_client_with_timeout(request.config)
                payload = request.payload
                response = google_client.models.generate_content(**payload)
                logger.info(f"[TIMEOUT FIX] Google API call completed successfully with {GOOGLE_CLIENT_TIMEOUT_MS}ms timeout")
                return response
            except Exception as e:
                logger.error(f"[TIMEOUT FIX] Google API call failed: {e}")
                raise
        
        # Apply the monkey patch
        GoogleCompletionTasks.request_completion_task = patched_request_completion_task
        logger.info(f"✅ [TIMEOUT FIX] Applied Google Client timeout configuration: {GOOGLE_CLIENT_TIMEOUT_MS}ms")
        logger.info(f"🔍 [DEBUG] New method name: {GoogleCompletionTasks.request_completion_task.__name__}")
        
        # Verify the patch was applied successfully
        if hasattr(GoogleCompletionTasks.request_completion_task, '__name__') and \
           GoogleCompletionTasks.request_completion_task.__name__ == 'patched_request_completion_task':
            logger.info("✅ [VERIFICATION] Monkey patch successfully applied and verified")
        else:
            logger.warning(f"⚠️ [VERIFICATION] Monkey patch may not be applied correctly - method name: {GoogleCompletionTasks.request_completion_task.__name__}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ [TIMEOUT FIX] Failed to apply timeout fix: {e}")
        return False

# Configure logging FIRST
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import MCP SessionService for unified session management
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ondc-shopping-mcp'))
from src.services.session_service import get_session_service

# ============================================================================
# Constants for DRY Code
# ============================================================================

# Context Types
class ContextTypes:
    SEARCH_RESULTS = 'search_results'
    CART_VIEW = 'cart_view'
    CART_UPDATED = 'cart_updated'
    ORDER_CONFIRMED = 'order_confirmed'
    ORDER_DETAILS = 'order_details'
    CHECKOUT_QUOTES = 'checkout_quotes'
    CHECKOUT_FLOW = 'checkout_flow'
    PAYMENT_STATUS = 'payment_status'
    ERROR_STATE = 'error_state'
    SUCCESS_MESSAGE = 'success_message'
    SESSION_INITIALIZED = 'session_initialized'
    DATA_RESPONSE = 'data_response'
    NO_RESULTS = 'no_results'
    INITIALIZATION_REQUIRED = 'initialization_required'

# MCP Server and Tool Names
class MCPConstants:
    SERVER_NAME = 'ondc-shopping'
    TOOLS = {
        'SEARCH_PRODUCTS': 'search_products',
        'ADD_TO_CART': 'add_to_cart',
        'VIEW_CART': 'view_cart',
        'REMOVE_FROM_CART': 'remove_from_cart',
        'UPDATE_CART_QUANTITY': 'update_cart_quantity',
        'CLEAR_CART': 'clear_cart'
    }

# Data Field Names
class DataFields:
    PRODUCTS = 'products'
    CART = 'cart'
    TOTAL_ITEMS = 'total_items'
    ORDER_ID = 'order_id'
    ORDER_DETAILS = 'order_details'
    QUOTE_DATA = 'quote_data'
    QUOTES = 'quotes'
    DELIVERY = 'delivery'
    PAYMENT_ID = 'payment_id'
    PAYMENT_STATUS = 'payment_status'
    TRANSACTION_ID = 'transaction_id'
    ERROR = 'error'
    ERRORS = 'errors'
    SUCCESS = 'success'
    SESSION_ID = 'session_id'
    DEVICE_ID = 'device_id'
    USER_ID = 'user_id'
    NEXT_STEP = 'next_step'
    STAGE = 'stage'
    ITEM_ADDED = 'item_added'
    PRODUCT = 'product'
    STRUCTURED_DATA = '_structured_data'
    CONTEXT_TYPE = '_context_type'

# Initialization Keywords
class InitializationKeywords:
    KEYWORDS = [
        "initialize_shopping", "initialize", "start shopping", "begin shopping",
        "setup session", "create session", "userId", "deviceId"
    ]

# Heavy Fields to Remove
class HeavyFields:
    FIELDS = [
        '_raw', 'detailed_info', 'full_ondc_data',
        'bpp_uri', 'domain', 'contextCity', 'ttl', 'createdAt', 
        'updatedAt', '__v', 'hasCustomisations', 'customisations',
        'bpp_id', 'time', 'fulfillments', 'tags'
    ]
    PREFIXES = ['_']  # Remove ALL underscore fields (debug/internal)

# Validation Messages
class ValidationMessages:
    INITIALIZATION_REQUIRED = (
        "🚨 **Session Initialization Required**\n\n"
        "Before you can start shopping, you must initialize your session with your Himira credentials:\n\n"
        "**Call:** `initialize_shopping(userId='your_user_id', deviceId='your_device_id')`\n\n"
        "**How to get credentials:**\n"
        "• Log into your Himira frontend\n"
        "• Open browser Developer Tools (F12)\n" 
        "• Check localStorage for userId and deviceId\n\n"
        "**Why required:**\n"
        "• Access your existing cart and order history\n"
        "• Proper session isolation between users\n"
        "• Full ONDC shopping functionality\n\n"
        "Please provide your credentials to start shopping!"
    )

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

# Global instances
mcp_app = None
agent = None
llm = None
session_llms = {}  # Session-specific LLM instances with conversation history
tool_results_cache = {}  # Simple cache to capture tool results for API responses

# ============================================================================
# Helper Functions for DRY Code
# ============================================================================

def generate_device_id() -> str:
    """Generate a unique device ID."""
    return f"device_{uuid.uuid4().hex[:8]}"

def generate_session_id() -> str:
    """Generate a unique session ID."""
    return f"session_{uuid.uuid4().hex}"

def extract_session_id_from_message(message: str) -> Optional[str]:
    """Extract session_id from MCP tool calls in message text.
    
    Supports patterns like:
    - search_products(query='turmeric', session_id='session_abc123')
    - add_to_cart(item={...}, session_id="session_xyz789")
    
    Args:
        message: The message text that may contain MCP tool calls
        
    Returns:
        The extracted session_id if found, None otherwise
    """
    # Regex pattern to match session_id parameter in MCP tool calls
    # Matches: session_id='session_xyz' or session_id="session_xyz"
    pattern = r"session_id\s*=\s*['\"]?(session_[a-zA-Z0-9_]+)['\"]?"
    
    match = re.search(pattern, message)
    if match:
        session_id = match.group(1)
        logger.info(f"[SESSION EXTRACTION] Found session_id in message: {session_id}")
        return session_id
    
    logger.debug("[SESSION EXTRACTION] No session_id found in message text")
    return None


def extract_essential_product_fields(product: Dict[str, Any]) -> Dict[str, Any]:
    """Extract essential fields including full image arrays and clean provider info"""
    filtered_product = {
        # Core product data
        'id': product.get('id'),
        'local_id': product.get('local_id'),
        'name': product.get('name'),
        'description': product.get('description'),
        'price': product.get('price'),
        'category': product.get('category'),
        'rating': product.get('rating'),
        
        # Complete image array for gallery/carousel (preserve all images)
        'images': optimize_image_array(product.get('images', [])),
        
        # Clean provider info (using helper function)
        'provider': extract_clean_provider_info(product.get('provider', {})),
        
        # Location/availability (essential only)
        'location': {
            'id': product.get('location', {}).get('id'),
            'serviceability': product.get('location', {}).get('serviceability')
        } if product.get('location') else None,
        
        # Stock/availability
        'availability': product.get('availability_status'),
        'stock': product.get('stock_availability')
    }
    return filtered_product

def optimize_image_array(images: List[Dict], context: str = 'search') -> List[Dict]:
    """Keep full image array but optimize image objects"""
    if not images:
        return []
        
    optimized = []
    for img in images:
        if not isinstance(img, dict):
            continue
            
        optimized_img = {
            'url': img.get('url'),
            'type': img.get('type', 'image'),
        }
        # Remove: alt_text (usually empty), complex metadata
        if optimized_img['url']:  # Only keep images with valid URLs
            optimized.append(optimized_img)
    
    return optimized

def remove_none_values(data: Dict[str, Any]) -> Dict[str, Any]:
    """Remove None values from dictionary recursively"""
    if not isinstance(data, dict):
        return data
        
    cleaned = {k: v for k, v in data.items() if v is not None}
    
    # Clean nested dictionaries
    for key, value in cleaned.items():
        if isinstance(value, dict):
            cleaned[key] = {k: v for k, v in value.items() if v is not None}
            # Remove empty nested dictionaries
            if not cleaned[key]:
                cleaned[key] = None
    
    # Remove keys that became None after cleaning
    return {k: v for k, v in cleaned.items() if v is not None}

def remove_heavy_fields(data: Dict[str, Any]) -> Dict[str, Any]:
    """Remove heavy/bloat fields from data"""
    if not isinstance(data, dict):
        return data
        
    # Define heavy fields to remove
    heavy_fields = HeavyFields.FIELDS
    
    filtered_data = {}
    for key, value in data.items():
        # Skip fields that start with heavy prefixes or are in heavy_fields list
        if any(key.startswith(prefix) for prefix in HeavyFields.PREFIXES) or key in heavy_fields:
            continue
        filtered_data[key] = value
    
    return filtered_data

def extract_minimal_product_info(data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract only essential product fields for frontend rendering, removing data bloat"""
    if not isinstance(data, dict):
        return data
    
    # If this is a products array, filter each product
    if DataFields.PRODUCTS in data and isinstance(data[DataFields.PRODUCTS], list):
        filtered_data = data.copy()
        filtered_products = []
        
        for product in data[DataFields.PRODUCTS]:
            if isinstance(product, dict):
                # Extract essential fields and clean up None values
                filtered_product = extract_essential_product_fields(product)
                cleaned_product = remove_none_values(filtered_product)
                filtered_products.append(cleaned_product)
        
        filtered_data[DataFields.PRODUCTS] = filtered_products
        
        # Remove heavy fields from top-level data
        filtered_data = remove_heavy_fields(filtered_data)
        
        log_performance("Product Filter", {
            "count": len(data[DataFields.PRODUCTS]),
            "original_size": len(str(data)),
            "final_size": len(str(filtered_data)),
            "size_reduction": True
        })
        return filtered_data
    
    # For non-product data, just remove heavy fields
    return remove_heavy_fields(data)

def capture_tool_result_for_session(session_id: str, tool_name: str, tool_result: Any):
    """Capture tool result for later use in API response"""
    try:
        # Extract actual content from CallToolResult if needed
        actual_result = extract_content_from_call_tool_result(tool_result)
        
        # Store the processed result
        tool_results_cache[session_id] = {
            'tool_name': tool_name,
            'result': actual_result,
            'timestamp': datetime.now()
        }
        
        # Standardized logging for tool capture
        if isinstance(actual_result, dict):
            details = {
                'keys': list(actual_result.keys()),
                'content_extracted': actual_result != tool_result
            }
            if DataFields.PRODUCTS in actual_result:
                details['product_count'] = len(actual_result[DataFields.PRODUCTS]) if isinstance(actual_result[DataFields.PRODUCTS], list) else 0
            log_tool_capture(tool_name, "dict", details)
        else:
            log_tool_capture(tool_name, str(type(actual_result)))
            
    except Exception as e:
        logger.warning(f"[Tool Capture] Failed to capture tool result: {e}")

def get_captured_tool_result(session_id: str) -> Optional[Dict[str, Any]]:
    """Get the last captured tool result for a session"""
    return tool_results_cache.get(session_id)

def extract_content_from_call_tool_result(tool_result: Any) -> Any:
    """
    Extract actual content from CallToolResult objects.
    
    Args:
        tool_result: Raw tool result which may be a CallToolResult object
        
    Returns:
        Parsed content or original result if parsing fails
    """
    # Handle mcp.types.CallToolResult objects
    if hasattr(tool_result, 'content') and tool_result.content:
        try:
            # Try to parse the content as JSON
            if isinstance(tool_result.content, list) and len(tool_result.content) > 0:
                # Get the first content item
                content_item = tool_result.content[0]
                if hasattr(content_item, 'text'):
                    return json.loads(content_item.text)
            elif hasattr(tool_result.content, 'text'):
                return json.loads(tool_result.content.text)
        except (json.JSONDecodeError, AttributeError) as e:
            logger.warning(f"[CallToolResult] Failed to parse content: {e}")
    
    # Return original result if no special processing needed
    return tool_result

def process_tool_result_for_structured_data(tool_result: Dict[str, Any], source: str = "unknown") -> tuple[Optional[Dict[str, Any]], Optional[str], bool]:
    """
    Unified tool result processing for both function_response and captured results.
    
    Args:
        tool_result: The tool result dictionary to process
        source: Source of the tool result for logging ("function_response" or "captured")
        
    Returns:
        tuple: (structured_data, context_type, action_required)
    """
    if not isinstance(tool_result, dict):
        log_data_extraction("Tool result validation", source, False, {"error": f"Not dict: {type(tool_result)}"})
        return None, None, False
    
    structured_data = None
    context_type = None
    action_required = False
    
    # Check for products array (search results)
    if DataFields.PRODUCTS in tool_result and tool_result[DataFields.PRODUCTS]:
        structured_data = tool_result
        context_type = ContextTypes.SEARCH_RESULTS
        log_data_extraction("Product extraction", source, True, {"context_type": context_type})
    
    # Check for cart data
    elif DataFields.CART in tool_result or DataFields.TOTAL_ITEMS in tool_result:
        structured_data = tool_result
        context_type = ContextTypes.CART_VIEW
        log_data_extraction("Cart extraction", source, True, {"context_type": context_type})
    
    # Fallback to legacy context detection
    else:
        structured_data = tool_result
        context_type, action_required = determine_context_type(tool_result)
        log_data_extraction("Context detection", source, True, {"context_type": context_type})
    
    return structured_data, context_type, action_required

def log_tool_capture(tool_name: str, result_type: str, details: Dict[str, Any] = None):
    """Standardized logging for tool capture operations"""
    if details:
        if 'keys' in details:
            logger.info(f"[Tool Capture] Captured {tool_name} with keys: {details['keys']}")
        if 'product_count' in details:
            logger.info(f"[Tool Capture] Found {details['product_count']} products in result")
        if 'content_extracted' in details and details['content_extracted']:
            logger.info(f"[Tool Capture] Extracted content from CallToolResult for {tool_name}")
    else:
        logger.info(f"[Tool Capture] Captured {tool_name} result type: {result_type}")

def log_data_extraction(operation: str, source: str, success: bool, details: Dict[str, Any] = None):
    """Standardized logging for data extraction operations"""
    status = "✅" if success else "❌"
    base_msg = f"[Data Processing] {status} {operation} from {source}"
    
    if details:
        if 'context_type' in details:
            logger.info(f"{base_msg}: {details['context_type']}")
        elif 'size_reduction' in details:
            logger.info(f"{base_msg} - {details['size_reduction']}")
        else:
            logger.info(base_msg)
    else:
        logger.info(base_msg)

def log_performance(operation: str, metrics: Dict[str, Any]):
    """Standardized logging for performance metrics"""
    if 'size_reduction' in metrics:
        original_size = metrics.get('original_size', 0)
        final_size = metrics.get('final_size', 0)
        count = metrics.get('count', 0)
        logger.info(f"[{operation}] Filtered {count} items, "
                   f"reduced from {original_size} to {final_size} chars")
    elif 'execution_time' in metrics:
        logger.info(f"[{operation}] Completed in {metrics['execution_time']}ms")

def remove_debug_fields(data: Dict[str, Any]) -> Dict[str, Any]:
    """Remove all debug/internal fields starting with underscore"""
    if not isinstance(data, dict):
        return data
        
    filtered = {}
    for key, value in data.items():
        # Skip ALL underscore fields (debug/internal)
        if key.startswith('_'):
            continue
            
        # Recursively clean nested dictionaries
        if isinstance(value, dict):
            cleaned_value = remove_debug_fields(value)
            if cleaned_value:  # Only keep non-empty objects
                filtered[key] = cleaned_value
        elif isinstance(value, list):
            # Clean lists of dictionaries
            filtered[key] = [remove_debug_fields(item) if isinstance(item, dict) else item for item in value]
        else:
            filtered[key] = value
            
    return filtered

def extract_clean_provider_info(provider: Dict[str, Any]) -> Dict[str, Any]:
    """Extract clean provider info without bloat"""
    if not provider:
        return None
        
    # Get provider name from various possible locations
    provider_name = None
    if 'descriptor' in provider and isinstance(provider['descriptor'], dict):
        provider_name = provider['descriptor'].get('name')
    if not provider_name:
        provider_name = provider.get('name')
        
    return {
        'id': provider.get('id'),
        'name': provider_name,
        'rating': provider.get('rating'),
        'delivery_available': bool(provider.get('contact')),
        # Remove: symbol URLs, long descriptions, complex time objects, fulfillments
    }

def smart_filter_for_frontend(data: Dict[str, Any]) -> Dict[str, Any]:
    """Keep essential frontend data, remove debug/protocol bloat"""
    if not isinstance(data, dict):
        return data
    
    # Step 1: Remove all debug fields (massive space saving)
    cleaned_data = remove_debug_fields(data)
    
    # Step 2: Keep essential structure but filter products
    if DataFields.PRODUCTS in cleaned_data and isinstance(cleaned_data[DataFields.PRODUCTS], list):
        filtered_products = []
        for product in cleaned_data[DataFields.PRODUCTS]:
            if not isinstance(product, dict):
                continue
                
            essential_product = {
                # Core product info (required for frontend)
                'id': product.get('id'),
                'local_id': product.get('local_id'),
                'name': product.get('name'),
                'description': product.get('description'),
                'price': product.get('price'),
                'category': product.get('category'),
                'rating': product.get('rating'),
                
                # Provider info (for store selection)
                'provider': extract_clean_provider_info(product.get('provider', {})),
                
                # Complete images (for galleries/carousels)  
                'images': optimize_image_array(product.get('images', [])),
                
                # Location (for serviceability)
                'location': {
                    'id': product.get('location', {}).get('id'),
                    'serviceability': product.get('location', {}).get('serviceability')
                } if product.get('location') else None,
                
                # Availability (for UI state)
                'availability': product.get('availability_status'),
                'stock': product.get('stock_availability')
            }
            filtered_products.append(remove_none_values(essential_product))
        
        cleaned_data[DataFields.PRODUCTS] = filtered_products
    
    # Step 3: Handle cart data if present
    if DataFields.CART in cleaned_data:
        cleaned_data[DataFields.CART] = remove_debug_fields(cleaned_data[DataFields.CART])
    
    return cleaned_data

@contextmanager
def session_context_file(session_id: str):
    """
    Context manager for session context file lifecycle.
    Ensures proper setup and guaranteed cleanup of session context file.
    
    Args:
        session_id: The session ID to write to the context file
    """
    context_file = None
    try:
        # Create session context file that MCP server can read
        context_dir = os.path.expanduser("~/.ondc-mcp")
        os.makedirs(context_dir, exist_ok=True)
        context_file = os.path.join(context_dir, "chat_session_context.txt")
        
        with open(context_file, 'w') as f:
            f.write(session_id)
        
        logger.info("Set MCP session context file: %s", session_id)
        yield context_file
        
    except Exception as e:
        logger.warning("Failed to set MCP session context file: %s", e)
        yield None
        
    finally:
        # Guaranteed cleanup
        if context_file and os.path.exists(context_file):
            try:
                os.remove(context_file)
                logger.debug("Cleaned up MCP session context file")
            except Exception as cleanup_error:
                logger.debug("Session context file cleanup failed: %s", cleanup_error)


def check_agent_ready():
    """Check if agent is ready and raise appropriate error."""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not ready")
    if not llm:
        raise HTTPException(status_code=503, detail="LLM not ready")

def is_session_initialized(session_id: str) -> bool:
    """Check if session has been properly initialized with credentials using MCP SessionService"""
    session_service = get_session_service()
    
    # CRITICAL: Force fresh read from disk to bypass cache issues
    # The MCP tools and chat API use different SessionService instances
    # So we need to ensure we get the latest data from disk, not stale cache
    session_obj = session_service._load_from_disk(session_id)
    
    # ⚠️ FIX: Check for LLM session ID corruption (hex digits → letters) - SAME AS MCP SERVER
    if not session_obj:
        logger.warning(f"🔧 Session not found, checking for LLM corruption: {session_id}")
        
        # Try to find existing sessions with similar pattern
        import os
        session_files = os.listdir(session_service.storage_path)
        corrected_id = None
        
        for session_file in session_files:
            if session_file.endswith('.json'):
                file_session_id = session_file[:-5]  # Remove .json
                
                # Check if this could be the corrupted version
                if len(file_session_id) == len(session_id):
                    # Compare character by character, allowing hex digit corrections
                    diff_count = 0
                    correction_made = False
                    
                    for i, (orig, corrupted) in enumerate(zip(file_session_id, session_id)):
                        if orig != corrupted:
                            diff_count += 1
                            # Check common hex corruption patterns
                            if (orig in '0123456789abcdef' and corrupted in 'G-Z' and 
                                abs(ord(orig) - ord(corrupted.lower())) <= 10):
                                correction_made = True
                    
                    # If only a few characters differ and they look like hex corruption
                    if diff_count <= 3 and correction_made:
                        corrected_id = file_session_id
                        logger.info(f"🔧 API: Found potential session match: {corrected_id}")
                        break
        
        # Try loading with corrected ID
        if corrected_id:
            logger.info(f"🔧 API: Attempting to load corrected session: {corrected_id}")
            session_obj = session_service._load_from_disk(corrected_id)
            if session_obj:
                logger.info(f"🔧 API: ✅ Successfully loaded corrected session: {corrected_id}")
    
    if not session_obj:
        logger.info(f"🔍 Session {session_id} not found in MCP SessionService")
        return False
    
    # Use the same validation logic as MCP server
    has_auth = session_obj.user_authenticated
    has_user_id = bool(session_obj.user_id)
    has_device_id = bool(session_obj.device_id)
    
    # Debug: log the actual session object values
    logger.info(f"🔍 Session {session_id} check - authenticated: {has_auth}, user_id: {has_user_id}, device_id: {has_device_id}")
    logger.debug(f"🔍 Session raw values - user_authenticated: '{session_obj.user_authenticated}', user_id: '{session_obj.user_id}', device_id: '{session_obj.device_id}'")
    
    # Session is initialized if it has authentication and both user_id and device_id
    return (has_auth and has_user_id and has_device_id)

def is_initialization_request(message: str) -> bool:
    """Check if the message is requesting initialization"""
    message_lower = message.lower()
    return any(keyword in message_lower for keyword in InitializationKeywords.KEYWORDS)

def get_initialization_required_response() -> str:
    """Get the standard response for initialization requirement"""
    return ValidationMessages.INITIALIZATION_REQUIRED

def determine_context_type(tool_result: Dict[str, Any]) -> tuple[str, bool]:
    """Determine context type and action requirement from tool result.
    
    Enhanced with granular context detection for better frontend routing.
    
    Returns:
        tuple: (context_type, action_required)
    """
    if not isinstance(tool_result, dict):
        return None, False
    
    # 🚀 ENHANCED: Granular context detection
    
    # Product search results
    if DataFields.PRODUCTS in tool_result:
        products = tool_result[DataFields.PRODUCTS]
        if products and len(products) > 0:
            return ContextTypes.SEARCH_RESULTS, False
        else:
            return ContextTypes.NO_RESULTS, False
    
    # Cart operations - differentiate between view and update
    if DataFields.CART in tool_result or 'cart_summary' in tool_result:
        return ContextTypes.CART_VIEW, False
    
    # Single item operations (add to cart)
    if DataFields.ITEM_ADDED in tool_result or DataFields.PRODUCT in tool_result:
        return ContextTypes.CART_UPDATED, False
    
    # Order operations - differentiate between confirmed and details
    if DataFields.ORDER_ID in tool_result:
        return ContextTypes.ORDER_CONFIRMED, False
    elif DataFields.ORDER_DETAILS in tool_result:
        return ContextTypes.ORDER_DETAILS, False
    
    # Checkout flow - differentiate stages
    if DataFields.QUOTE_DATA in tool_result or DataFields.QUOTES in tool_result or DataFields.DELIVERY in tool_result:
        return ContextTypes.CHECKOUT_QUOTES, True
    elif DataFields.NEXT_STEP in tool_result or DataFields.STAGE in tool_result:
        return ContextTypes.CHECKOUT_FLOW, True
    
    # Payment operations
    if any(key in tool_result for key in [DataFields.PAYMENT_ID, DataFields.PAYMENT_STATUS, DataFields.TRANSACTION_ID]):
        return ContextTypes.PAYMENT_STATUS, False
    
    # Error states
    if DataFields.ERROR in tool_result or DataFields.ERRORS in tool_result:
        return ContextTypes.ERROR_STATE, False
    
    # Success messages
    if DataFields.SUCCESS in tool_result and tool_result.get(DataFields.SUCCESS) is True:
        return ContextTypes.SUCCESS_MESSAGE, False
    
    # Session/initialization responses
    if DataFields.SESSION_ID in tool_result and DataFields.DEVICE_ID in tool_result:
        return ContextTypes.SESSION_INITIALIZED, False
    
    # Default fallback for any data
    if tool_result:
        return ContextTypes.DATA_RESPONSE, False
    
    return None, False

def generate_contextual_response(structured_data: Dict[str, Any], context_type: str, user_message: str) -> str:
    """Generate meaningful conversational responses from structured data when LLM response is empty"""
    
    try:
        # Handle different context types with appropriate responses
        if context_type == ContextTypes.SEARCH_RESULTS:
            return generate_search_response(structured_data)
        elif context_type == ContextTypes.CART_VIEW:
            return generate_cart_response(structured_data)
        elif context_type == ContextTypes.CART_UPDATED:
            return generate_cart_updated_response(structured_data)
        elif context_type == ContextTypes.CHECKOUT_QUOTES:
            return generate_checkout_quotes_response(structured_data)
        elif context_type == ContextTypes.ORDER_CONFIRMED:
            return generate_order_confirmed_response(structured_data)
        elif context_type == ContextTypes.SUCCESS_MESSAGE:
            return generate_success_response(structured_data)
        elif context_type == ContextTypes.SESSION_INITIALIZED:
            return generate_session_initialized_response(structured_data)
        else:
            # Generic fallback
            return generate_generic_response(structured_data, user_message)
    except Exception as e:
        logger.error(f"[SMART FALLBACK] Error generating contextual response: {e}")
        return "I've processed your request successfully. Please check the data below for details."

def generate_search_response(data: Dict[str, Any]) -> str:
    """Generate response for search results"""
    if DataFields.PRODUCTS in data:
        products = data[DataFields.PRODUCTS]
        count = len(products) if isinstance(products, list) else 0
        
        if count == 0:
            return "I couldn't find any products matching your search. Please try different keywords."
        elif count == 1:
            product = products[0]
            name = product.get('name', 'Product')
            price = product.get('price', 'N/A')
            provider = product.get('provider', {}).get('name', 'Unknown Store')
            return f"I found **{name}** for ₹{price} from {provider}. Would you like to add it to your cart?"
        else:
            return f"I found {count} products matching your search. Here are the results:"
    
    return "I've completed your search. Please check the results below."

def generate_cart_response(data: Dict[str, Any]) -> str:
    """Generate response for cart view"""
    if DataFields.CART in data:
        cart = data[DataFields.CART]
        if isinstance(cart, dict):
            total_items = cart.get(DataFields.TOTAL_ITEMS, 0)
            if total_items == 0:
                return "Your cart is currently empty. Start shopping by searching for products!"
            else:
                return f"Here's your cart with {total_items} item(s). Ready to proceed to checkout?"
    
    return "Here's your current cart:"

def generate_cart_updated_response(data: Dict[str, Any]) -> str:
    """Generate response for cart updates"""
    if DataFields.ITEM_ADDED in data:
        return "Great! I've added the item to your cart."
    elif "quantity" in str(data).lower():
        return "I've updated the quantity in your cart."
    else:
        return "Your cart has been updated successfully."

def generate_checkout_quotes_response(data: Dict[str, Any]) -> str:
    """Generate response for delivery quotes"""
    if DataFields.QUOTE_DATA in data:
        quote_data = data[DataFields.QUOTE_DATA]
        total_value = quote_data.get('total_value', 0)
        total_delivery = quote_data.get('total_delivery', 0)
        
        response = f"Great! I've gotten delivery quotes for your cart.\n\n"
        response += f"**Total Items:** ₹{total_value}\n"
        response += f"**Delivery:** ₹{total_delivery}\n"
        response += f"**Grand Total:** ₹{total_value + total_delivery}\n\n"
        response += "Ready to proceed with order initialization! Please provide your customer details."
        
        return response
    
    return "I've retrieved delivery quotes for your location. Ready to proceed with checkout!"

def generate_order_confirmed_response(data: Dict[str, Any]) -> str:
    """Generate response for order confirmation"""
    if DataFields.ORDER_ID in data:
        order_id = data[DataFields.ORDER_ID]
        return f"🎉 Order confirmed! Your order ID is **{order_id}**. You'll receive tracking updates soon."
    
    return "🎉 Your order has been confirmed successfully!"

def generate_success_response(data: Dict[str, Any]) -> str:
    """Generate response for success messages"""
    if 'message' in data:
        return data['message']
    elif 'next_action' in data:
        next_action = data['next_action']
        if next_action == 'start_shopping':
            return "Session initialized successfully! Ready to start shopping. What would you like to find?"
    
    return "Operation completed successfully!"

def generate_session_initialized_response(data: Dict[str, Any]) -> str:
    """Generate response for session initialization"""
    user_id = data.get(DataFields.USER_ID, 'User')
    session_id = data.get(DataFields.SESSION_ID, 'Unknown')
    
    response = f"✅ **Session Ready!**\n\n"
    response += f"**Session ID:** `{session_id}`\n"
    response += f"**User ID:** `{user_id}`\n\n"
    response += "🛍️ **Ready to shop!** What would you like to find?"
    
    return response

def generate_generic_response(data: Dict[str, Any], user_message: str) -> str:
    """Generate generic response based on data content"""
    if DataFields.SUCCESS in data and data[DataFields.SUCCESS]:
        return "Operation completed successfully!"
    elif DataFields.ERROR in data:
        return f"There was an issue: {data[DataFields.ERROR]}"
    elif 'message' in data:
        return data['message']
    else:
        return "I've processed your request. Please check the details below."

def create_ai_context_summary(data: Dict[str, Any]) -> Dict[str, Any]:
    """Create highly compressed context summary for AI to reduce token usage (90% reduction)"""
    if not data:
        return {}
    
    summary = {}
    
    # Compress product data (major token saver)
    if DataFields.PRODUCTS in data:
        products = data[DataFields.PRODUCTS]
        if isinstance(products, list) and len(products) > 0:
            # Limit to 3 products max for AI context
            limited_products = products[:3]
            summary['products_summary'] = []
            
            for product in limited_products:
                # Only essential fields for AI response generation
                product_summary = {
                    'name': product.get('name', 'Unknown'),
                    'price': product.get('price', 0),
                    'provider': product.get('provider', {}).get('name', 'Unknown Store')
                }
                summary['products_summary'].append(product_summary)
            
            summary['total_products'] = len(products)
            summary['showing_summary'] = f"Showing {len(limited_products)} of {len(products)} products"
    
    # Compress cart data
    if DataFields.CART in data:
        cart = data[DataFields.CART]
        if isinstance(cart, dict):
            summary['cart_summary'] = {
                'items': cart.get('total_items', 0),
                'total': cart.get('total_value', 0)
            }
    
    # Include essential metadata
    if 'success' in data:
        summary['success'] = data['success']
    if 'message' in data:
        summary['message'] = data['message'][:100]  # Truncate long messages
    
    return summary

async def get_session_llm(session_id: str):
    """Get or create a session-specific LLM with conversation history"""
    if session_id not in session_llms:
        if not agent:
            raise HTTPException(status_code=503, detail="Agent not ready")
        
        # Create a new LLM instance for this session
        session_llm = await agent.attach_llm(GoogleAugmentedLLM)
        session_llms[session_id] = session_llm
        logger.info(f"Created new session LLM for session: {session_id}")
    
    return session_llms[session_id]

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    global mcp_app, agent, llm
    
    logger.info("🚀 Starting ONDC Shopping Backend API...")
    
    # Apply Google API timeout fix FIRST - before any Google class usage
    timeout_fix_applied = apply_google_timeout_fix()
    if timeout_fix_applied:
        logger.info("✅ Google API timeout fix applied successfully")
        
        # Additional verification that GoogleAugmentedLLM is available after patch
        if GoogleAugmentedLLM is not None:
            logger.info("✅ GoogleAugmentedLLM class is now available for agent attachment")
        else:
            logger.error("❌ GoogleAugmentedLLM is None after timeout fix - this may cause issues")
    else:
        logger.warning("⚠️ Google API timeout fix not applied - may cause timeout issues")
    
    try:
        # Initialize MCP App - uses config file for server configuration
        mcp_app = MCPApp(
            name="ondc_backend",
            settings="/app/mcp_agent.config.yaml"
        )
        # Initialize MCP app context
        async with mcp_app.run():
            
            # Create agent connected to MCP server via STDIO
            agent = Agent(
                name="shopping_assistant",
                instruction="""🤖 INTELLIGENT ONDC Shopping Assistant - Tool Chaining Expert

MANDATORY FIRST STEP: initialize_shopping(userId, deviceId) before ANY operations.

🚀 INTELLIGENT WORKFLOW PATTERNS:

**Smart Tool Chaining** - Chain tools automatically based on user intent:
1. "add [product] to cart" → auto-search if needed → present options OR add directly
2. "buy [product]" → search → add → view cart → suggest checkout
3. "I want [product]" → search → show options → guide to cart
4. "checkout" → view cart → select_items_for_order → initialize_order

**Natural Language Processing:**
- "add kiwi jam" → add_to_cart(item={"name": "kiwi jam"}, session_id=session) → auto-searches → presents options
- "add kiwi jam to cart" → add_to_cart(item={"name": "kiwi jam"}, session_id=session)
- "show my cart" → view_cart(session_id=session)
- "I want organic rice" → search_products(query="organic rice", session_id=session) → present options
- "buy the first one" → add_to_cart(item=from_last_search[0], session_id=session)

🔧 TOOL USAGE INTELLIGENCE:

**Smart Add-to-Cart Pattern:**
- When user says "add X to cart" → DIRECTLY call add_to_cart(item={"name": "X"}, session_id=session)
- The tool will auto-search if needed and handle the workflow intelligently
- Alternative: search_products(query=X) first → then add_to_cart(selected_product)
- Both patterns work - choose based on user preference

**Context-Aware Actions:**
- Track search_history and cart_state for intelligent decisions
- Use recent search results for smart product selection
- Suggest next logical steps in the journey

**Multi-Tool Workflows:**
- Search → Add → View → Checkout (full journey)
- Search → Add (quick addition)
- View → Select → Initialize (checkout flow)

📋 AVAILABLE TOOLS & PATTERNS:

**Session Management:**
- initialize_shopping(userId, deviceId) - MANDATORY FIRST
- get_session_info(session_id) - session status

**Search & Discovery:**
- search_products(query, session_id) - find products by name/description
- advanced_search(filters, session_id) - multi-criteria filtering
- browse_categories(session_id) - explore categories

**Cart Management:**
- add_to_cart(item, quantity, session_id) - 🤖 INTELLIGENT: accepts full products OR partial data (auto-searches from recent results)
- view_cart(session_id) - show current cart contents
- clear_cart(session_id) - empty cart
- get_cart_total(session_id) - calculate totals

**Checkout Journey:**
- select_items_for_order(delivery_city, delivery_state, delivery_pincode, session_id)
- initialize_order(customer_name, delivery_address, phone, email, session_id)
- create_payment(payment_method, amount, session_id)
- confirm_order(session_id, payment_status)

🎯 INTELLIGENT RESPONSES:
- Search: "Found [N] products. [Top result] ₹[price] from [store]. Add to cart?"
- Smart Add: "🤖 Smart Selection: [Product] (₹[price]) from recent search. Added to cart!"
- Cart View: "Cart: [N] items, ₹[total]. [List items]. Ready to checkout?"
- Workflow: Always suggest logical next steps

RULES:
- Use session_id in ALL tool calls
- Chain tools intelligently based on user intent
- Provide helpful, contextual responses
- Guide users through the complete journey""",
                server_names=[MCPConstants.SERVER_NAME]  # Connects to our MCP server
            )
            
            await agent.__aenter__()
            
            # Attach Gemini LLM
            llm = await agent.attach_llm(GoogleAugmentedLLM)
            
            logger.info("✅ Backend API ready with MCP-Agent!")
            
            yield
        
    except Exception as e:
        logger.error(f"❌ Startup error: {e}")
        raise
    finally:
        # Cleanup
        if agent:
            await agent.__aexit__(None, None, None)
        if mcp_app:
            await mcp_app.cleanup()
        logger.info("👋 Backend API shutdown complete")

# Create FastAPI app
app = FastAPI(
    title="ONDC Shopping Backend API",
    description="Backend API for ONDC shopping with AI-powered assistance",
    version="1.0.0",
    lifespan=lifespan
)

# Add rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class ChatRequest(BaseModel):
    message: str = Field(..., description="User's message")
    session_id: Optional[str] = Field(None, description="Session ID")
    device_id: Optional[str] = Field(None, description="Device ID")

class ChatResponse(BaseModel):
    response: str
    session_id: str
    device_id: str
    timestamp: datetime
    data: Optional[Any] = None  # Structured data from tool results
    context_type: Optional[str] = None  # Type of data (products/cart/order/checkout)
    action_required: Optional[bool] = False  # If user action is needed
    tools_called: Optional[List[str]] = None  # List of MCP tools called during this conversation
    agent_thoughts: Optional[str] = None  # Agent reasoning process (from Gemini thinking)

class SessionCreateRequest(BaseModel):
    device_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = {}

class SessionResponse(BaseModel):
    session_id: str
    device_id: str
    created_at: datetime
    last_activity: datetime
    metadata: Dict[str, Any]

class SearchRequest(BaseModel):
    query: str
    filters: Optional[Dict[str, Any]] = {}
    limit: int = 20

class CartRequest(BaseModel):
    action: str  # add, remove, update, view
    item: Optional[Dict[str, Any]] = None
    quantity: Optional[int] = 1

# Health check
@app.get("/health")
@limiter.limit("60/minute")
async def health_check(request: Request):
    """Health check endpoint"""
    # Count active sessions from SessionService
    session_service = get_session_service()
    active_sessions = len([f for f in session_service.storage_path.glob("*.json")])
    
    return {
        "status": "healthy" if llm else "initializing",
        "timestamp": datetime.now(),
        "agent_ready": llm is not None,
        "active_sessions": active_sessions,
        "enhanced_features": {
            "tool_call_detection": True,
            "parallel_extraction": True,
            "performance_monitoring": True
        }
    }

# Performance metrics endpoint
@app.get("/api/v1/metrics/performance")
@limiter.limit("30/minute")
async def get_performance_metrics(request: Request):
    """Get tool call performance metrics and optimization insights"""
    if not llm:
        raise HTTPException(status_code=503, detail="LLM not ready")
    
    try:
        # Get metrics from the enhanced LLM
        metrics = llm.get_performance_metrics()
        
        # Add system-wide metrics
        session_service = get_session_service()
        system_metrics = {
            "system_status": {
                "active_sessions": len(session_llms),
                "total_session_files": len([f for f in session_service.storage_path.glob("*.json")]),
                "api_status": "operational"
            },
            "llm_metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }
        
        return system_metrics
        
    except Exception as e:
        logger.error(f"Error retrieving performance metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")

# Reset metrics endpoint (useful for testing)
@app.post("/api/v1/metrics/reset")
@limiter.limit("5/minute")
async def reset_performance_metrics(request: Request):
    """Reset performance metrics (useful for testing and fresh starts)"""
    if not llm:
        raise HTTPException(status_code=503, detail="LLM not ready")
    
    try:
        # Reset metrics in the main LLM
        await llm.reset_metrics()
        
        # Reset metrics in all session LLMs
        reset_count = 0
        for session_id, session_llm in session_llms.items():
            await session_llm.reset_metrics()
            reset_count += 1
        
        logger.info(f"Performance metrics reset for main LLM and {reset_count} session LLMs")
        
        return {
            "message": "Performance metrics reset successfully",
            "reset_targets": {
                "main_llm": True,
                "session_llms_count": reset_count
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error resetting performance metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reset metrics: {str(e)}")

# Session management
@app.post("/api/v1/sessions", response_model=SessionResponse)
@limiter.limit("10/minute")
async def create_session(request: Request, session_req: SessionCreateRequest):
    """Create a new shopping session"""
    session_service = get_session_service()
    session_id = generate_session_id()
    device_id = session_req.device_id or generate_device_id()
    
    # Create session using SessionService
    session_obj = session_service.create_with_id(session_id)
    session_obj.device_id = device_id
    session_service.update(session_obj)
    
    logger.info(f"Created session: {session_id}")
    
    return SessionResponse(
        session_id=session_obj.session_id,
        device_id=session_obj.device_id,
        created_at=session_obj.created_at,
        last_activity=session_obj.last_accessed,
        metadata=session_req.metadata or {}
    )

@app.get("/api/v1/sessions/{session_id}", response_model=SessionResponse)
@limiter.limit("30/minute")
async def get_session(request: Request, session_id: str):
    """Get session information"""
    session_service = get_session_service()
    session_obj = session_service.get(session_id)
    
    if not session_obj:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return SessionResponse(
        session_id=session_obj.session_id,
        device_id=session_obj.device_id,
        created_at=session_obj.created_at,
        last_activity=session_obj.last_accessed,
        metadata={}  # Could extract from session if needed
    )

@app.delete("/api/v1/sessions/{session_id}")
@limiter.limit("20/minute")
async def delete_session(request: Request, session_id: str):
    """End a shopping session"""
    session_service = get_session_service()
    
    if not session_service.get(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Clean up session data and LLM
    session_service.delete(session_id)
    if session_id in session_llms:
        del session_llms[session_id]
    
    logger.info(f"Deleted session and LLM: {session_id}")
    
    return {"message": "Session deleted"}

# Main chat endpoint
@app.post("/api/v1/chat", response_model=ChatResponse)
@limiter.limit("20/minute")
async def chat(request: Request, chat_req: ChatRequest):
    """Chat with the shopping assistant - Enforces mandatory initialization"""
    
    check_agent_ready()
    
    # FIXED: Extract session_id from MCP tool calls FIRST before validation
    # Priority: JSON body session_id → extract from message → generate new  
    if chat_req.session_id:
        actual_session_id = chat_req.session_id
        logger.info(f"[SESSION ID] Using session_id from JSON body: {actual_session_id}")
    else:
        extracted_session_id = extract_session_id_from_message(chat_req.message)
        if extracted_session_id:
            actual_session_id = extracted_session_id
            logger.info(f"[SESSION ID] Using session_id extracted from message: {actual_session_id}")
        else:
            actual_session_id = None
            logger.info("[SESSION ID] No session_id found - will generate new for initialization")
    
    # Check if this is a new session without credentials
    if not actual_session_id:
        # New session - check if this is an initialization request
        if not is_initialization_request(chat_req.message):
            # Block all non-initialization requests for new sessions
            return ChatResponse(
                response=get_initialization_required_response(),
                session_id="",  # No session until initialized
                device_id="",   # No device until initialized
                timestamp=datetime.now(),
                data=None,
                context_type=ContextTypes.INITIALIZATION_REQUIRED,
                action_required=True
            )
    else:
        # Existing session - check if it's properly initialized
        # IMPORTANT: Only validate for non-initialization requests
        # Initialization requests need to be processed first to create the session data
        if not is_initialization_request(chat_req.message):
            if not is_session_initialized(actual_session_id):
                # Session exists but not initialized - require initialization
                return ChatResponse(
                    response=get_initialization_required_response(),
                    session_id=actual_session_id,
                    device_id=chat_req.device_id or "",
                    timestamp=datetime.now(),
                    data=None,
                    context_type=ContextTypes.INITIALIZATION_REQUIRED, 
                    action_required=True
                )
    
    # Generate IDs if not provided (only for initialization requests)
    device_id = chat_req.device_id or generate_device_id()
    session_id = actual_session_id or generate_session_id()
    
    # FIXED: API server should ONLY READ MCP sessions, never write
    # Session creation and management is handled exclusively by MCP tools
    # This prevents race conditions and architectural violations
    
    try:
        # CRITICAL: Set file-based session context for MCP tools
        # This ensures all MCP tool calls use the correct chat API session
        # Uses file-based approach since MCP server runs in separate process
        # Get session-specific LLM with conversation history
        session_llm = await get_session_llm(session_id)
        
        # Debug: Verify enhanced LLM is being used
        logger.debug("🔍 Using LLM type: %s for session: %s", type(session_llm).__name__, session_id)
        
        # Enhance message with context (simplified, no need for manual injection now)
        # ⚠️ FIX: Don't put session_id directly in message - LLM corrupts hex patterns like d0fa→dof_a
        # Session context is handled by session_context_file() below, not by enhanced message
        enhanced_message = f"[Device: {device_id}] {chat_req.message}"
        
        # Configure request parameters with AGENT REASONING TRANSPARENCY
        request_params = RequestParams(
            max_iterations=20,         # MAXIMUM: High iterations for all complex operations
            use_history=True,
            temperature=0.7,
            maxTokens=6000,           # MAXIMUM: High token limit for all responses
            parallel_tool_calls=True,
            model="gemini-2.5-pro",
            # 🧠 ENABLE AGENT REASONING VISIBILITY (2025 Native Features)
            thinking_budget=-1,        # Dynamic thinking based on request complexity
            include_thoughts=True      # Return raw agent thoughts and reasoning process
        )
        
        # Clear any previous tool results for this session
        tool_results_cache.pop(session_id, None)
        
        # Initialize tools called tracking for this session
        session_tools_called = []
        
        # Temporarily patch agent to capture tool results
        original_call_tool = None
        if agent and hasattr(agent, 'call_tool'):
            original_call_tool = agent.call_tool
            
            async def capturing_call_tool(*args, **kwargs):
                # Call original tool method
                result = await original_call_tool(*args, **kwargs)
                
                # ENHANCED DEBUG: Detailed analysis of call_tool signature
                logger.info(f"[TOOL DEBUG] call_tool invoked with {len(args)} args, {len(kwargs)} kwargs")
                for i, arg in enumerate(args):
                    logger.info(f"[TOOL DEBUG] args[{i}]: {type(arg).__name__} = {str(arg)[:100]}...")
                for key, value in kwargs.items():
                    logger.info(f"[TOOL DEBUG] kwargs[{key}]: {type(value).__name__} = {str(value)[:100]}...")
                
                # Extract tool name using 2025 mcp-agent framework signature
                # call_tool(name: str, arguments: dict, server_name: str) 
                tool_name = 'unknown_tool'
                
                # Strategy 1: Tool name in kwargs (most reliable for 2025)
                if 'name' in kwargs:
                    tool_name = str(kwargs['name'])
                    logger.info(f"[TOOL EXTRACT] ✅ 2025 kwargs name: {tool_name}")
                # Strategy 2: First positional argument is tool name
                elif len(args) >= 1 and isinstance(args[0], str):
                    tool_name = args[0]
                    logger.info(f"[TOOL EXTRACT] ✅ 2025 args[0] string: {tool_name}")
                else:
                    logger.warning(f"[TOOL EXTRACT] ❌ Could not extract tool name from call_tool parameters")
                
                # Clean tool name (remove server prefixes)
                if tool_name.startswith('mcp-'):
                    tool_name = tool_name[4:]
                
                logger.info(f"[TOOL EXTRACT] Final tool name: {tool_name}")
                
                capture_tool_result_for_session(session_id, tool_name, result)
                
                # Track tool calls for transparency
                session_tools_called.append(tool_name)
                logger.info(f"[TOOL TRANSPARENCY] Registered tool call: {tool_name}")
                
                return result
            
            # Temporarily replace the method
            agent.call_tool = capturing_call_tool
        
        # Use context manager for LLM generation that requires MCP tools
        with session_context_file(session_id):
            try:
                # Get AI response using full conversation loop
                contents = await session_llm.generate(
                    message=enhanced_message,
                    request_params=request_params
                )
            except Exception as llm_error:
                # Enhanced Google API error handling
                error_str = str(llm_error)
                logger.error(f"[LLM ERROR] {error_str}")
                
                # Detect Google API quota exhaustion
                if "429" in error_str and "RESOURCE_EXHAUSTED" in error_str:
                    logger.error("[QUOTA EXHAUSTED] Google API quota limit reached")
                    
                    # Extract retry delay if available
                    retry_delay = "unknown"
                    if "retry in" in error_str.lower():
                        import re
                        delay_match = re.search(r'retry in (\d+\.?\d*)s', error_str)
                        if delay_match:
                            retry_delay = delay_match.group(1) + " seconds"
                    
                    # Return user-friendly quota error response
                    return ChatResponse(
                        response=f"⚠️ **Service Temporarily Busy**\n\nThe AI service is currently experiencing high demand. Please try again in {retry_delay}.\n\n**What happened:** Google AI API quota limit reached\n**Solution:** Wait a moment and retry your request\n\n*Your session and cart are preserved.*",
                        session_id=session_id,
                        device_id=device_id,
                        timestamp=datetime.now(),
                        data=None,
                        context_type="error_state",
                        action_required=False
                    )
                
                # Detect rate limiting
                elif "rate limit" in error_str.lower() or "too many requests" in error_str.lower():
                    logger.error("[RATE LIMITED] Google API rate limit exceeded")
                    
                    return ChatResponse(
                        response="⚠️ **Too Many Requests**\n\nPlease slow down and try again in a few seconds.\n\n*Your session and cart are preserved.*",
                        session_id=session_id,
                        device_id=device_id,
                        timestamp=datetime.now(),
                        data=None,
                        context_type="error_state",
                        action_required=False
                    )
                
                # Detect service unavailable
                elif "503" in error_str and "UNAVAILABLE" in error_str:
                    logger.error("[SERVICE UNAVAILABLE] Google AI service temporarily unavailable")
                    
                    return ChatResponse(
                        response="⚠️ **AI Service Temporarily Unavailable**\n\nGoogle's AI service is currently down. Please try again in a few minutes.\n\n*Your session and cart are preserved.*",
                        session_id=session_id,
                        device_id=device_id,
                        timestamp=datetime.now(),
                        data=None,
                        context_type="error_state",
                        action_required=False
                    )
                
                # Other Google API errors
                elif "google" in error_str.lower() or "gemini" in error_str.lower():
                    logger.error(f"[GOOGLE API ERROR] {error_str}")
                    
                    return ChatResponse(
                        response="⚠️ **AI Service Error**\n\nThe AI service encountered an issue. Please try again.\n\n*Your session and cart are preserved.*",
                        session_id=session_id,
                        device_id=device_id,
                        timestamp=datetime.now(),
                        data=None,
                        context_type="error_state",
                        action_required=False
                    )
                
                # Unknown errors - re-raise for general handling
                else:
                    raise llm_error
                    
            finally:
                # Restore original method
                if original_call_tool:
                    agent.call_tool = original_call_tool
        
        # Process the response using proper MCP integration
        response_text = ""
        structured_data = None
        context_type = None
        action_required = False
        agent_thoughts = None  # Extract agent reasoning from Gemini thinking
        
        # The GoogleAugmentedLLM should handle tool calling automatically
        # We just need to extract the final response and any structured data
        logger.debug(f"[Chat API] Processing {len(contents) if contents else 0} content items")
        
        if contents and len(contents) > 0:
            for content_idx, content in enumerate(contents):
                logger.debug(f"[Chat API] Processing content {content_idx}, parts: {len(content.parts) if content.parts else 0}")
                if content.parts:
                    for part_idx, part in enumerate(content.parts):
                        logger.debug(f"[Chat API] Processing part {part_idx}, type: {type(part)}")
                        if hasattr(part, 'text') and part.text:
                            # Check if this text part contains thinking content (2025 Gemini format)
                            if part.text.startswith('<thinking>') or 'Thinking:' in part.text[:100] or part.text.startswith('🤔'):
                                agent_thoughts = part.text
                                logger.info(f"[AGENT THOUGHTS] ✅ 2025 thinking in text: {len(agent_thoughts)} chars")
                            else:
                                response_text += part.text
                                logger.debug(f"[Chat API] Added text part: {len(part.text)} chars")
                        elif hasattr(part, 'thought') and part.thought:
                            # Extract agent thinking/reasoning from Gemini 2.5 (legacy attribute)
                            agent_thoughts = part.thought
                            logger.info(f"[AGENT THOUGHTS] ✅ 2025 thought attribute: {len(agent_thoughts)} chars")
                        elif hasattr(part, 'thinking_content') and part.thinking_content:
                            # Extract agent thinking/reasoning from Gemini 2.5 (alternative attribute)
                            agent_thoughts = part.thinking_content
                            logger.info(f"[AGENT THOUGHTS] ✅ 2025 thinking_content: {len(agent_thoughts)} chars")
                        elif hasattr(part, 'function_response'):
                            logger.debug("[Chat API] Found function_response part")
                            # Handle tool execution results if present
                            if hasattr(part.function_response, 'content'):
                                logger.debug(f"[Chat API] function_response has content: {len(part.function_response.content)} chars")
                                try:
                                    # Parse tool result as JSON if possible
                                    tool_result = json.loads(part.function_response.content)
                                    logger.debug(f"[Chat API] Parsed tool result with keys: {list(tool_result.keys())}")
                                    
                                    # 🚀 ENHANCED: Extract structured data from MCP response
                                    if DataFields.STRUCTURED_DATA in tool_result:
                                        structured_data = tool_result[DataFields.STRUCTURED_DATA]
                                        log_data_extraction("MCP structured data", "function_response", True, 
                                                          {"keys": list(structured_data.keys()) if structured_data else None})
                                    
                                    # 🚀 ENHANCED: Extract context type from MCP response
                                    if DataFields.CONTEXT_TYPE in tool_result:
                                        context_type = tool_result[DataFields.CONTEXT_TYPE]
                                        log_data_extraction("MCP context type", "function_response", True, 
                                                          {"context_type": context_type})
                                    
                                    
                                    # Extract structured data directly from tool result
                                    if not structured_data:
                                        structured_data, context_type, action_required = process_tool_result_for_structured_data(
                                            tool_result, source="function_response"
                                        )
                                    
                                except (json.JSONDecodeError, AttributeError) as e:
                                    # If not JSON or no content attribute, just add to response text
                                    logger.warning(f"[Chat API] Failed to parse function_response as JSON: {e}")
                                    response_text += str(part.function_response)
                            else:
                                logger.debug("[Chat API] function_response has no content attribute")
                        else:
                            logger.debug(f"[Chat API] Unknown part type: {type(part)}")
        
        # Clean up response text - remove any "None" prefix
        if response_text.startswith("None"):
            response_text = response_text[4:]
        
        # 🔍 ENHANCED LOGGING: Track response text generation
        logger.info(f"[RESPONSE DEBUG] Response text length: {len(response_text)} chars")
        logger.info(f"[RESPONSE DEBUG] Response text preview: '{response_text[:100]}...' " if response_text else "[RESPONSE DEBUG] Response text is EMPTY")
        logger.info(f"[RESPONSE DEBUG] Has structured_data: {structured_data is not None}")
        logger.info(f"[RESPONSE DEBUG] Context type: {context_type}")
        
        # 🚀 FIX: Check for captured tool results BEFORE fallback response
        # This ensures tool responses are processed even when agent doesn't generate text
        if not structured_data:
            captured_result = get_captured_tool_result(session_id)
            if captured_result:
                tool_result = captured_result['result']
                tool_name = captured_result['tool_name']
                
                logger.info(f"[TOOL RESPONSE FIX] Processing captured {tool_name} result before fallback")
                log_tool_capture(tool_name, "dict", {"source": "captured_result", "keys": list(tool_result.keys()) if isinstance(tool_result, dict) else None})
                
                # Debug: Log the captured result structure
                if isinstance(tool_result, dict):
                    details = {"keys": list(tool_result.keys()), "has_products": DataFields.PRODUCTS in tool_result}
                    if DataFields.PRODUCTS in tool_result:
                        products = tool_result[DataFields.PRODUCTS]
                        details["product_count"] = len(products) if isinstance(products, list) else 0
                        details["products_type"] = str(type(products))
                    log_tool_capture(f"{tool_name}_structure", "analysis", details)
                
                # Apply unified data processing logic
                structured_data, context_type, action_required = process_tool_result_for_structured_data(
                    tool_result, source="captured"
                )
                logger.info(f"[TOOL RESPONSE FIX] Extracted context_type: {context_type}, has_structured_data: {structured_data is not None}")
        
        # 🚨 FIX: Honest error reporting - only generate responses for successful operations
        if not response_text and structured_data:
            # Check if this is a tool failure (captured tool results preserve the success field)
            tool_success = True  # Default to success for non-tool structured data
            tool_error_message = None
            
            if captured_result and isinstance(captured_result.get('result'), dict):
                tool_result_data = captured_result['result']
                tool_success = tool_result_data.get('success', True)
                if not tool_success:
                    tool_error_message = tool_result_data.get('message', 'Tool operation failed')
                    logger.warning(f"[HONEST ERROR] Tool failed with message: {tool_error_message}")
            
            # Only generate contextual responses for successful operations
            if tool_success:
                logger.info(f"[CONTEXTUAL RESPONSE] Generating response for successful operation, context_type: {context_type}")
                response = generate_contextual_response(structured_data, context_type, chat_req.message)
            else:
                # Return actual tool error instead of fake success
                logger.info(f"[HONEST ERROR] Returning tool error instead of fake success: {tool_error_message}")
                response = tool_error_message
        else:
            # Only use generic fallback for Google LLM issues, not tool failures
            if response_text:
                response = response_text
            else:
                logger.warning("[FALLBACK] No LLM response and no successful tool data - likely Google LLM issue")
                response = "I'm ready to help you with your shopping needs!"
        
        # Apply smart filtering for frontend data (preserve essential, remove bloat)
        if structured_data:
            structured_data = smart_filter_for_frontend(structured_data)
            log_performance("Smart Frontend Filter", {"operation": "frontend_optimization", "applied": True})
        
        # Create AI-optimized data summary for token reduction
        ai_context_summary = create_ai_context_summary(structured_data) if structured_data else None
        
        # Enhance message with compressed context for AI (token optimization)
        if ai_context_summary:
            context_str = json.dumps(ai_context_summary, separators=(',', ':'))[:200]  # Max 200 chars
            enhanced_message += f" [Context: {context_str}]"
            logger.info(f"[TOKEN OPTIMIZATION] Added compressed context: {len(context_str)} chars vs full data: {len(str(structured_data)) if structured_data else 0} chars")
        

        # FIX: Extract actual device_id from session after MCP tool execution
        # This ensures we return the correct device_id that was set by the MCP tool
        try:
            logger.info(f"[DEVICE_ID FIX] Attempting to extract device_id from session: {session_id}")
            # Force reading from disk to get the latest session data updated by MCP tools
            session_service = get_session_service()
            updated_session = session_service._load_from_disk(session_id)
            logger.info(f"[DEVICE_ID FIX] Session retrieved from disk: {updated_session is not None}")
            
            if updated_session and updated_session.device_id:
                # Use the actual device_id from the session (set by MCP tools)
                actual_device_id = updated_session.device_id
                logger.info(f"[DEVICE_ID FIX] Using actual device_id from session: {actual_device_id}")
                logger.info(f"[DEVICE_ID FIX] Generated device_id was: {device_id}")
                
                # Also log if they're different to confirm the fix is working
                if actual_device_id != device_id:
                    logger.info("[DEVICE_ID FIX] ✅ SUCCESS: Device ID corrected from generated to user-provided")
                else:
                    logger.info("[DEVICE_ID FIX] ⚠️ WARNING: Device IDs match, might be using fallback")
            else:
                # Fallback to generated device_id if session lookup fails
                actual_device_id = device_id
                logger.info(f"[DEVICE_ID FIX] Fallback to generated device_id: {actual_device_id}")
                if updated_session:
                    logger.info(f"[DEVICE_ID FIX] Session exists but device_id is: {updated_session.device_id}")
        except Exception as e:
            logger.error(f"[DEVICE_ID FIX] Failed to extract device_id from session, using generated: {e}")
            actual_device_id = device_id

        return ChatResponse(
            response=response,
            session_id=session_id,
            device_id=actual_device_id,
            timestamp=datetime.now(),
            data=structured_data,  # Re-enabled with smart filtering for frontend
            context_type=context_type,
            action_required=action_required,
            tools_called=session_tools_called if session_tools_called else None,  # Tool transparency
            agent_thoughts=agent_thoughts if agent_thoughts else None  # Agent reasoning
        )
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Search endpoint
@app.post("/api/v1/search")
@limiter.limit("30/minute")
async def search_products(request: Request, search_req: SearchRequest):
    """Hybrid search for products"""
    
    check_agent_ready()
    
    try:
        # Use agent to search with proper tool calling
        search_prompt = f"Search for products: {search_req.query}"
        if search_req.filters:
            search_prompt += f" with filters: {search_req.filters}"
        
        # Call search tool directly
        try:
            tool_result = await agent.call_tool(
                server_name=MCPConstants.SERVER_NAME,
                name=MCPConstants.TOOLS['SEARCH_PRODUCTS'],
                arguments={"query": search_req.query}
            )
            
            # Format search results
            if tool_result and isinstance(tool_result, dict) and DataFields.PRODUCTS in tool_result:
                products = tool_result[DataFields.PRODUCTS]
                response = f"Found {len(products)} products:\\n"
                for i, product in enumerate(products[:10], 1):
                    response += f"{i}. {product.get('name', 'Unknown')} - ₹{product.get('price', 'N/A')}\\n"
            else:
                response = str(tool_result) if tool_result else "No results found"
        except Exception as e:
            logger.error(f"Search tool error: {e}")
            response = f"Search failed: {str(e)}"
        
        return {
            "query": search_req.query,
            "results": response or "No results found",
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Cart management
@app.post("/api/v1/cart/{device_id}")
@limiter.limit("20/minute")
async def manage_cart(request: Request, device_id: str, cart_req: CartRequest):
    """Manage shopping cart"""
    
    check_agent_ready()
    
    try:
        # Use agent for cart management with proper tool calling
        cart_prompt = f"Cart action for device {device_id}: {cart_req.action}"
        if cart_req.item:
            cart_prompt += f" with item: {cart_req.item}"
        
        # Map cart actions to appropriate tools using if/elif
        if cart_req.action == "add" and cart_req.item:
            tool_name = MCPConstants.TOOLS['ADD_TO_CART']
            arguments = {"item": cart_req.item, "quantity": cart_req.quantity}
        elif cart_req.action == "view":
            tool_name = MCPConstants.TOOLS['VIEW_CART']
            arguments = {"device_id": device_id}
        elif cart_req.action == "remove" and cart_req.item:
            tool_name = MCPConstants.TOOLS['REMOVE_FROM_CART']
            arguments = {"item_id": cart_req.item.get("id", "")}
        elif cart_req.action == "update" and cart_req.item:
            tool_name = MCPConstants.TOOLS['UPDATE_CART_QUANTITY']
            arguments = {"item_id": cart_req.item.get("id", ""), "quantity": cart_req.quantity}
        elif cart_req.action == "clear":
            tool_name = MCPConstants.TOOLS['CLEAR_CART']
            arguments = {"device_id": device_id}
        else:
            return {
                "device_id": device_id,
                "action": cart_req.action,
                "result": f"Unsupported cart action: {cart_req.action}",
                "timestamp": datetime.now()
            }
        
        # Call the appropriate cart tool
        try:
            tool_result = await agent.call_tool(
                server_name=MCPConstants.SERVER_NAME,
                name=tool_name,
                arguments=arguments
            )
            response = str(tool_result) if tool_result else f"Cart {cart_req.action} completed"
        except Exception as e:
            logger.error(f"Cart tool error: {e}")
            response = f"Cart operation failed: {str(e)}"
        
        return {
            "device_id": device_id,
            "action": cart_req.action,
            "result": response or f"Cart {cart_req.action} completed",
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Cart error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Root endpoint
@app.get("/")
async def root():
    """API information"""
    session_service = get_session_service()
    return {
        "name": "ONDC Shopping Backend API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "health": "/health",
            "chat": "/api/v1/chat",
            "sessions": "/api/v1/sessions",
            "search": "/api/v1/search",
            "cart": "/api/v1/cart/{device_id}",
            "metrics": "/api/v1/metrics/performance",
            "metrics_reset": "/api/v1/metrics/reset"
        },
        "docs": "/docs",
        "active_sessions": len(session_service.sessions_cache)
    }

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("API_PORT", 8001))
    logger.info(f"🚀 Starting API server on port {port}")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info",
        reload=False
    )