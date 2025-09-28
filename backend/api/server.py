#!/usr/bin/env python3
"""
ONDC Shopping Backend API Server
FastAPI server with MCP-Agent integration for frontend applications
"""

import os
import uuid
import json
import logging
from datetime import datetime
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm_google import GoogleAugmentedLLM
from mcp_agent.workflows.llm.augmented_llm import RequestParams

# Import MCP SessionService for unified session management
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ondc-shopping-mcp'))
from src.services.session_service import get_session_service

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

def extract_minimal_product_info(data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract only essential product fields for frontend rendering, removing data bloat"""
    if not isinstance(data, dict):
        return data
    
    # If this is a products array, filter each product
    if 'products' in data and isinstance(data['products'], list):
        filtered_data = data.copy()
        filtered_products = []
        
        for product in data['products']:
            if isinstance(product, dict):
                filtered_product = {
                    # Essential item fields
                    'id': product.get('id'),
                    'local_id': product.get('local_id'),
                    'name': product.get('name'),
                    'description': product.get('description'),
                    'price': product.get('price'),
                    'category': product.get('category'),
                    
                    # Provider info (essential fields only)
                    'provider': {
                        'id': product.get('provider', {}).get('id'),
                        'local_id': product.get('provider', {}).get('local_id'),
                        'name': product.get('provider', {}).get('name')
                    } if product.get('provider') else None,
                    
                    # Location/serviceability
                    'location': {
                        'id': product.get('location', {}).get('id'),
                        'local_id': product.get('location', {}).get('local_id'),
                        'serviceability': product.get('location', {}).get('serviceability')
                    } if product.get('location') else None,
                    
                    # Essential metadata
                    'images': product.get('images'),
                    'stock_availability': product.get('stock_availability'),
                    'availability_status': product.get('availability_status'),
                    'rating': product.get('rating')
                }
                
                # Remove None values to keep response clean
                filtered_product = {k: v for k, v in filtered_product.items() if v is not None}
                if filtered_product.get('provider'):
                    filtered_product['provider'] = {k: v for k, v in filtered_product['provider'].items() if v is not None}
                if filtered_product.get('location'):
                    filtered_product['location'] = {k: v for k, v in filtered_product['location'].items() if v is not None}
                    
                filtered_products.append(filtered_product)
        
        filtered_data['products'] = filtered_products
        
        # Keep other top-level fields but remove heavy ones
        for key in list(filtered_data.keys()):
            if key.startswith('_raw') or key in ['detailed_info', 'full_ondc_data']:
                del filtered_data[key]
        
        logger.info(f"[Product Filter] Filtered {len(data['products'])} products, "
                   f"reduced from {len(str(data))} to {len(str(filtered_data))} chars")
        return filtered_data
    
    # For non-product data, just remove heavy fields
    if isinstance(data, dict):
        filtered_data = {k: v for k, v in data.items() 
                        if not k.startswith('_raw') and k not in ['detailed_info', 'full_ondc_data']}
        return filtered_data
    
    return data

def capture_tool_result_for_session(session_id: str, tool_name: str, tool_result: Any):
    """Capture tool result for later use in API response"""
    try:
        # Extract actual content from CallToolResult if needed
        actual_result = tool_result
        
        # Handle mcp.types.CallToolResult objects
        if hasattr(tool_result, 'content') and tool_result.content:
            try:
                # Try to parse the content as JSON
                if isinstance(tool_result.content, list) and len(tool_result.content) > 0:
                    # Get the first content item
                    content_item = tool_result.content[0]
                    if hasattr(content_item, 'text'):
                        actual_result = json.loads(content_item.text)
                        logger.info(f"[Tool Capture] Extracted content from CallToolResult for {tool_name}")
                elif hasattr(tool_result.content, 'text'):
                    actual_result = json.loads(tool_result.content.text)
                    logger.info(f"[Tool Capture] Extracted text content from CallToolResult for {tool_name}")
            except (json.JSONDecodeError, AttributeError) as e:
                logger.warning(f"[Tool Capture] Failed to parse CallToolResult content: {e}")
                # Fallback to original result
                actual_result = tool_result
        
        # Store the processed result
        tool_results_cache[session_id] = {
            'tool_name': tool_name,
            'result': actual_result,
            'timestamp': datetime.now()
        }
        
        # Debug: Log the structure of captured tool result
        if isinstance(actual_result, dict):
            logger.info(f"[Tool Capture] Captured {tool_name} with keys: {list(actual_result.keys())}")
            if 'products' in actual_result:
                product_count = len(actual_result['products']) if isinstance(actual_result['products'], list) else 0
                logger.info(f"[Tool Capture] Found {product_count} products in result")
        else:
            logger.info(f"[Tool Capture] Captured {tool_name} result type: {type(actual_result)}")
            
    except Exception as e:
        logger.warning(f"[Tool Capture] Failed to capture tool result: {e}")

def get_captured_tool_result(session_id: str) -> Optional[Dict[str, Any]]:
    """Get the last captured tool result for a session"""
    return tool_results_cache.get(session_id)


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
    initialization_keywords = [
        "initialize_shopping", "initialize", "start shopping", "begin shopping",
        "setup session", "create session", "userId", "deviceId"
    ]
    message_lower = message.lower()
    return any(keyword in message_lower for keyword in initialization_keywords)

def get_initialization_required_response() -> str:
    """Get the standard response for initialization requirement"""
    return (
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
    if 'products' in tool_result:
        products = tool_result['products']
        if products and len(products) > 0:
            return 'search_results', False
        else:
            return 'no_results', False
    
    # Cart operations - differentiate between view and update
    if 'cart' in tool_result or 'cart_summary' in tool_result:
        return 'cart_view', False
    
    # Single item operations (add to cart)
    if 'item_added' in tool_result or 'product' in tool_result:
        return 'cart_updated', False
    
    # Order operations - differentiate between confirmed and details
    if 'order_id' in tool_result:
        return 'order_confirmed', False
    elif 'order_details' in tool_result:
        return 'order_details', False
    
    # Checkout flow - differentiate stages
    if 'quote_data' in tool_result or 'quotes' in tool_result or 'delivery' in tool_result:
        return 'checkout_quotes', True
    elif 'next_step' in tool_result or 'stage' in tool_result:
        return 'checkout_flow', True
    
    # Payment operations
    if any(key in tool_result for key in ['payment_id', 'payment_status', 'transaction_id']):
        return 'payment_status', False
    
    # Error states
    if 'error' in tool_result or 'errors' in tool_result:
        return 'error_state', False
    
    # Success messages
    if 'success' in tool_result and tool_result.get('success') is True:
        return 'success_message', False
    
    # Session/initialization responses
    if 'session_id' in tool_result and 'device_id' in tool_result:
        return 'session_initialized', False
    
    # Default fallback for any data
    if tool_result:
        return 'data_response', False
    
    return None, False

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
                instruction="""You are an intelligent ONDC shopping assistant with MANDATORY INITIALIZATION requirement.

🚨 CRITICAL: MANDATORY INITIALIZATION FLOW 🚨
Every new shopping session MUST start with initialize_shopping(userId, deviceId) before ANY other operations.

SESSION FLOW:
1. **FIRST STEP (MANDATORY)**: initialize_shopping(userId, deviceId) - Required for every new session
2. search_products → Search for products (requires session_id from step 1)
3. add_to_cart → Add items to cart (requires session_id)
4. view_cart → Show cart contents (requires session_id)
5. select_items_for_order → Get delivery quotes (requires session_id)
6. initialize_order → Set customer details (requires session_id)
7. create_payment → Create payment (requires session_id)
8. confirm_order → Confirm order (requires session_id)

MANDATORY INITIALIZATION RULES:
- If user asks for ANY shopping operation without proper initialization, REFUSE and explain they need to initialize first
- NO guest mode defaults - userId and deviceId are MANDATORY
- Session ID from initialize_shopping must be used in ALL subsequent operations
- Users must provide their Himira frontend credentials (userId, deviceId)

INITIALIZATION REQUIREMENT RESPONSES:
If user tries to shop without initialization, respond with:
"🚨 **Session Initialization Required**

Before you can start shopping, you must initialize your session with your Himira credentials:

**Call:** `initialize_shopping(userId='your_user_id', deviceId='your_device_id')`

**How to get credentials:**
• Log into your Himira frontend
• Open browser Developer Tools (F12) 
• Check localStorage for userId and deviceId

**Why required:**
• Access your existing cart and order history
• Proper session isolation between users
• Full ONDC shopping functionality

Please provide your credentials to start shopping!"

SEARCH PATTERNS - ONLY after initialization:
- "search for X", "find X", "looking for X", "I need X", "show me X", "get X"
- "X products", "buy X", "shop for X" 
- Any product names (jams, apples, oils, etc.)
- ANY food, grocery, or merchandise query

CRITICAL RULES:
- NEVER bypass initialization requirement
- NEVER use default guest credentials 
- ALWAYS check for proper session before tools
- Guide users to initialize first if missing
- Use session_id in ALL tool calls after initialization

Always enforce mandatory initialization before any shopping operations.""",
                server_names=["ondc-shopping"]  # Connects to our MCP server
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
    
    # Check if this is a new session without credentials
    if not chat_req.session_id:
        # New session - check if this is an initialization request
        if not is_initialization_request(chat_req.message):
            # Block all non-initialization requests for new sessions
            return ChatResponse(
                response=get_initialization_required_response(),
                session_id="",  # No session until initialized
                device_id="",   # No device until initialized
                timestamp=datetime.now(),
                data=None,
                context_type="initialization_required",
                action_required=True
            )
    else:
        # Existing session - check if it's properly initialized
        # IMPORTANT: Only validate for non-initialization requests
        # Initialization requests need to be processed first to create the session data
        if not is_initialization_request(chat_req.message):
            if not is_session_initialized(chat_req.session_id):
                # Session exists but not initialized - require initialization
                return ChatResponse(
                    response=get_initialization_required_response(),
                    session_id=chat_req.session_id,
                    device_id=chat_req.device_id or "",
                    timestamp=datetime.now(),
                    data=None,
                    context_type="initialization_required", 
                    action_required=True
                )
    
    # Generate IDs if not provided (only for initialization requests)
    device_id = chat_req.device_id or generate_device_id()
    session_id = chat_req.session_id or generate_session_id()
    
    # Create or get session using SessionService
    session_service = get_session_service()
    session_obj = session_service.get_or_create(session_id)
    if not session_obj.device_id:
        session_obj.device_id = device_id
        session_service.update(session_obj)
    
    try:
        # CRITICAL: Set file-based session context for MCP tools
        # This ensures all MCP tool calls use the correct chat API session
        # Uses file-based approach since MCP server runs in separate process
        try:
            # Create session context file that MCP server can read
            context_dir = os.path.expanduser("~/.ondc-mcp")
            os.makedirs(context_dir, exist_ok=True)
            context_file = os.path.join(context_dir, "chat_session_context.txt")
            
            with open(context_file, 'w') as f:
                f.write(session_id)
            
            logger.info("Set MCP session context file: %s", session_id)
        except Exception as e:
            logger.warning("Failed to set MCP session context file: %s", e)
        
        # Session context is managed via file-based approach for MCP tools
        
        # Get session-specific LLM with conversation history
        session_llm = await get_session_llm(session_id)
        
        # Debug: Verify enhanced LLM is being used
        logger.debug("🔍 Using LLM type: %s for session: %s", type(session_llm).__name__, session_id)
        
        # Enhance message with context (simplified, no need for manual injection now)
        enhanced_message = f"[Session: {session_id}] [Device: {device_id}] {chat_req.message}"
        
        # Configure request parameters for proper tool calling
        request_params = RequestParams(
            max_iterations=5,  # Allow multi-turn tool conversations
            use_history=True,  # Maintain conversation history
            temperature=0.7,
            maxTokens=2000
        )
        
        # Clear any previous tool results for this session
        tool_results_cache.pop(session_id, None)
        
        # Temporarily patch agent to capture tool results
        original_call_tool = None
        if agent and hasattr(agent, 'call_tool'):
            original_call_tool = agent.call_tool
            
            async def capturing_call_tool(*args, **kwargs):
                # Call original tool method
                result = await original_call_tool(*args, **kwargs)
                
                # Capture the result for this session
                tool_name = args[0] if args else kwargs.get('name', 'unknown_tool')
                capture_tool_result_for_session(session_id, tool_name, result)
                
                return result
            
            # Temporarily replace the method
            agent.call_tool = capturing_call_tool
        
        try:
            # Get AI response using full conversation loop
            contents = await session_llm.generate(
                message=enhanced_message,
                request_params=request_params
            )
        finally:
            # Restore original method
            if original_call_tool:
                agent.call_tool = original_call_tool
        
        # Process the response using proper MCP integration
        response_text = ""
        structured_data = None
        context_type = None
        action_required = False
        
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
                            response_text += part.text
                            logger.debug(f"[Chat API] Added text part: {len(part.text)} chars")
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
                                    if '_structured_data' in tool_result:
                                        structured_data = tool_result['_structured_data']
                                        logger.info(f"[Chat API] ✅ Extracted structured data keys: {list(structured_data.keys()) if structured_data else 'None'}")
                                    
                                    # 🚀 ENHANCED: Extract context type from MCP response
                                    if '_context_type' in tool_result:
                                        context_type = tool_result['_context_type']
                                        logger.info(f"[Chat API] ✅ Extracted context type: {context_type}")
                                    
                                    
                                    # Extract structured data directly from tool result
                                    if not structured_data:
                                        # Check for products array (search results)
                                        if 'products' in tool_result and tool_result['products']:
                                            structured_data = tool_result
                                            context_type = 'search_results'
                                            logger.info("[Chat API] ✅ Extracted search products data")
                                        
                                        # Check for cart data
                                        elif 'cart' in tool_result or 'total_items' in tool_result:
                                            structured_data = tool_result
                                            context_type = 'cart_view'
                                            logger.info("[Chat API] ✅ Extracted cart data")
                                        
                                        # Fallback to legacy context detection
                                        else:
                                            structured_data = tool_result
                                            context_type, action_required = determine_context_type(tool_result)
                                            logger.info("[Chat API] Using legacy context detection: %s", context_type)
                                    
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
        
        response = response_text or "I'm ready to help you with your shopping needs!"
        
        # Check for captured tool results if no structured data was found via function_response
        if not structured_data:
            captured_result = get_captured_tool_result(session_id)
            if captured_result:
                tool_result = captured_result['result']
                tool_name = captured_result['tool_name']
                
                logger.info(f"[Tool Capture] Using captured result from {tool_name}")
                
                # Debug: Log the captured result structure
                if isinstance(tool_result, dict):
                    logger.info(f"[Tool Capture] Captured result keys: {list(tool_result.keys())}")
                    logger.info(f"[Tool Capture] Has products: {'products' in tool_result}")
                    if 'products' in tool_result:
                        products = tool_result['products']
                        logger.info(f"[Tool Capture] Products type: {type(products)}, count: {len(products) if isinstance(products, list) else 'not list'}")
                
                # Apply the same logic as the function_response processing
                if isinstance(tool_result, dict):
                    # Check for products array (search results)
                    if 'products' in tool_result and tool_result['products']:
                        structured_data = tool_result
                        context_type = 'search_results'
                        logger.info("[Tool Capture] ✅ Extracted search products from captured result")
                    
                    # Check for cart data  
                    elif 'cart' in tool_result or 'total_items' in tool_result:
                        structured_data = tool_result
                        context_type = 'cart_view'
                        logger.info("[Tool Capture] ✅ Extracted cart data from captured result")
                    
                    # Use general context detection
                    else:
                        structured_data = tool_result
                        context_type, action_required = determine_context_type(tool_result)
                        logger.info(f"[Tool Capture] Using context detection: {context_type}")
                else:
                    logger.warning(f"[Tool Capture] Tool result is not dict: {type(tool_result)}")
        
        # Apply minimal product info extraction to reduce data bloat
        if structured_data:
            structured_data = extract_minimal_product_info(structured_data)
            logger.info("[Chat API] Applied minimal product filtering to structured data")
        
        # IMPORTANT: Clean up session context file AFTER LLM response is processed
        # This ensures MCP tools can access the session during the request
        try:
            context_file = os.path.expanduser("~/.ondc-mcp/chat_session_context.txt")
            if os.path.exists(context_file):
                os.remove(context_file)
                logger.debug("Cleaned up MCP session context file after LLM response")
        except Exception as e:
            logger.debug("Session context file cleanup failed (not critical): %s", e)

        # FIX: Extract actual device_id from session after MCP tool execution
        # This ensures we return the correct device_id that was set by the MCP tool
        try:
            logger.info(f"[DEVICE_ID FIX] Attempting to extract device_id from session: {session_id}")
            # Force reading from disk to get the latest session data updated by MCP tools
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
            data=structured_data,
            context_type=context_type,
            action_required=action_required
        )
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        # Clean up session context file on error too
        try:
            context_file = os.path.expanduser("~/.ondc-mcp/chat_session_context.txt")
            if os.path.exists(context_file):
                os.remove(context_file)
                logger.debug("Cleaned up MCP session context file on error")
        except Exception as cleanup_error:
            logger.debug("Session context file cleanup on error failed: %s", cleanup_error)
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
                server_name="ondc-shopping",
                name="search_products",
                arguments={"query": search_req.query}
            )
            
            # Format search results
            if tool_result and isinstance(tool_result, dict) and 'products' in tool_result:
                products = tool_result['products']
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
            tool_name = "add_to_cart"
            arguments = {"item": cart_req.item, "quantity": cart_req.quantity}
        elif cart_req.action == "view":
            tool_name = "view_cart"
            arguments = {"device_id": device_id}
        elif cart_req.action == "remove" and cart_req.item:
            tool_name = "remove_from_cart"
            arguments = {"item_id": cart_req.item.get("id", "")}
        elif cart_req.action == "update" and cart_req.item:
            tool_name = "update_cart_quantity"
            arguments = {"item_id": cart_req.item.get("id", ""), "quantity": cart_req.quantity}
        elif cart_req.action == "clear":
            tool_name = "clear_cart"
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
                server_name="ondc-shopping",
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