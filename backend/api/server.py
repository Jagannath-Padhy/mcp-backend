#!/usr/bin/env python3
"""
ONDC Shopping Backend API Server
FastAPI server with MCP-Agent integration for frontend applications
"""

import os
import uuid
import json
import time
import asyncio
import logging
from datetime import datetime
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm_google import GoogleAugmentedLLM
from mcp_agent.workflows.llm.augmented_llm import RequestParams

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

# Global instances
mcp_app = None
agent = None
llm = None
sessions = {}  # In-memory session storage (use Redis/MongoDB in production)
session_llms = {}  # Session-specific LLM instances with conversation history
raw_data_queues = {}  # Session-specific queues for raw data from MCP callbacks

# ============================================================================
# Universal SSE Data Transmission System
# ============================================================================

# Tool to event type mapping for SSE streaming
TOOL_EVENT_MAPPING = {
    'search_products': 'raw_products',
    'add_to_cart': 'raw_cart', 
    'view_cart': 'raw_cart',
    'update_cart_quantity': 'raw_cart',
    'remove_from_cart': 'raw_cart',
    'clear_cart': 'raw_cart',
    'get_cart_total': 'raw_cart',
    # Checkout/Order tools (Universal Pattern)
    'select_items_for_order': 'raw_checkout',
    'initialize_order': 'raw_checkout',
    'confirm_order': 'raw_checkout',
    'create_payment': 'raw_payment',
    # Payment verification tools
    'verify_payment': 'raw_payment',
    'get_payment_status': 'raw_payment',
    # Future tools can be added here
    # 'get_order_history': 'raw_orders',
    # 'get_delivery_addresses': 'raw_addresses',
    # 'get_active_offers': 'raw_offers',
}

def create_sse_event(tool_name, raw_data, session_id):
    """Create universal SSE event based on tool type using DRY pattern"""
    event_type = TOOL_EVENT_MAPPING.get(tool_name, 'raw_data')  # Generic fallback
    
    # Base event structure
    event_data = {
        'tool_name': tool_name,
        'session_id': session_id,
        'raw_data': True,
        'biap_specifications': True,
        'timestamp': datetime.now().isoformat()
    }
    
    # Tool-specific data mappings
    cart_tools = ['add_to_cart', 'view_cart', 'update_cart_quantity', 'remove_from_cart', 'clear_cart', 'get_cart_total']
    payment_tools = ['create_payment', 'verify_payment', 'get_payment_status']
    
    tool_data_mappings = {
        'search_products': {
            'products': raw_data.get('products', []),
            'total_results': raw_data.get('total_results', 0),
            'search_type': raw_data.get('search_type', 'hybrid'),
            'page': raw_data.get('page', 1),
            'page_size': raw_data.get('page_size', 10)
        },
        'cart_tools': {
            'cart_items': raw_data.get('cart_items', []),
            'cart_summary': raw_data.get('cart_summary', {})
        },
        'payment_tools': {
            'payment_status': raw_data.get('payment_status', 'unknown'),
            'payment_id': raw_data.get('payment_id'),
            'payment_verification': raw_data.get('payment_verification'),
            'razorpay_order_id': raw_data.get('razorpay_order_id'),
            'next_step': raw_data.get('next_step'),
            'user_action_required': raw_data.get('user_action_required')
        }
    }
    
    # Apply appropriate data mapping
    if tool_name == 'search_products':
        event_data.update(tool_data_mappings['search_products'])
    elif tool_name in cart_tools:
        event_data.update(tool_data_mappings['cart_tools'])
    elif tool_name in payment_tools:
        event_data.update(tool_data_mappings['payment_tools'])
    else:
        event_data.update(raw_data)  # Generic fallback
    
    return {
        'event_type': event_type,
        'data': event_data
    }

def get_log_message(tool_name, raw_data):
    """Generate appropriate log message based on tool type using DRY pattern"""
    base_msg = "[RAW-DATA] Queued"
    
    # Tool-specific data extraction
    tool_messages = {
        'search_products': f"{len(raw_data.get('products', []))} products",
        'cart_tools': f"cart data ({len(raw_data.get('cart_items', [])) if isinstance(raw_data.get('cart_items'), list) else 'dict'} items)",
        'payment_tools': f"payment data (status: {raw_data.get('payment_status', 'unknown')}, id: {raw_data.get('payment_id', 'none')})"
    }
    
    # Categorize tools
    cart_tools = ['add_to_cart', 'view_cart', 'update_cart_quantity', 'remove_from_cart', 'clear_cart', 'get_cart_total']
    payment_tools = ['create_payment', 'verify_payment', 'get_payment_status']
    
    if tool_name == 'search_products':
        detail = tool_messages['search_products']
    elif tool_name in cart_tools:
        detail = tool_messages['cart_tools']
    elif tool_name in payment_tools:
        detail = tool_messages['payment_tools']
    else:
        detail = f"{tool_name} data"
    
    return f"{base_msg} {detail} for SSE stream"

# ============================================================================
# Helper Functions for DRY Code
# ============================================================================

def generate_device_id() -> str:
    """Generate a unique device ID."""
    return f"device_{uuid.uuid4().hex[:8]}"

def generate_session_id() -> str:
    """Generate a unique session ID."""
    return f"session_{uuid.uuid4().hex}"

def create_or_update_session(session_id: str, device_id: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
    """Create a new session or update existing one."""
    if session_id not in sessions:
        sessions[session_id] = {
            "session_id": session_id,
            "device_id": device_id,
            "created_at": datetime.now(),
            "last_activity": datetime.now(),
            "metadata": metadata or {}
        }
    else:
        sessions[session_id]["last_activity"] = datetime.now()
        if metadata:
            sessions[session_id]["metadata"].update(metadata)
    return sessions[session_id]

def check_agent_ready():
    """Check if agent is ready and raise appropriate error."""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not ready")
    if not llm:
        raise HTTPException(status_code=503, detail="LLM not ready")

def determine_context_type(tool_result: Dict[str, Any]) -> tuple[str, bool]:
    """Determine context type and action requirement from tool result.
    
    Returns:
        tuple: (context_type, action_required)
    """
    if not isinstance(tool_result, dict):
        return None, False
    
    # Check for known data patterns - enhanced to detect search results
    for key in tool_result:
        match key:
            case 'products' | 'search_results':  # Enhanced: detect both products and search_results
                return 'products', False
            case 'cart' | 'cart_summary':
                return 'cart', False
            case 'order_id' | 'order_details':
                return 'order', False
            case 'quote_data' | 'delivery':
                return 'checkout', True
            case 'next_step' | 'stage':
                return 'checkout', True
    
    # Additional check for search response patterns
    if ('success' in tool_result and 'message' in tool_result and 
        any(search_term in str(tool_result.get('message', '')).lower() 
            for search_term in ['found', 'products', 'search'])):
        return 'products', False
    
    return None, False

async def get_session_llm(session_id: str):
    """Get or create a session-specific LLM with conversation history"""
    logger.info(f"[LLM-LIFECYCLE] Getting LLM for session: {session_id}")
    logger.info(f"[LLM-LIFECYCLE] Current session_llms keys: {list(session_llms.keys())}")
    
    if session_id not in session_llms:
        if not agent:
            raise HTTPException(status_code=503, detail="Agent not ready")
        
        # Create a new LLM instance for this session
        session_llm = await agent.attach_llm(GoogleAugmentedLLM)
        session_llms[session_id] = session_llm
        logger.info(f"[LLM-LIFECYCLE] Created NEW session LLM for session: {session_id}")
        logger.info(f"[LLM-LIFECYCLE] session_llms now has {len(session_llms)} entries")
    else:
        logger.info(f"[LLM-LIFECYCLE] Reusing EXISTING session LLM for session: {session_id}")
    
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
                instruction="""You are an intelligent shopping assistant that takes decisive action and chains tools automatically to fulfill user requests.

🚨 CRITICAL RULE: ALWAYS call the appropriate function for user requests!
• "search X" → MUST call search_products(query="X")
• "view cart" → MUST call view_cart()
• "add Y" → MUST call search_products() then add_to_cart()
• "checkout" or "proceed to checkout" → MUST call select_items_for_order()
• NEVER provide generic responses - ALWAYS use the specific tool

INTELLIGENT BEHAVIOR:
• Auto-add items when searching for specific products
• Choose best option based on price/quality
• Chain tools: search → add → view_cart → inform user
• Always execute the most relevant tool for each request
• CALCULATE quantities automatically when context is given
• ALWAYS call view_cart after add_to_cart to show updated cart state

QUANTITY INTELLIGENCE:
• "for X people" → Calculate appropriate quantities based on serving size
• "for cooking/family" → Add standard cooking quantities 
• "need X" without quantity → Add 1 unit as default
• NEVER ask for quantity confirmation - be decisive and add!

UNIVERSAL EXAMPLES:
User: "search for [item]" → Call search_products(query="[item]")
User: "view my cart" → Call view_cart()
User: "i need [item]" → Call search_products(query="[item]") → Call add_to_cart(quantity=1) → Call view_cart()
User: "[item] for X people" → Call search_products(query="[item]") → Call add_to_cart(quantity=[calculated]) → Call view_cart()
User: "[item] for family" → Call search_products(query="[item]") → Call add_to_cart(quantity=[reasonable]) → Call view_cart()
User: "checkout" → Call select_items_for_order() → initialize_order() → create_payment()

Be proactive - calculate quantities intelligently, use tools, don't ask for confirmation!

=== CHECKOUT AUTOMATION BOUNDARIES ===
Checkout automation: select_items_for_order → initialize_order → create_payment (then wait)
Payment processing: verify_payment and confirm_order require explicit user/frontend requests
After create_payment: Wait for manual payment verification before continuing""",
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


class SessionCreateRequest(BaseModel):
    device_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = {}

class SessionResponse(BaseModel):
    session_id: str
    device_id: str
    created_at: datetime
    last_activity: datetime
    metadata: Dict[str, Any]


class CartRequest(BaseModel):
    action: str  # add, remove, update, view
    item: Optional[Dict[str, Any]] = None
    quantity: Optional[int] = 1

# Internal tool result endpoint for raw data streaming
@app.post("/internal/tool-result")
async def receive_tool_result(tool_data: dict):
    """Internal endpoint for MCP server to send raw tool results"""
    session_id = tool_data.get('session_id')
    tool_name = tool_data.get('tool_name')
    raw_data = tool_data.get('raw_data', {})
    
    logger.info(f"[RAW-DATA] Received {tool_name} data for session {session_id}")
    
    # Send raw data to active SSE streams via queue using universal system
    if session_id and session_id in raw_data_queues:
        try:
            # Check if tool has any raw data to transmit
            has_data = False
            if tool_name == 'search_products' and raw_data.get('products'):
                has_data = True
            elif tool_name in TOOL_EVENT_MAPPING and raw_data:
                has_data = True
            
            if has_data:
                # Create universal SSE event
                raw_event = create_sse_event(tool_name, raw_data, session_id)
                
                # Put raw data into the session's queue for SSE streaming
                await raw_data_queues[session_id].put(raw_event)
                
                # Log with appropriate message
                log_message = get_log_message(tool_name, raw_data)
                logger.info(f"{log_message} in session {session_id}")
            else:
                logger.debug(f"[RAW-DATA] No data to transmit for {tool_name} in session {session_id}")
            
        except Exception as e:
            logger.error(f"[RAW-DATA] Failed to queue raw data for session {session_id}: {e}")
    elif session_id:
        logger.info(f"[RAW-DATA] No active SSE stream for session {session_id} - data received but not queued")
    
    return {"status": "received"}

# Internal tool event endpoint for real-time tool execution events
@app.post("/internal/tool-event")
async def receive_tool_event(event_data: dict):
    """Internal endpoint for MCP server to send real-time tool execution events"""
    session_id = event_data.get('session_id')
    event_type = event_data.get('event_type')
    tool_name = event_data.get('tool_name')
    message = event_data.get('message')
    
    logger.debug(f"[TOOL-EVENT] {event_type} for {tool_name} in session {session_id}")
    
    # Send tool event to active SSE streams
    if session_id and session_id in raw_data_queues:
        try:
            # Create tool execution event
            tool_event = {
                'event_type': event_type,
                'data': {
                    'type': event_type,
                    'tool_name': tool_name,
                    'message': message,
                    'session_id': session_id,
                    'timestamp': event_data.get('timestamp'),
                    'execution_time_ms': event_data.get('execution_time_ms'),
                    'success': event_data.get('success')
                }
            }
            
            # Put tool event into the session's queue for SSE streaming
            await raw_data_queues[session_id].put(tool_event)
            
            logger.debug(f"[TOOL-EVENT] Queued {event_type} event for {tool_name} in session {session_id}")
            
        except Exception as e:
            logger.error(f"[TOOL-EVENT] Failed to queue tool event for session {session_id}: {e}")
    
    return {"status": "received"}

# Health check
@app.get("/health")
@limiter.limit("60/minute")
async def health_check(request: Request):
    """Health check endpoint"""
    return {
        "status": "healthy" if llm else "initializing",
        "timestamp": datetime.now(),
        "agent_ready": llm is not None,
        "active_sessions": len(sessions)
    }

# Session management
@app.post("/api/v1/sessions", response_model=SessionResponse)
@limiter.limit("10/minute")
async def create_session(request: Request, session_req: SessionCreateRequest):
    """Create a new shopping session"""
    session_id = generate_session_id()
    device_id = session_req.device_id or generate_device_id()
    
    session = create_or_update_session(session_id, device_id, session_req.metadata)
    logger.info(f"Created session: {session_id}")
    
    return SessionResponse(**session)

@app.get("/api/v1/sessions/{session_id}", response_model=SessionResponse)
@limiter.limit("30/minute")
async def get_session(request: Request, session_id: str):
    """Get session information"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return SessionResponse(**sessions[session_id])

@app.delete("/api/v1/sessions/{session_id}")
@limiter.limit("20/minute")
async def delete_session(request: Request, session_id: str):
    """End a shopping session"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Clean up session data and LLM
    del sessions[session_id]
    if session_id in session_llms:
        del session_llms[session_id]
    
    logger.info(f"Deleted session and LLM: {session_id}")
    
    return {"message": "Session deleted"}

# ============================================================================
# SSE Streaming Helper Functions
# ============================================================================

def sse_event(event_type: str, data: dict) -> str:
    """Format SSE event with proper structure"""
    return f"data: {json.dumps({'type': event_type, **data})}\n\n"

# Removed hardcoded detection functions - agent should naturally understand requests

def process_mcp_results(contents) -> tuple:
    """Process MCP results and extract structured data with comprehensive tracing"""
    response_text = ""
    structured_data = None
    context_type = None
    action_required = False
    function_calls_detected = False
    
    # ENHANCED LOGGING: Track response generation sources
    logger.info(f"[RESPONSE-TRACE] process_mcp_results called with contents type: {type(contents)}")
    if contents:
        logger.info(f"[RESPONSE-TRACE] Contents length: {len(contents)}")
    else:
        logger.warning(f"[RESPONSE-TRACE] Contents is None/empty!")
    
    # Process MCP agent response structure
    if contents and len(contents) > 0:
        for i, content in enumerate(contents):
            logger.info(f"[RESPONSE-TRACE] Processing content {i}: type={type(content)}")
            if hasattr(content, 'parts') and content.parts:
                logger.info(f"[RESPONSE-TRACE] Content {i} has {len(content.parts)} parts")
                for j, part in enumerate(content.parts):
                    # Check text content
                    if hasattr(part, 'text') and part.text:
                        logger.info(f"[RESPONSE-TRACE] Part {j} text ({len(part.text)} chars): {repr(part.text[:200])}")
                        response_text += part.text
                    
                    # Check function response (separate from text - both can exist)
                    if hasattr(part, 'function_response') and part.function_response is not None:
                        function_calls_detected = True
                        logger.info(f"[RESPONSE-TRACE] Part {j} has function_response: {type(part.function_response)}")
                        # Try different ways to access the structured data
                        tool_result = None
                        if hasattr(part.function_response, 'content'):
                            try:
                                if isinstance(part.function_response.content, str):
                                    tool_result = json.loads(part.function_response.content)
                                    logger.info(f"[RESPONSE-TRACE] Parsed JSON tool result: {json.dumps(tool_result, indent=2)[:500]}")
                                else:
                                    tool_result = part.function_response.content
                                    logger.info(f"[RESPONSE-TRACE] Direct tool result: {str(tool_result)[:500]}")
                            except (json.JSONDecodeError, AttributeError) as e:
                                logger.warning(f"Failed to parse function response content: {e}")
                        elif hasattr(part.function_response, 'result'):
                            tool_result = part.function_response.result
                            logger.info(f"[RESPONSE-TRACE] Result tool result: {str(tool_result)[:500]}")
                        elif isinstance(part.function_response, dict):
                            tool_result = part.function_response
                            logger.info(f"[RESPONSE-TRACE] Dict tool result: {str(tool_result)[:500]}")
                        else:
                            try:
                                tool_result = dict(part.function_response)
                                logger.info(f"[RESPONSE-TRACE] Converted tool result: {str(tool_result)[:500]}")
                            except Exception as e:
                                logger.warning(f"Could not extract structured data from function_response: {e}")
                        
                        if tool_result and isinstance(tool_result, dict):
                            structured_data = tool_result
                            context_type, action_required = determine_context_type(tool_result)
                            logger.info(f"[RESPONSE-TRACE] Extracted context_type: {context_type}, action_required: {action_required}")
    
    # Log function call detection for debugging
    if function_calls_detected:
        logger.info("[FUNCTION-CALLS] Verified function calls were executed properly")
    
    # Clean up response text
    if response_text.startswith("None"):
        response_text = response_text[4:]
    
    response_text = response_text or "I'm ready to help you with your shopping needs!"
    
    # CRITICAL: Validate agent response against structured data
    if structured_data and context_type == 'cart':
        logger.info(f"[RESPONSE-VALIDATION] Validating cart response against structured data")
        cart_data = structured_data.get('cart', {})
        
        # Extract actual cart numbers from structured data
        actual_total_items = cart_data.get('total_items', 0)
        actual_total_value = cart_data.get('total_value', 0)
        
        logger.info(f"[RESPONSE-VALIDATION] Structured data shows: {actual_total_items} items, ₹{actual_total_value}")
        logger.info(f"[RESPONSE-VALIDATION] Agent response text: {repr(response_text)}")
        
        # Check for number discrepancies in response text
        import re
        item_numbers = re.findall(r'(\d+)\s*items?', response_text.lower())
        price_numbers = re.findall(r'₹(\d+(?:,\d+)*(?:\.\d+)?)', response_text)
        
        if item_numbers:
            claimed_items = int(item_numbers[-1])  # Get last/most recent item count
            logger.info(f"[RESPONSE-VALIDATION] Agent claims {claimed_items} items, actual: {actual_total_items}")
            
            if claimed_items != actual_total_items:
                logger.error(f"[RESPONSE-VALIDATION] CART DISCREPANCY DETECTED!")
                logger.error(f"[RESPONSE-VALIDATION] Agent claims: {claimed_items} items")
                logger.error(f"[RESPONSE-VALIDATION] Backend reality: {actual_total_items} items")
                # Replace with correct data
                response_text = response_text.replace(f"{claimed_items} items", f"{actual_total_items} items")
        
        if price_numbers:
            claimed_price = float(price_numbers[-1].replace(',', ''))  # Get last/most recent price
            logger.info(f"[RESPONSE-VALIDATION] Agent claims ₹{claimed_price}, actual: ₹{actual_total_value}")
            
            if abs(claimed_price - actual_total_value) > 0.01:  # Allow for rounding
                logger.error(f"[RESPONSE-VALIDATION] PRICE DISCREPANCY DETECTED!")
                logger.error(f"[RESPONSE-VALIDATION] Agent claims: ₹{claimed_price}")
                logger.error(f"[RESPONSE-VALIDATION] Backend reality: ₹{actual_total_value}")
                # Replace with correct data
                response_text = response_text.replace(f"₹{claimed_price}", f"₹{actual_total_value}")
    
    logger.info(f"[RESPONSE-TRACE] Final response_text: {repr(response_text[:200])}")
    logger.info(f"[RESPONSE-TRACE] Final structured_data present: {structured_data is not None}")
    
    return response_text, structured_data, context_type, action_required


# SSE Streaming chat endpoint
@app.post("/api/v1/chat/stream")
@limiter.limit("20/minute")
async def chat_stream(request: Request, chat_req: ChatRequest):
    """Streaming chat with agent thoughts and structured events"""
    
    check_agent_ready()
    
    # Generate IDs if not provided - same logic as regular chat
    device_id = chat_req.device_id or generate_device_id()
    session_id = chat_req.session_id or generate_session_id()
    
    # Create or update session - same session management as regular chat
    create_or_update_session(session_id, device_id)
    
    async def robust_event_stream():
        
        # Configurable SSE connection timeout (seconds)
        connection_timeout = int(os.getenv('SSE_CONNECTION_TIMEOUT', 300))
        start_time = time.time()
        
        # Create asyncio queue for this session to receive raw data from MCP callbacks
        raw_data_queues[session_id] = asyncio.Queue()
        logger.info(f"[SSE-RAW] Created raw data queue for session {session_id}")
        
        try:
            # 1. THINKING EVENTS - User Experience
            yield sse_event('thinking', {
                'message': 'Analyzing your request...',
                'timestamp': datetime.now().isoformat(),
                'session_id': session_id
            })
            
            await asyncio.sleep(0.5)  # Brief pause for better UX
            
            # 2. INTELLIGENT PROCESSING - Let agent decide tools naturally
            yield sse_event('thinking', {'message': 'Understanding your request...', 'session_id': session_id})
            
            # 3. EXECUTE MCP TOOLS (Same logic as regular chat endpoint)
            logger.info(f"[CHAT-SSE] Processing message for session: {session_id}")
            session_llm = await get_session_llm(session_id)
            
            # Enhanced message with context
            enhanced_message = f"[Session: {session_id}] [Device: {device_id}] {chat_req.message}"
            logger.info(f"[CHAT-SSE] Enhanced message: {enhanced_message}")
            
            # Configure request parameters for precise tool selection
            request_params = RequestParams(
                max_iterations=5,
                use_history=True,
                temperature=0.2,  # Low temperature for smart tool picking
                maxTokens=2000
            )
            
            # Indicate processing
            yield sse_event('thinking', {'message': 'Processing with available tools...', 'session_id': session_id})
            
            # Start progressive streaming with real-time tool execution
            yield sse_event('conversation_chunk', {
                'message': 'Let me help you with that...',
                'session_id': session_id,
                'stage': 'initial'
            })
            
            # Execute with progressive monitoring - run LLM generation in background
            # Create a future for the LLM generation
            loop = asyncio.get_event_loop()
            generation_task = loop.create_task(session_llm.generate(
                message=enhanced_message,
                request_params=request_params
            ))
            
            # Monitor raw data queue for real-time tool events while generation runs
            contents = None
            tool_events_sent = []
            
            while not generation_task.done():
                # Check for tool events in real-time
                if session_id in raw_data_queues:
                    queue = raw_data_queues[session_id]
                    try:
                        # Short timeout to keep checking generation status
                        raw_event = await asyncio.wait_for(queue.get(), timeout=0.1)
                        
                        # Send tool events immediately as they happen
                        if raw_event['event_type'] == 'tool_start':
                            tool_name = raw_event['data'].get('tool_name', 'unknown')
                            yield sse_event('tool_start', raw_event['data'])
                            yield sse_event('conversation_chunk', {
                                'message': f'Executing {tool_name}...',
                                'session_id': session_id,
                                'stage': 'tool_execution'
                            })
                            tool_events_sent.append(('start', tool_name))
                            logger.info(f"[SSE-REALTIME] Sent tool_start for {tool_name}")
                            
                        elif raw_event['event_type'] == 'tool_complete':
                            tool_name = raw_event['data'].get('tool_name', 'unknown')
                            yield sse_event('tool_complete', raw_event['data'])
                            tool_events_sent.append(('complete', tool_name))
                            logger.info(f"[SSE-REALTIME] Sent tool_complete for {tool_name}")
                            
                        elif raw_event['event_type'] == 'raw_products':
                            yield sse_event('raw_products', raw_event['data'])
                            products_count = len(raw_event['data'].get('products', []))
                            yield sse_event('conversation_chunk', {
                                'message': f'Found {products_count} products...',
                                'session_id': session_id,
                                'stage': 'tool_results'
                            })
                            logger.info(f"[SSE-REALTIME] Sent {products_count} raw products")
                            
                        elif raw_event['event_type'] == 'raw_cart':
                            yield sse_event('raw_cart', raw_event['data'])
                            logger.info(f"[SSE-REALTIME] Sent raw cart data")
                            
                    except asyncio.TimeoutError:
                        # No events in queue, continue monitoring
                        pass
                    except Exception as e:
                        logger.error(f"[SSE-REALTIME] Error processing queue event: {e}")
                else:
                    # Short sleep if no queue exists yet
                    await asyncio.sleep(0.1)
            
            # Get the final result
            contents = await generation_task
            logger.info(f"[SSE-REALTIME] LLM generation completed, tool events sent: {tool_events_sent}")
            
            # 4. STRUCTURED RESULT EVENTS - Frontend Integration
            response_text, structured_data, context_type, action_required = process_mcp_results(contents)
            
            # Send different event types based on content
            if structured_data and context_type:
                # Products found - enhanced with raw BIAP data and intelligent search metadata
                if context_type == 'products':
                    products_list = structured_data.get('products', [])
                    search_metadata = structured_data.get('search_metadata', {})
                    
                    yield sse_event('products', {
                        'products': products_list,  # Now contains raw BIAP data with full specifications
                        'search_results': structured_data.get('search_results', []),  # Full search context if available
                        'total_count': len(products_list),
                        'total_results': structured_data.get('total_results', len(products_list)),
                        'search_query': chat_req.message,
                        'search_type': structured_data.get('search_type', 'hybrid'),
                        'page': structured_data.get('page', 1),
                        'page_size': structured_data.get('page_size', 10),
                        'session_id': session_id,
                        'raw_data': True,  # Signal to frontend this contains unformatted BIAP data
                        'biap_specifications': True,  # Signal that full ONDC specifications are available
                        # Enhanced search intelligence metadata
                        'search_intelligence': {
                            'relevance_threshold': search_metadata.get('relevance_threshold'),
                            'adaptive_results': search_metadata.get('adaptive_results', False),
                            'context_aware': search_metadata.get('context_aware', False),
                            'original_limit_requested': search_metadata.get('original_limit_requested', 'auto'),
                            'final_limit_applied': search_metadata.get('final_limit_applied', 10),
                            'search_suggestions': search_metadata.get('search_suggestions', []),
                            'semantic_validation': search_metadata.get('relevance_threshold', 0) > 0.6,
                            'filtered_by_relevance': structured_data.get('filtered_by_relevance', False)
                        }
                    })
                
                # Cart updated
                elif context_type == 'cart':
                    yield sse_event('cart_update', {
                        'cart': structured_data.get('cart', {}),
                        'total_items': structured_data.get('total_items', 0),
                        'total_amount': structured_data.get('total_amount', 0),
                        'session_id': session_id
                    })
                
                # Generic tool result
                else:
                    yield sse_event('tool_result', {
                        'data': structured_data,
                        'context_type': context_type,
                        'action_required': action_required,
                        'session_id': session_id
                    })
            
            # Process any remaining events in queue after generation completes
            final_events_processed = 0
            if session_id in raw_data_queues:
                queue = raw_data_queues[session_id]
                while not queue.empty():
                    try:
                        raw_event = await asyncio.wait_for(queue.get(), timeout=0.1)
                        final_events_processed += 1
                        
                        # Handle any remaining tool events
                        if raw_event['event_type'] == 'tool_start':
                            yield sse_event('tool_start', raw_event['data'])
                        elif raw_event['event_type'] == 'tool_complete':
                            yield sse_event('tool_complete', raw_event['data'])
                        elif raw_event['event_type'] == 'raw_products':
                            yield sse_event('raw_products', raw_event['data'])
                        elif raw_event['event_type'] == 'raw_cart':
                            yield sse_event('raw_cart', raw_event['data'])
                    except asyncio.TimeoutError:
                        break
                    except Exception as e:
                        logger.error(f"[SSE-REALTIME] Error processing final events: {e}")
                        break
            
            logger.info(f"[SSE-REALTIME] Processed {final_events_processed} final events")
            
            # Add conversational context based on tool execution
            if tool_events_sent:
                tool_names = [event[1] for event in tool_events_sent if event[0] == 'complete']
                if 'search_products' in tool_names and 'add_to_cart' in tool_names:
                    yield sse_event('conversation_chunk', {
                        'message': 'Perfect! I found great options and added the best one to your cart.',
                        'session_id': session_id,
                        'stage': 'completion'
                    })
                elif 'search_products' in tool_names:
                    yield sse_event('conversation_chunk', {
                        'message': 'Great! I found some excellent options for you.',
                        'session_id': session_id,
                        'stage': 'completion'
                    })
            
            # 5. FINAL RESPONSE EVENT (now comes AFTER tool events)
            yield sse_event('response', {
                'content': response_text,
                'session_id': session_id,
                'timestamp': datetime.now().isoformat(),
                'complete': True
            })
            
            # 6. COMPLETION SIGNAL
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            
            # Handle Google AI quota exhaustion gracefully
            error_str = str(e)
            if "RESOURCE_EXHAUSTED" in error_str or "quota" in error_str.lower():
                yield sse_event('error', {
                    'message': '🤖 AI assistant temporarily unavailable due to quota limits. Cart and shopping tools continue to work normally.',
                    'recoverable': True,
                    'retry_suggestion': 'Please try again later, or use direct commands like "search turmeric" or "view cart"',
                    'session_id': session_id,
                    'error_type': 'quota_exhausted'
                })
            else:
                yield sse_event('error', {
                    'message': f'Error: {str(e)}',
                    'recoverable': True,
                    'retry_suggestion': 'Please try rephrasing your request',
                    'session_id': session_id
                })
            yield "data: [DONE]\n\n"
        
        finally:
            # Clean up raw data queue for this session
            if session_id in raw_data_queues:
                del raw_data_queues[session_id]
                logger.info(f"[SSE-RAW] Cleaned up raw data queue for session {session_id}")
            
            # Connection cleanup if needed
            if time.time() - start_time > connection_timeout:
                logger.warning(f"SSE connection timeout for session {session_id}")

    return StreamingResponse(robust_event_stream(), media_type="text/event-stream")

# Search endpoint

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
        
        # Map cart actions to appropriate tools using match/case
        match cart_req.action:
            case "add" if cart_req.item:
                tool_name = "add_to_cart"
                arguments = {"item": cart_req.item, "quantity": cart_req.quantity}
            case "view":
                tool_name = "view_cart"
                arguments = {"device_id": device_id}
            case "remove" if cart_req.item:
                tool_name = "remove_from_cart"
                arguments = {"item_id": cart_req.item.get("id", "")}
            case "update" if cart_req.item:
                tool_name = "update_cart_quantity"
                arguments = {"item_id": cart_req.item.get("id", ""), "quantity": cart_req.quantity}
            case "clear":
                tool_name = "clear_cart"
                arguments = {"device_id": device_id}
            case _:
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
    return {
        "name": "ONDC Shopping Backend API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "health": "/health",
            "chat_stream": "/api/v1/chat/stream",
            "sessions": "/api/v1/sessions",
            "search": "/api/v1/search",
            "cart": "/api/v1/cart/{device_id}"
        },
        "docs": "/docs",
        "active_sessions": len(sessions)
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