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
    'create_payment': 'raw_checkout',
    # Future tools can be added here
    # 'get_order_history': 'raw_orders',
    # 'get_delivery_addresses': 'raw_addresses',
    # 'get_active_offers': 'raw_offers',
}

def create_sse_event(tool_name, raw_data, session_id):
    """Create universal SSE event based on tool type"""
    event_type = TOOL_EVENT_MAPPING.get(tool_name, 'raw_data')  # Generic fallback
    
    # Create base event structure
    event_data = {
        'tool_name': tool_name,
        'session_id': session_id,
        'raw_data': True,
        'biap_specifications': True,
        'timestamp': datetime.now().isoformat()
    }
    
    # Add tool-specific data
    if tool_name == 'search_products':
        event_data.update({
            'products': raw_data.get('products', []),
            'total_results': raw_data.get('total_results', 0),
            'search_type': raw_data.get('search_type', 'hybrid'),
            'page': raw_data.get('page', 1),
            'page_size': raw_data.get('page_size', 10)
        })
    elif tool_name in ['add_to_cart', 'view_cart', 'update_cart_quantity', 'remove_from_cart', 'clear_cart', 'get_cart_total']:
        event_data.update({
            'cart_items': raw_data.get('cart_items', []),
            'cart_summary': raw_data.get('cart_summary', {})
        })
    else:
        # Generic data structure for future tools
        event_data.update(raw_data)
    
    return {
        'event_type': event_type,
        'data': event_data
    }

def get_log_message(tool_name, raw_data):
    """Generate appropriate log message based on tool type"""
    if tool_name == 'search_products':
        return f"[RAW-DATA] Queued {len(raw_data.get('products', []))} products for SSE stream"
    elif tool_name in ['add_to_cart', 'view_cart', 'update_cart_quantity', 'remove_from_cart', 'clear_cart', 'get_cart_total']:
        cart_data = raw_data.get('cart_items', [])
        cart_count = len(cart_data) if isinstance(cart_data, list) else "dict"
        return f"[RAW-DATA] Queued cart data ({cart_count} items) for SSE stream"
    else:
        return f"[RAW-DATA] Queued {tool_name} data for SSE stream"

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
        async with mcp_app.run() as app_context:
            
            # Create agent connected to MCP server via STDIO
            agent = Agent(
                name="shopping_assistant",
                instruction="""You are a proactive ONDC shopping assistant. When customers mention dishes or cooking needs, automatically search for ALL required ingredients and ask about quantities/servings before adding to cart.

KEY BEHAVIORS:
• Recipe Intelligence: "jeera rice" → search jeera AND rice, suggest quantities for servings
• Tool Chaining: Search multiple ingredients → present options → ask serving size → add to cart → suggest complementary items  
• Always ask "How many people will this serve?" before adding items
• Anticipate needs: rice → suggest oil/spices, pasta → suggest sauce/cheese

CORE TOOLS:
Search: search_products(query), advanced_search(filters), browse_categories()
Cart: add_to_cart(item, quantity), view_cart(), update_cart_quantity(), remove_from_cart(), clear_cart()
Checkout: select_items_for_order(city, state, pincode), initialize_order(name, address, phone, email), confirm_order()

EXAMPLE: 
Wrong: "Found jeera ₹120. Add to cart?"
Right: "For jeera rice, you need jeera (₹120) and basmati rice (₹250). How many people? For 4 people, I recommend 1 pack jeera + 2 cups rice. Add both?"

Be the helpful shopkeeper who anticipates complete meal needs, not just individual items.""",
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

def is_search_request(message: str) -> bool:
    """Check if message is a search request"""
    search_keywords = ['search', 'find', 'look for', 'show me', 'get me']
    return any(keyword in message.lower() for keyword in search_keywords)

def is_cart_request(message: str) -> bool:
    """Check if message is a cart operation"""
    cart_keywords = ['add to cart', 'cart', 'add', 'remove', 'delete']
    return any(keyword in message.lower() for keyword in cart_keywords)

def process_mcp_results(contents) -> tuple:
    """Process MCP results and extract structured data"""
    response_text = ""
    structured_data = None
    context_type = None
    action_required = False
    
    # Enhanced debugging for mcp-agent response structure
    logger.info(f"[SSE-DEBUG] Contents type: {type(contents)}, Length: {len(contents) if contents else 0}")
    
    if contents and len(contents) > 0:
        for i, content in enumerate(contents):
            logger.info(f"[SSE-DEBUG] Content {i} type: {type(content)}, Has parts: {hasattr(content, 'parts')}")
            if hasattr(content, 'parts') and content.parts:
                logger.info(f"[SSE-DEBUG] Content {i} parts count: {len(content.parts)}")
                for j, part in enumerate(content.parts):
                    logger.info(f"[SSE-DEBUG] Part {j} type: {type(part)}, Attributes: {[attr for attr in dir(part) if not attr.startswith('_')]}")
                    
                    if hasattr(part, 'text') and part.text:
                        response_text += part.text
                        logger.info(f"[SSE-DEBUG] Found text part: {part.text[:100]}...")
                    elif hasattr(part, 'function_response'):
                        logger.info(f"[SSE-DEBUG] Found function_response, attributes: {[attr for attr in dir(part.function_response) if not attr.startswith('_')]}")
                        logger.info(f"[SSE-DEBUG] Function response object: {part.function_response}")
                        logger.info(f"[SSE-DEBUG] Function response type: {type(part.function_response)}")
                        
                        # Try different ways to access the structured data
                        tool_result = None
                        if hasattr(part.function_response, 'content'):
                            # Old way - JSON string content
                            try:
                                if isinstance(part.function_response.content, str):
                                    tool_result = json.loads(part.function_response.content)
                                else:
                                    tool_result = part.function_response.content
                                logger.info(f"[SSE-DEBUG] SUCCESS via content! Tool result keys: {list(tool_result.keys()) if isinstance(tool_result, dict) else 'Not a dict'}")
                            except (json.JSONDecodeError, AttributeError) as e:
                                logger.warning(f"[SSE-DEBUG] Failed to parse content: {e}")
                        elif hasattr(part.function_response, 'result'):
                            # Try result attribute
                            tool_result = part.function_response.result
                            logger.info(f"[SSE-DEBUG] SUCCESS via result! Tool result keys: {list(tool_result.keys()) if isinstance(tool_result, dict) else 'Not a dict'}")
                        elif isinstance(part.function_response, dict):
                            # Direct dict access
                            tool_result = part.function_response
                            logger.info(f"[SSE-DEBUG] SUCCESS via direct dict! Tool result keys: {list(tool_result.keys())}")
                        else:
                            # Try converting to dict
                            try:
                                tool_result = dict(part.function_response)
                                logger.info(f"[SSE-DEBUG] SUCCESS via dict conversion! Tool result keys: {list(tool_result.keys())}")
                            except:
                                logger.warning(f"[SSE-DEBUG] Could not extract structured data from function_response")
                        
                        if tool_result and isinstance(tool_result, dict):
                            structured_data = tool_result
                            context_type, action_required = determine_context_type(tool_result)
                            logger.info(f"[SSE-DEBUG] Context type: {context_type}, Action required: {action_required}")
                        else:
                            logger.warning(f"[SSE-DEBUG] No structured data extracted from function_response")
                    else:
                        logger.info(f"[SSE-DEBUG] Part {j} is neither text nor function_response")
    
    
    # Clean up response text
    if response_text.startswith("None"):
        response_text = response_text[4:]
    
    response_text = response_text or "I'm ready to help you with your shopping needs!"
    
    return response_text, structured_data, context_type, action_required

# Main chat endpoint
@app.post("/api/v1/chat", response_model=ChatResponse)
@limiter.limit("20/minute")
async def chat(request: Request, chat_req: ChatRequest):
    """Chat with the shopping assistant"""
    
    check_agent_ready()
    
    # Generate IDs if not provided
    device_id = chat_req.device_id or generate_device_id()
    session_id = chat_req.session_id or generate_session_id()
    
    # Create or update session
    create_or_update_session(session_id, device_id)
    
    try:
        # Get session-specific LLM with conversation history
        logger.info(f"[CHAT-REGULAR] Processing message for session: {session_id}")
        session_llm = await get_session_llm(session_id)
        
        # Enhance message with context
        enhanced_message = f"[Session: {session_id}] [Device: {device_id}] {chat_req.message}"
        logger.info(f"[CHAT-REGULAR] Enhanced message: {enhanced_message}")
        
        # Configure request parameters for precise tool calling
        request_params = RequestParams(
            max_iterations=5,  # Allow multi-turn tool conversations
            use_history=True,  # Maintain conversation history
            temperature=0.2,  # Low temperature for precise tool selection
            maxTokens=2000
        )
        
        # Get AI response using full conversation loop
        logger.info(f"[AGENT-DEBUG] Calling session_llm.generate for session: {session_id}")
        logger.info(f"[AGENT-DEBUG] Enhanced message: {enhanced_message}")
        logger.info(f"[AGENT-DEBUG] Request params: max_iterations={request_params.max_iterations}, temperature={request_params.temperature}")
        
        contents = await session_llm.generate(
            message=enhanced_message,
            request_params=request_params
        )
        
        # Enhanced debug logging for agent response
        logger.info(f"[AGENT-DEBUG] LLM generate returned {len(contents) if contents else 0} contents")
        
        # Process the response using proper MCP integration
        response_text = ""
        structured_data = None
        context_type = None
        action_required = False
        tool_calls_found = 0
        
        # The GoogleAugmentedLLM should handle tool calling automatically
        # We just need to extract the final response and any structured data
        if contents and len(contents) > 0:
            for i, content in enumerate(contents):
                logger.info(f"[AGENT-DEBUG] Processing content {i}, has parts: {hasattr(content, 'parts')}")
                if content.parts:
                    logger.info(f"[AGENT-DEBUG] Content {i} has {len(content.parts)} parts")
                    for j, part in enumerate(content.parts):
                        logger.info(f"[AGENT-DEBUG] Part {j} type: {type(part)}")
                        logger.info(f"[AGENT-DEBUG] Part {j} attributes: {[attr for attr in dir(part) if not attr.startswith('_')]}")
                        
                        if hasattr(part, 'text') and part.text:
                            logger.info(f"[AGENT-DEBUG] Found text part: {part.text[:200]}...")
                            response_text += part.text
                        elif hasattr(part, 'function_response'):
                            tool_calls_found += 1
                            logger.info(f"[AGENT-DEBUG] Found function_response part! Tool call #{tool_calls_found}")
                            logger.info(f"[AGENT-DEBUG] Function response type: {type(part.function_response)}")
                            logger.info(f"[AGENT-DEBUG] Function response attributes: {[attr for attr in dir(part.function_response) if not attr.startswith('_')]}")
                            
                            # Handle tool execution results if present
                            if hasattr(part.function_response, 'content'):
                                try:
                                    # Parse tool result as JSON if possible
                                    tool_result = json.loads(part.function_response.content)
                                    structured_data = tool_result
                                    
                                    # Determine context type using helper function
                                    context_type, action_required = determine_context_type(tool_result)
                                except (json.JSONDecodeError, AttributeError):
                                    # If not JSON or no content attribute, just add to response text
                                    response_text += str(part.function_response)
        
        # Clean up response text - remove any "None" prefix
        if response_text.startswith("None"):
            response_text = response_text[4:]
        
        response = response_text or "I'm ready to help you with your shopping needs!"
        
        return ChatResponse(
            response=response,
            session_id=session_id,
            timestamp=datetime.now(),
            data=structured_data,
            context_type=context_type,
            action_required=action_required
        )
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        
        # Handle Google AI quota exhaustion gracefully
        error_str = str(e)
        if "RESOURCE_EXHAUSTED" in error_str or "quota" in error_str.lower():
            raise HTTPException(
                status_code=503, 
                detail="AI assistant temporarily unavailable due to quota limits. Shopping tools continue to work normally."
            )
        else:
            raise HTTPException(status_code=500, detail=str(e))

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
        connection_timeout = 300  # 5 minutes max connection time
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
            
            # 2. TOOL EXECUTION EVENTS - Progress Feedback
            if is_search_request(chat_req.message):
                yield sse_event('thinking', {'message': 'Searching product catalog...', 'session_id': session_id})
                yield sse_event('tool_start', {'tool': 'search_products', 'status': 'executing', 'session_id': session_id})
            elif is_cart_request(chat_req.message):
                yield sse_event('thinking', {'message': 'Managing your shopping cart...', 'session_id': session_id})
                yield sse_event('tool_start', {'tool': 'cart_operation', 'status': 'executing', 'session_id': session_id})
            else:
                yield sse_event('thinking', {'message': 'Processing your request...', 'session_id': session_id})
            
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
            
            # Execute normally - MCP tools work as before
            contents = await session_llm.generate(
                message=enhanced_message,
                request_params=request_params
            )
            
            # 4. STRUCTURED RESULT EVENTS - Frontend Integration
            response_text, structured_data, context_type, action_required = process_mcp_results(contents)
            
            # Send different event types based on content
            if structured_data and context_type:
                # Products found - enhanced with raw BIAP data for frontend rendering
                if context_type == 'products':
                    products_list = structured_data.get('products', [])
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
                        'biap_specifications': True  # Signal that full ONDC specifications are available
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
            
            # 5. FINAL RESPONSE EVENT
            yield sse_event('response', {
                'content': response_text,
                'session_id': session_id,
                'timestamp': datetime.now().isoformat(),
                'complete': True
            })
            
            # 5.5. CHECK FOR QUEUED RAW DATA from MCP callbacks
            # Give a brief moment for any pending MCP callbacks to arrive
            await asyncio.sleep(0.1)
            
            # Send any queued raw data to frontend
            if session_id in raw_data_queues:
                queue = raw_data_queues[session_id]
                while not queue.empty():
                    try:
                        raw_event = await asyncio.wait_for(queue.get(), timeout=0.1)
                        if raw_event['event_type'] == 'raw_products':
                            yield sse_event('raw_products', raw_event['data'])
                            logger.info(f"[SSE-RAW] Sent {len(raw_event['data'].get('products', []))} raw products to frontend for session {session_id}")
                        elif raw_event['event_type'] == 'raw_cart':
                            yield sse_event('raw_cart', raw_event['data'])
                            cart_data = raw_event['data'].get('cart_items', [])
                            cart_count = len(cart_data) if isinstance(cart_data, list) else "dict"
                            logger.info(f"[SSE-RAW] Sent raw cart data ({cart_count} items) to frontend for session {session_id}")
                    except asyncio.TimeoutError:
                        break
                    except Exception as e:
                        logger.error(f"[SSE-RAW] Error sending queued raw data: {e}")
                        break
            
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
            "chat": "/api/v1/chat",
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