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
import os
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

# ============================================================================
# Helper Functions for DRY Code
# ============================================================================

def generate_device_id() -> str:
    """Generate a unique device ID."""
    return f"device_{uuid.uuid4().hex[:8]}"

def generate_session_id() -> str:
    """Generate a unique session ID."""
    return f"session_{uuid.uuid4().hex}"


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
    
    Returns:
        tuple: (context_type, action_required)
    """
    if not isinstance(tool_result, dict):
        return None, False
    
    # Check for known data patterns
    for key in tool_result:
        match key:
            case 'products':
                return 'products', False
            case 'cart' | 'cart_summary':
                return 'cart', False
            case 'order_id' | 'order_details':
                return 'order', False
            case 'quote_data' | 'delivery':
                return 'checkout', True
            case 'next_step' | 'stage':
                return 'checkout', True
    
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
        async with mcp_app.run() as app_context:
            
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
        "active_sessions": active_sessions
    }

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
            import tempfile
            import os
            
            # Create session context file that MCP server can read
            context_dir = os.path.expanduser("~/.ondc-mcp")
            os.makedirs(context_dir, exist_ok=True)
            context_file = os.path.join(context_dir, "chat_session_context.txt")
            
            with open(context_file, 'w') as f:
                f.write(session_id)
            
            logger.info(f"Set MCP session context file: {session_id}")
        except Exception as e:
            logger.warning(f"Failed to set MCP session context file: {e}")
        
        # Get session-specific LLM with conversation history
        session_llm = await get_session_llm(session_id)
        
        # Enhance message with context (simplified, no need for manual injection now)
        enhanced_message = f"[Session: {session_id}] [Device: {device_id}] {chat_req.message}"
        
        # Configure request parameters for proper tool calling
        request_params = RequestParams(
            max_iterations=5,  # Allow multi-turn tool conversations
            use_history=True,  # Maintain conversation history
            temperature=0.7,
            maxTokens=2000
        )
        
        # Get AI response using full conversation loop
        contents = await session_llm.generate(
            message=enhanced_message,
            request_params=request_params
        )
        
        # Process the response using proper MCP integration
        response_text = ""
        structured_data = None
        context_type = None
        action_required = False
        
        # The GoogleAugmentedLLM should handle tool calling automatically
        # We just need to extract the final response and any structured data
        if contents and len(contents) > 0:
            for content in contents:
                if content.parts:
                    for part in content.parts:
                        if hasattr(part, 'text') and part.text:
                            response_text += part.text
                        elif hasattr(part, 'function_response'):
                            # Handle tool execution results if present
                            if hasattr(part.function_response, 'content'):
                                try:
                                    # Parse tool result as JSON if possible
                                    tool_result = json.loads(part.function_response.content)
                                    structured_data = tool_result
                                    
                                    # Tool result processing - no longer need to track initialization 
                                    # since SessionService handles this automatically
                                    
                                    # Determine context type using helper function
                                    context_type, action_required = determine_context_type(tool_result)
                                except (json.JSONDecodeError, AttributeError):
                                    # If not JSON or no content attribute, just add to response text
                                    response_text += str(part.function_response)
        
        # Clean up response text - remove any "None" prefix
        if response_text.startswith("None"):
            response_text = response_text[4:]
        
        response = response_text or "I'm ready to help you with your shopping needs!"
        
        # IMPORTANT: Clean up session context file AFTER LLM response is processed
        # This ensures MCP tools can access the session during the request
        try:
            import os
            context_file = os.path.expanduser("~/.ondc-mcp/chat_session_context.txt")
            if os.path.exists(context_file):
                os.remove(context_file)
                logger.debug(f"Cleaned up MCP session context file after LLM response")
        except Exception as e:
            logger.debug(f"Session context file cleanup failed (not critical): {e}")

        return ChatResponse(
            response=response,
            session_id=session_id,
            device_id=device_id,
            timestamp=datetime.now(),
            data=structured_data,
            context_type=context_type,
            action_required=action_required
        )
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        # Clean up session context file on error too
        try:
            import os
            context_file = os.path.expanduser("~/.ondc-mcp/chat_session_context.txt")
            if os.path.exists(context_file):
                os.remove(context_file)
                logger.debug(f"Cleaned up MCP session context file on error")
        except Exception as cleanup_error:
            logger.debug(f"Session context file cleanup on error failed: {cleanup_error}")
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
            "cart": "/api/v1/cart/{device_id}"
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