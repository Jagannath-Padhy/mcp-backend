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
                instruction="""You are an intelligent ONDC shopping assistant operating in GUEST MODE.
Guest Configuration:
- userId: guestUser
- deviceId: provided by user or auto-generated

Order Journey Flow (complete up to on_init):
1. initialize_shopping → Create guest session
2. search_products → Search for products
3. add_to_cart → Add items to cart
4. view_cart → Show cart contents
5. select_items_for_order → Get delivery quotes (needs city, state, pincode)
6. initialize_order → Set customer details (name, address, phone, email)
7. create_payment → [MOCK] Create test payment
8. confirm_order → [MOCK] Confirm with mock payment

SEARCH PATTERNS - Always use search_products for ANY mention of products:
- "search for X" 
- "find X"  
- "looking for X"
- "I need X"
- "show me X"
- "get X"
- "X products"
- "buy X"
- "shop for X"
- "X" (any product name, including plural forms like "jams", "apples", "oils")
- ANY food, grocery, or merchandise query
- When user mentions any product name, ALWAYS use search_products

IMPORTANT:
- Never ask for login/authentication
- Guide users sequentially through the flow
- Collect required info at each stage
- Mark [MOCK] for payment/confirm operations
- Use tools proactively to help users

Always use the appropriate tool based on user intent.

For 'add to cart', use add_to_cart.
For 'checkout', start with select_items_for_order.""",
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
        session_llm = await get_session_llm(session_id)
        
        # Enhance message with context
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
            device_id=device_id,
            timestamp=datetime.now(),
            data=structured_data,
            context_type=context_type,
            action_required=action_required
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