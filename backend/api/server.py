#!/usr/bin/env python3
"""
ONDC Shopping Backend API Server
FastAPI server with MCP-Agent integration for frontend applications
"""

import os
import uuid
import asyncio
import logging
from datetime import datetime
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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
                instruction="""You are an intelligent ONDC shopping assistant.

Your capabilities include:
- Product search with hybrid vector + backend API search
- Shopping cart management
- Order processing and tracking
- Session management

Always be helpful and provide specific guidance.
Use the available tools to help users with their shopping needs.""",
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
    session_id = f"session_{uuid.uuid4().hex}"
    device_id = session_req.device_id or f"device_{uuid.uuid4().hex[:8]}"
    
    session = {
        "session_id": session_id,
        "device_id": device_id,
        "created_at": datetime.now(),
        "last_activity": datetime.now(),
        "metadata": session_req.metadata
    }
    
    sessions[session_id] = session
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
    
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not ready")
    
    # Generate IDs if not provided
    device_id = chat_req.device_id or f"device_{uuid.uuid4().hex[:8]}"
    session_id = chat_req.session_id or f"session_{uuid.uuid4().hex}"
    
    # Update session
    if session_id not in sessions:
        sessions[session_id] = {
            "session_id": session_id,
            "device_id": device_id,
            "created_at": datetime.now(),
            "last_activity": datetime.now(),
            "metadata": {}
        }
    else:
        sessions[session_id]["last_activity"] = datetime.now()
    
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
        
        # Process the response and handle tool calls
        response_text = ""
        if contents and len(contents) > 0:
            for content in contents:
                if content.parts:
                    for part in content.parts:
                        if hasattr(part, 'text') and part.text:
                            # Check if this is a tool call request
                            if '```tool_code' in part.text or '"tool_name"' in part.text:
                                # Extract tool name and parameters
                                import json
                                import re
                                
                                # Try to parse JSON tool call
                                try:
                                    # Extract JSON from tool_code block or direct JSON
                                    json_match = re.search(r'\{.*"tool_name".*\}', part.text, re.DOTALL)
                                    if json_match:
                                        tool_data = json.loads(json_match.group())
                                        tool_name = tool_data.get('tool_name')
                                        tool_input = tool_data.get('tool_input', tool_data.get('query', {}))
                                        
                                        # Map common tool names to MCP tool names
                                        tool_mapping = {
                                            'search': 'search_products',
                                            'search_products': 'search_products',
                                            'add_to_cart': 'add_to_cart',
                                            'view_cart': 'view_cart',
                                            'init_session': 'init_session'
                                        }
                                        
                                        actual_tool = tool_mapping.get(tool_name, tool_name)
                                        
                                        # Call the tool through the agent
                                        try:
                                            tool_result = await agent.call_tool(
                                                server_name="ondc-shopping",
                                                tool_name=actual_tool,
                                                arguments=tool_input if isinstance(tool_input, dict) else {"query": tool_input}
                                            )
                                            
                                            # Format the tool result for human consumption
                                            if tool_result:
                                                if isinstance(tool_result, dict):
                                                    if 'products' in tool_result:
                                                        # Format search results
                                                        products = tool_result['products']
                                                        response_text = f"I found {len(products)} products for your search:\\n\\n"
                                                        for i, product in enumerate(products[:5], 1):  # Show top 5
                                                            response_text += f"{i}. {product.get('name', 'Unknown')} - ₹{product.get('price', 'N/A')}\\n"
                                                            if product.get('description'):
                                                                response_text += f"   {product['description'][:100]}...\\n"
                                                    elif 'message' in tool_result:
                                                        response_text = tool_result['message']
                                                    else:
                                                        response_text = json.dumps(tool_result, indent=2)
                                                else:
                                                    response_text = str(tool_result)
                                        except Exception as tool_error:
                                            logger.error(f"Tool execution error: {tool_error}")
                                            response_text = f"I encountered an error while processing your request: {str(tool_error)}"
                                        
                                except json.JSONDecodeError:
                                    # If not JSON, just use the text as is
                                    response_text += part.text
                            else:
                                response_text += part.text
                        elif hasattr(part, 'function_response'):
                            response_text += str(part.function_response)
        
        response = response_text or "I'm ready to help you with your shopping needs!"
        
        return ChatResponse(
            response=response,
            session_id=session_id,
            device_id=device_id,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Search endpoint
@app.post("/api/v1/search")
@limiter.limit("30/minute")
async def search_products(request: Request, search_req: SearchRequest):
    """Hybrid search for products"""
    
    if not llm:
        raise HTTPException(status_code=503, detail="Agent not ready")
    
    try:
        # Use agent to search with proper tool calling
        search_prompt = f"Search for products: {search_req.query}"
        if search_req.filters:
            search_prompt += f" with filters: {search_req.filters}"
        
        request_params = RequestParams(
            max_iterations=3,
            use_history=False,  # Don't maintain history for search
            temperature=0.3
        )
        
        # Call search tool directly
        try:
            tool_result = await agent.call_tool(
                server_name="ondc-shopping",
                tool_name="search_products",
                arguments={"search_query": search_req.query}
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
    
    if not llm:
        raise HTTPException(status_code=503, detail="Agent not ready")
    
    try:
        # Use agent for cart management with proper tool calling
        cart_prompt = f"Cart action for device {device_id}: {cart_req.action}"
        if cart_req.item:
            cart_prompt += f" with item: {cart_req.item}"
        
        request_params = RequestParams(
            max_iterations=3,
            use_history=False,
            temperature=0.3
        )
        
        # Map cart actions to appropriate tools
        if cart_req.action == "add" and cart_req.item:
            tool_name = "add_to_cart"
            arguments = {"item": cart_req.item, "quantity": cart_req.quantity}
        elif cart_req.action == "view":
            tool_name = "view_cart"
            arguments = {"device_id": device_id}
        else:
            response = f"Unsupported cart action: {cart_req.action}"
            return {
                "device_id": device_id,
                "action": cart_req.action,
                "result": response,
                "timestamp": datetime.now()
            }
        
        # Call the appropriate cart tool
        try:
            tool_result = await agent.call_tool(
                server_name="ondc-shopping",
                tool_name=tool_name,
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