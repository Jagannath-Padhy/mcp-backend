"""
Tool Registry - Single Source of Truth for All Tools

This module implements DRY principle by defining all tools in one place.
No more repetitive tool definitions or if-elif chains!
"""

from typing import Dict, List, Any, Callable, Optional
from dataclasses import dataclass, field
from mcp.types import Tool

# Import all adapters
from ..mcp_adapters import (
    # Cart adapters
    add_to_cart as add_to_cart_adapter,
    view_cart as view_cart_adapter,
    update_cart_quantity as update_cart_quantity_adapter,
    remove_from_cart as remove_from_cart_adapter,
    clear_cart as clear_cart_adapter,
    get_cart_total as get_cart_total_adapter,
    # Search adapters
    search_products as search_products_adapter,
    advanced_search as advanced_search_adapter,
    browse_categories as browse_categories_adapter,
    # Session adapters
    initialize_shopping as initialize_shopping_adapter,
    # Authentication adapters
    phone_login as phone_login_adapter,
    # ONDC Order Flow adapters
    select_items_for_order as select_items_for_order_adapter,
    initialize_order as initialize_order_adapter,
    create_payment as create_payment_adapter,
    confirm_order as confirm_order_adapter,
    # Order Management adapters
    get_order_status as get_order_status_adapter
)


@dataclass
class ToolDefinition:
    """Complete definition of a tool including metadata and implementation"""
    name: str
    description: str
    adapter: Callable
    parameters: Dict[str, Any] = field(default_factory=dict)
    required_params: List[str] = field(default_factory=list)
    category: str = "general"
    format_response: Optional[Callable] = None


class ToolRegistry:
    """Central registry for all MCP tools - DRY implementation"""
    
    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}
        self._register_all_tools()
    
    def _register_all_tools(self):
        """Register all 26 ONDC tools in one place"""
        
        # ========== Session Management ==========
        self.register(ToolDefinition(
            name="initialize_shopping",
            description="START SHOPPING: Initialize new shopping session",
            adapter=initialize_shopping_adapter,
            parameters={
                "session_id": {
                    "type": "string",
                    "description": "Optional session ID for persistence"
                }
            },
            category="session"
        ))
        
        # ========== Search Tools ==========
        self.register(ToolDefinition(
            name="search_products",
            description="MAIN SEARCH: Find products by name/keyword (e.g. 'ghee', 'rice')",
            adapter=search_products_adapter,
            parameters={
                "query": {"type": "string", "description": "Search query"},
                "session_id": {"type": "string", "description": "Optional session ID"},
                "latitude": {"type": "number", "description": "Optional latitude"},
                "longitude": {"type": "number", "description": "Optional longitude"},
                "page": {"type": "integer", "description": "Page number (default: 1)"},
                "limit": {"type": "integer", "description": "Results per page (default: 10)"}
            },
            required_params=["query"],
            category="search"
        ))
        
        self.register(ToolDefinition(
            name="browse_categories",
            description="CATEGORY LIST: See all available product categories",
            adapter=browse_categories_adapter,
            parameters={
                "session_id": {"type": "string", "description": "Optional session ID"}
            },
            category="search"
        ))
        
        self.register(ToolDefinition(
            name="advanced_search",
            description="FILTERED SEARCH: Search with category/brand/price filters",
            adapter=advanced_search_adapter,
            parameters={
                "query": {"type": "string", "description": "Search query"},
                "category": {"type": "string", "description": "Category filter"},
                "brand": {"type": "string", "description": "Brand filter"},
                "price_min": {"type": "number", "description": "Minimum price"},
                "price_max": {"type": "number", "description": "Maximum price"},
                "location": {"type": "string", "description": "Location filter"},
                "session_id": {"type": "string", "description": "Optional session ID"},
                "page": {"type": "integer", "description": "Page number"},
                "limit": {"type": "integer", "description": "Results per page"}
            },
            category="search"
        ))
        
        # ========== Cart Management ==========
        self.register(ToolDefinition(
            name="add_to_cart",
            description="Add item to cart with quantity",
            adapter=add_to_cart_adapter,
            parameters={
                "item": {
                    "type": "object",
                    "description": "Product item to add"
                },
                "quantity": {
                    "type": "integer",
                    "description": "Quantity to add (default: 1)",
                    "minimum": 1,
                    "maximum": 100
                },
                "session_id": {"type": "string", "description": "Optional session ID"}
            },
            required_params=["item"],
            category="cart"
        ))
        
        self.register(ToolDefinition(
            name="view_cart",
            description="View current cart contents",
            adapter=view_cart_adapter,
            parameters={
                "session_id": {"type": "string", "description": "Optional session ID"}
            },
            category="cart"
        ))
        
        self.register(ToolDefinition(
            name="update_cart_quantity",
            description="Update quantity of cart item",
            adapter=update_cart_quantity_adapter,
            parameters={
                "item_id": {"type": "string", "description": "Item ID to update"},
                "quantity": {"type": "integer", "description": "New quantity"},
                "session_id": {"type": "string", "description": "Optional session ID"}
            },
            required_params=["item_id", "quantity"],
            category="cart"
        ))
        
        self.register(ToolDefinition(
            name="remove_from_cart",
            description="Remove item from cart",
            adapter=remove_from_cart_adapter,
            parameters={
                "item_id": {"type": "string", "description": "Item ID to remove"},
                "session_id": {"type": "string", "description": "Optional session ID"}
            },
            required_params=["item_id"],
            category="cart"
        ))
        
        self.register(ToolDefinition(
            name="clear_cart",
            description="Clear entire cart",
            adapter=clear_cart_adapter,
            parameters={
                "session_id": {"type": "string", "description": "Optional session ID"}
            },
            category="cart"
        ))
        
        self.register(ToolDefinition(
            name="get_cart_total",
            description="Get cart total value",
            adapter=get_cart_total_adapter,
            parameters={
                "session_id": {"type": "string", "description": "Optional session ID"}
            },
            category="cart"
        ))
        
        # ========== Authentication ==========
        self.register(ToolDefinition(
            name="phone_login",
            description="Quick login with phone number - instant authentication",
            adapter=phone_login_adapter,
            parameters={
                "phone": {
                    "type": "string",
                    "description": "10-digit phone number (e.g., 9876543210)"
                },
                "session_id": {"type": "string", "description": "Optional session ID"}
            },
            required_params=["phone"],
            category="auth"
        ))
        
        # ========== ONDC Order Flow ==========
        self.register(ToolDefinition(
            name="select_items_for_order",
            description="🚀 CHECKOUT ENTRY POINT: Always start checkout here! Get delivery quotes for cart items (ONDC SELECT stage). Use this when user wants to 'checkout', 'proceed with address', or 'start order process'.",
            adapter=select_items_for_order_adapter,
            parameters={
                "delivery_city": {"type": "string", "description": "Delivery city (optional - auto-fetched from saved addresses)"},
                "delivery_state": {"type": "string", "description": "Delivery state (optional - auto-fetched from saved addresses)"},
                "delivery_pincode": {"type": "string", "description": "Delivery pincode (optional - auto-fetched from saved addresses)"},
                "session_id": {"type": "string", "description": "Optional session ID"}
            },
            category="order"
        ))
        
        self.register(ToolDefinition(
            name="initialize_order",
            description="⚡ STEP 2: Initialize order with customer details (ONDC INIT stage). ⚠️ REQUIRES: Must call select_items_for_order first! Auto-extracts customer details from saved addresses.",
            adapter=initialize_order_adapter,
            parameters={
                "customer_name": {"type": "string", "description": "Customer's full name (optional - auto-extracted from saved addresses)"},
                "delivery_address": {"type": "string", "description": "Complete street address (optional - auto-extracted from saved addresses)"},
                "phone": {"type": "string", "description": "Contact phone number (optional - auto-extracted from saved addresses)"},
                "email": {"type": "string", "description": "Contact email address (optional - auto-extracted from saved addresses)"},
                "payment_method": {
                    "type": "string",
                    "description": "Payment method: razorpay, upi, card, netbanking (default: razorpay)"
                },
                "city": {"type": "string", "description": "City (optional - auto-extracted)"},
                "state": {"type": "string", "description": "State (optional - auto-extracted)"},
                "pincode": {"type": "string", "description": "Pincode (optional - auto-extracted)"},
                "session_id": {"type": "string", "description": "Optional session ID"}
            },
            category="order"
        ))
        
        self.register(ToolDefinition(
            name="create_payment",
            description="💳 STEP 3: Create payment order (MOCK mode). ⚠️ REQUIRES: Must call initialize_order first! Prepares payment and STOPS for user payment completion.",
            adapter=create_payment_adapter,
            parameters={
                "payment_method": {
                    "type": "string",
                    "description": "Payment method (default: razorpay)"
                },
                "session_id": {"type": "string", "description": "Optional session ID"}
            },
            category="order"
        ))
        
        self.register(ToolDefinition(
            name="confirm_order",
            description="CHECKOUT FINAL STEP: Confirm order after payment (ONDC CONFIRM)",
            adapter=confirm_order_adapter,
            parameters={
                "payment_status": {
                    "type": "string",
                    "description": "Payment status: PENDING, PAID, CAPTURED, SUCCESS, FAILED"
                },
                "session_id": {"type": "string", "description": "Optional session ID"}
            },
            category="order"
        ))
        
        self.register(ToolDefinition(
            name="get_order_status",
            description="Check order status",
            adapter=get_order_status_adapter,
            parameters={
                "order_id": {"type": "string", "description": "Order ID to check"},
                "session_id": {"type": "string", "description": "Optional session ID"}
            },
            required_params=["order_id"],
            category="order"
        ))
    
    def register(self, tool_def: ToolDefinition):
        """Register a tool definition"""
        self._tools[tool_def.name] = tool_def
    
    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        """Get a tool definition by name"""
        return self._tools.get(name)
    
    def list_tools(self) -> List[ToolDefinition]:
        """List all registered tools"""
        return list(self._tools.values())
    
    def get_mcp_tools(self) -> List[Tool]:
        """Generate MCP Tool objects from registry"""
        tools = []
        for tool_def in self._tools.values():
            # Build JSON Schema for the tool
            schema = {
                "type": "object",
                "properties": tool_def.parameters
            }
            
            # Add required fields if any
            if tool_def.required_params:
                schema["required"] = tool_def.required_params
            
            # Create MCP Tool object
            tools.append(Tool(
                name=tool_def.name,
                description=tool_def.description,
                inputSchema=schema
            ))
        
        return tools
    
    async def dispatch_tool(self, name: str, arguments: dict) -> Any:
        """
        Dispatch tool execution to appropriate adapter.
        Single dispatcher instead of if-elif chain - DRY!
        """
        tool_def = self.get_tool(name)
        if not tool_def:
            raise ValueError(f"Unknown tool: {name}")
        
        # Call the adapter with arguments
        result = await tool_def.adapter(**arguments)
        
        # Apply custom formatting if defined
        if tool_def.format_response:
            return tool_def.format_response(result)
        
        return result
    
    def get_tools_by_category(self, category: str) -> List[ToolDefinition]:
        """Get all tools in a specific category"""
        return [
            tool for tool in self._tools.values()
            if tool.category == category
        ]
    
    def get_categories(self) -> List[str]:
        """Get all unique tool categories"""
        return list(set(tool.category for tool in self._tools.values()))


# Singleton instance
_tool_registry: Optional[ToolRegistry] = None


def get_tool_registry() -> ToolRegistry:
    """Get singleton ToolRegistry instance"""
    global _tool_registry
    if _tool_registry is None:
        _tool_registry = ToolRegistry()
    return _tool_registry