"""MCP Protocol Adapters - Modular Implementation

This file imports and re-exports all MCP adapter functions from organized modules.
Functions are grouped by domain for better maintainability and organization.
"""

# Import all adapter functions from organized modules
from .adapters.utils import (
    get_persistent_session,
    save_persistent_session,
    extract_session_id,
    format_mcp_response,
    format_products_for_display,
    get_services
)

from .adapters.cart import (
    add_to_cart,
    view_cart,
    remove_from_cart,
    update_cart_quantity,
    clear_cart,
    get_cart_total
)

from .adapters.search import (
    search_products,
    advanced_search,
    browse_categories
)

from .adapters.checkout import (
    select_items_for_order,
    initialize_order,
    create_payment,
    confirm_order
)

from .adapters.auth import (
    phone_login
)

from .adapters.session import (
    initialize_shopping,
    get_session_info
)

from .adapters.orders import (
    initiate_payment,
    confirm_order_simple,
    get_order_status,
    track_order
)

# Re-export all functions for backward compatibility
__all__ = [
    # Utility functions
    'get_persistent_session',
    'save_persistent_session', 
    'extract_session_id',
    'format_mcp_response',
    'format_products_for_display',
    'get_services',
    
    # Cart operations (6 functions)
    'add_to_cart',
    'view_cart',
    'remove_from_cart',
    'update_cart_quantity',
    'clear_cart',
    'get_cart_total',
    
    # Search operations (3 functions)
    'search_products',
    'advanced_search',
    'browse_categories',
    
    # ONDC checkout flow (4 functions)
    'select_items_for_order',
    'initialize_order',
    'create_payment',
    'confirm_order',
    
    # Authentication (1 function)
    'phone_login',
    
    # Session management (2 functions)
    'initialize_shopping',
    'get_session_info',
    
    # Order management (4 functions)
    'initiate_payment',
    'confirm_order_simple',
    'get_order_status',
    'track_order'
]

# Total: 21 MCP tools organized into 6 focused modules