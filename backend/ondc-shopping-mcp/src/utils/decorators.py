"""Decorators for reducing code duplication in tools

NOTE: These decorators are legacy code and not actively used.
The new architecture uses services and MCP adapters instead.
"""

import functools
import asyncio
import inspect
from typing import Dict, Any, Optional, Callable
import logging

from .logger import get_logger

logger = get_logger(__name__)

# Stub function to prevent import errors in legacy code
def get_or_create_session(session: Optional[Dict] = None) -> Dict:
    """Legacy stub - not actively used"""
    return session or {}


def with_session(func: Callable) -> Callable:
    """
    Decorator to ensure a valid session exists before executing a tool.
    Properly handles both positional and keyword arguments to avoid conflicts.
    """
    # Get function signature outside wrapper for preservation
    sig = inspect.signature(func)
    
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        # Try to bind arguments to understand what was passed
        try:
            bound = sig.bind_partial(*args, **kwargs)
            bound.apply_defaults()
        except TypeError:
            # If binding fails, try alternative approach
            bound_args = {}
            if 'session' in kwargs:
                bound_args['session'] = kwargs['session']
            elif args and isinstance(args[0], (dict, type(None))):
                bound_args['session'] = args[0]
                args = args[1:]  # Remove session from args
            else:
                bound_args['session'] = None
            
            # Merge with remaining kwargs
            for k, v in kwargs.items():
                if k != 'session':
                    bound_args[k] = v
            
            # Ensure valid session
            bound_args['session'] = get_or_create_session(bound_args.get('session'))
            
            # Call function with cleaned arguments
            return await func(**bound_args)
        
        # Extract session from bound arguments
        session = bound.arguments.get('session')
        
        # Ensure we have a valid session
        session = get_or_create_session(session)
        
        # Update the session in bound arguments
        bound.arguments['session'] = session
        
        # Call the function with properly bound arguments
        return await func(**bound.arguments)
    
    # Preserve the original function signature for introspection
    wrapper.__signature__ = sig
    return wrapper


def with_error_handling(default_message: str = "An error occurred"):
    """
    Decorator to standardize error handling across tools.
    Catches exceptions and returns a consistent error format.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except asyncio.TimeoutError:
                logger.error(f"Timeout in {func.__name__}")
                session = kwargs.get('session', {})
                return {
                    "success": False,
                    "session": session,
                    "message": f"⏱ Operation timed out. Please try again.",
                    "error_type": "timeout"
                }
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {e}", exc_info=True)
                session = kwargs.get('session', {})
                return {
                    "success": False,
                    "session": session,
                    "message": f" {default_message}: {str(e)}",
                    "error_type": "exception"
                }
        return wrapper
    return decorator


def validate_quantity(min_qty: int = 1, max_qty: int = 100):
    """
    Decorator to validate quantity parameters in cart operations.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            quantity = kwargs.get('quantity', 1)
            
            # Ensure quantity is an integer
            try:
                quantity = int(quantity)
            except (ValueError, TypeError):
                session = kwargs.get('session', {})
                return {
                    "success": False,
                    "session": session,
                    "message": f" Invalid quantity. Please provide a number between {min_qty} and {max_qty}."
                }
            
            # Validate range
            if quantity < min_qty or quantity > max_qty:
                session = kwargs.get('session', {})
                return {
                    "success": False,
                    "session": session,
                    "message": f" Quantity must be between {min_qty} and {max_qty}."
                }
            
            # Update kwargs with validated quantity
            kwargs['quantity'] = quantity
            return await func(*args, **kwargs)
        return wrapper
    return decorator


def with_timeout(seconds: int = 30):
    """
    Decorator to add timeout to async operations.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=seconds
                )
            except asyncio.TimeoutError:
                logger.error(f"Timeout in {func.__name__} after {seconds}s")
                session = kwargs.get('session', {})
                return {
                    "success": False,
                    "session": session,
                    "message": f"⏱ Operation timed out after {seconds} seconds. Please try again."
                }
        return wrapper
    return decorator


def require_cart_items(func: Callable) -> Callable:
    """
    Decorator to ensure cart has items before proceeding with operations.
    """
    @functools.wraps(func)
    async def wrapper(session: Dict, *args, **kwargs):
        # Ensure cart_items exists in session
        if "cart_items" not in session:
            session["cart_items"] = []
        
        # Check if cart has items
        if not session.get("cart_items", []):
            return {
                "success": False,
                "session": session,
                "message": " Your cart is empty. Please add some items first."
            }
        
        return await func(session=session, *args, **kwargs)
    return wrapper