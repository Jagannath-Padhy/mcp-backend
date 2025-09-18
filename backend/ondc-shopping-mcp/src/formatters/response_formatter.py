"""
Response Formatter System - DRY Implementation

Centralizes all response formatting logic to avoid repetition.
Provides consistent formatting across all tool responses.
"""

from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod


class BaseFormatter(ABC):
    """Base class for all response formatters"""
    
    @abstractmethod
    def format(self, result: Dict[str, Any]) -> str:
        """Format the result into a string response"""
        pass


class SearchResultFormatter(BaseFormatter):
    """Formatter for search results (products, categories, etc.)"""
    
    def format(self, result: Dict[str, Any]) -> str:
        """Format search results with products"""
        message = result.get("message", "")
        
        if "products" in result and result["products"]:
            message = self._format_products(message, result["products"])
        elif "search_results" in result and result["search_results"]:
            message = self._format_products(message, result["search_results"])
        else:
            message += "\nNo products found. Please try a different search term."
        
        return message
    
    def _format_products(self, message: str, products: List[Dict]) -> str:
        """Format product list for display"""
        products_text = "\n\n**Products Available:**\n\n"
        
        for i, product in enumerate(products[:10], 1):  # Limit to 10 for readability
            if isinstance(product, dict):
                # Check for display_text first
                if product.get("display_text"):
                    products_text += f"**{i}. {product.get('name', 'Unknown Product')}**\n"
                    products_text += f"{product['display_text']}\n\n"
                else:
                    # Fallback formatting
                    name = product.get('name', 'Unknown Product')
                    price = product.get('price', 0)
                    desc = product.get('description', '')[:100]
                    
                    products_text += f"**{i}. {name}**\n"
                    products_text += f"   Price: ₹{price}\n"
                    if desc:
                        products_text += f"   {desc}{'...' if len(str(desc)) > 100 else ''}\n"
                    products_text += "\n"
        
        return message + products_text


class CategoryFormatter(BaseFormatter):
    """Formatter for category listings"""
    
    def format(self, result: Dict[str, Any]) -> str:
        """Format category list"""
        message = result.get("message", "")
        
        if "categories" in result and result["categories"]:
            categories_text = "\n\n**Available Categories:**\n\n"
            
            for i, category in enumerate(result["categories"], 1):
                if isinstance(category, dict):
                    name = category.get("name", "Unknown")
                    desc = category.get("description", "")
                    count = category.get("item_count", 0)
                    
                    category_text = f"{i}. **{name}**"
                    if desc:
                        category_text += f" - {desc}"
                    if count:
                        category_text += f" ({count} items)"
                    categories_text += category_text + "\n"
            
            message += categories_text
        else:
            message += "\nNo categories available."
        
        return message


class CartFormatter(BaseFormatter):
    """Formatter for cart-related responses"""
    
    def format(self, result: Dict[str, Any]) -> str:
        """Format cart response"""
        message = result.get("message", "")
        
        if "cart" in result and result["cart"]:
            cart = result["cart"]
            message += self._format_cart_contents(cart)
        
        if "total" in result:
            message += f"\n\n**Total: ₹{result['total']:.2f}**"
        
        return message
    
    def _format_cart_contents(self, cart: Dict[str, Any]) -> str:
        """Format cart items for display"""
        if not cart.get("items"):
            return "\n\nYour cart is empty."
        
        cart_text = "\n\n**Cart Contents:**\n\n"
        
        for i, item in enumerate(cart["items"], 1):
            name = item.get("name", "Unknown Item")
            quantity = item.get("quantity", 1)
            price = item.get("price", 0)
            total = price * quantity
            
            cart_text += f"{i}. **{name}**\n"
            cart_text += f"   Quantity: {quantity}\n"
            cart_text += f"   Price: ₹{price:.2f} each\n"
            cart_text += f"   Subtotal: ₹{total:.2f}\n\n"
        
        return cart_text


class OrderFormatter(BaseFormatter):
    """Formatter for order-related responses"""
    
    def format(self, result: Dict[str, Any]) -> str:
        """Format order response"""
        message = result.get("message", "")
        
        if "order" in result and result["order"]:
            order = result["order"]
            message += self._format_order_details(order)
        
        if "quote" in result and result["quote"]:
            quote = result["quote"]
            message += self._format_quote_details(quote)
        
        return message
    
    def _format_order_details(self, order: Dict[str, Any]) -> str:
        """Format order details"""
        order_text = "\n\n**Order Details:**\n"
        
        if order.get("id"):
            order_text += f"Order ID: {order['id']}\n"
        
        if order.get("status"):
            order_text += f"Status: {order['status']}\n"
        
        if order.get("total"):
            order_text += f"Total: ₹{order['total']:.2f}\n"
        
        if order.get("items"):
            order_text += f"Items: {len(order['items'])}\n"
        
        return order_text
    
    def _format_quote_details(self, quote: Dict[str, Any]) -> str:
        """Format delivery quote details"""
        quote_text = "\n\n**Delivery Quote:**\n"
        
        if quote.get("delivery_charge"):
            quote_text += f"Delivery Charge: ₹{quote['delivery_charge']:.2f}\n"
        
        if quote.get("estimated_delivery"):
            quote_text += f"Estimated Delivery: {quote['estimated_delivery']}\n"
        
        if quote.get("total_with_delivery"):
            quote_text += f"Total (including delivery): ₹{quote['total_with_delivery']:.2f}\n"
        
        return quote_text


class SimpleFormatter(BaseFormatter):
    """Simple formatter for basic messages"""
    
    def format(self, result: Dict[str, Any]) -> str:
        """Format simple message response"""
        if isinstance(result, str):
            return result
        
        return result.get("message", str(result))


class ResponseFormatterFactory:
    """Factory for creating appropriate formatters - DRY pattern"""
    
    # Map tool categories to formatters
    _formatter_map = {
        "search": SearchResultFormatter(),
        "category": CategoryFormatter(),
        "cart": CartFormatter(),
        "order": OrderFormatter(),
        "default": SimpleFormatter()
    }
    
    @classmethod
    def get_formatter(cls, tool_name: str, category: str = None) -> BaseFormatter:
        """
        Get appropriate formatter for a tool
        
        Args:
            tool_name: Name of the tool
            category: Optional category override
            
        Returns:
            Appropriate formatter instance
        """
        # Special cases for specific tools
        if tool_name == "browse_categories":
            return cls._formatter_map["category"]
        
        # Use category if provided
        if category and category in cls._formatter_map:
            return cls._formatter_map[category]
        
        # Infer from tool name patterns
        if "search" in tool_name:
            return cls._formatter_map["search"]
        elif "cart" in tool_name:
            return cls._formatter_map["cart"]
        elif "order" in tool_name or "payment" in tool_name:
            return cls._formatter_map["order"]
        
        # Default formatter
        return cls._formatter_map["default"]
    
    @classmethod
    def format_response(cls, tool_name: str, result: Dict[str, Any], category: str = None) -> str:
        """
        Convenience method to format a response
        
        Args:
            tool_name: Name of the tool that generated the result
            result: Result dictionary to format
            category: Optional category override
            
        Returns:
            Formatted string response
        """
        formatter = cls.get_formatter(tool_name, category)
        return formatter.format(result)