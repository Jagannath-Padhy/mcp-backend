"""
Schema Generator Utilities
==========================

Reusable functions to generate MCP tool schemas from ONDC schemas.
Follows DRY principle to avoid duplicating schema definitions.
"""

from typing import Dict, Any, List, Optional


class SchemaGenerator:
    """Generate MCP-compatible schemas from ONDC schemas"""
    
    @staticmethod
    def generate_mcp_schema(base_schema: Dict[str, Any], 
                           required_fields: Optional[List[str]] = None,
                           additional_properties: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Convert ONDC schema to MCP-compatible JSON schema
        
        Args:
            base_schema: Base ONDC schema
            required_fields: List of required fields (overrides base)
            additional_properties: Additional properties to merge
            
        Returns:
            MCP-compatible JSON schema
        """
        # Start with base schema
        mcp_schema = base_schema.copy()
        
        # Override required fields if specified
        if required_fields is not None:
            mcp_schema["required"] = required_fields
        
        # Merge additional properties
        if additional_properties:
            if "properties" not in mcp_schema:
                mcp_schema["properties"] = {}
            mcp_schema["properties"].update(additional_properties)
        
        # Ensure MCP compatibility
        mcp_schema = SchemaGenerator._ensure_mcp_compatibility(mcp_schema)
        
        return mcp_schema
    
    @staticmethod
    def generate_minimal_schema(fields: Dict[str, Dict[str, Any]], 
                               required: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate minimal schema for Langflow compatibility
        
        Args:
            fields: Dictionary of field definitions
            required: List of required field names
            
        Returns:
            Minimal MCP-compatible schema
        """
        schema = {
            "type": "object",
            "properties": {},
            "additionalProperties": False
        }
        
        # Add field definitions
        for field_name, field_def in fields.items():
            # Ensure no leading underscores (Langflow restriction)
            clean_name = field_name.lstrip('_')
            schema["properties"][clean_name] = field_def
        
        # Set required fields
        if required:
            schema["required"] = [name.lstrip('_') for name in required]
        
        return schema
    
    @staticmethod
    def _ensure_mcp_compatibility(schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensure schema is MCP/Langflow compatible
        
        Args:
            schema: Original schema
            
        Returns:
            Compatible schema
        """
        # Remove any fields with leading underscores
        if "properties" in schema:
            clean_properties = {}
            for key, value in schema["properties"].items():
                # Convert _raw to raw for Langflow
                clean_key = "raw" if key == "_raw" else key.lstrip('_')
                
                # Recursively clean nested objects
                if isinstance(value, dict) and value.get("type") == "object":
                    value = SchemaGenerator._ensure_mcp_compatibility(value)
                
                clean_properties[clean_key] = value
            
            schema["properties"] = clean_properties
        
        # Clean required fields
        if "required" in schema:
            schema["required"] = [
                "raw" if field == "_raw" else field.lstrip('_') 
                for field in schema["required"]
            ]
        
        return schema
    
    @staticmethod
    def merge_schemas(*schemas: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge multiple schemas into one
        
        Args:
            *schemas: Variable number of schemas to merge
            
        Returns:
            Merged schema
        """
        result = {
            "type": "object",
            "properties": {},
            "required": []
        }
        
        for schema in schemas:
            if not schema:
                continue
                
            # Merge properties
            if "properties" in schema:
                result["properties"].update(schema["properties"])
            
            # Merge required fields (union)
            if "required" in schema:
                existing = set(result.get("required", []))
                new = set(schema["required"])
                result["required"] = list(existing | new)
        
        return result if result["properties"] else {"type": "object"}


class ToolSchemaFactory:
    """Factory for generating tool-specific schemas"""
    
    @staticmethod
    def create_search_tool_schema() -> Dict[str, Any]:
        """Create schema for search_products tool"""
        return SchemaGenerator.generate_minimal_schema(
            fields={
                "query": {
                    "type": "string",
                    "description": "Search query for products"
                },
                "category": {
                    "type": "string",
                    "description": "Optional category filter"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results (default: 10)",
                    "default": 10
                }
            },
            required=["query"]
        )
    
    @staticmethod
    def create_add_to_cart_schema() -> Dict[str, Any]:
        """Create schema for add_to_cart tool"""
        from ..data_models.ondc_schemas import get_minimal_item_schema
        
        # Get minimal item schema
        item_schema = get_minimal_item_schema()
        item_schema = SchemaGenerator._ensure_mcp_compatibility(item_schema)
        
        # Wrap in item parameter
        return {
            "type": "object",
            "properties": {
                "item": {
                    **item_schema,
                    "description": "Product item to add to cart (use the product details from search results)"
                }
            },
            "required": ["item"]
        }
    
    @staticmethod
    def create_view_cart_schema() -> Dict[str, Any]:
        """Create schema for view_cart tool"""
        return SchemaGenerator.generate_minimal_schema(
            fields={
                "detailed": {
                    "type": "boolean",
                    "description": "Show detailed cart information",
                    "default": False
                }
            },
            required=[]
        )
    
    @staticmethod
    def create_checkout_schema() -> Dict[str, Any]:
        """Create schema for checkout tools"""
        return SchemaGenerator.generate_minimal_schema(
            fields={
                "delivery_address": {
                    "type": "string",
                    "description": "Delivery address"
                },
                "phone": {
                    "type": "string",
                    "description": "Contact phone number"
                },
                "email": {
                    "type": "string",
                    "description": "Email address"
                },
                "payment_method": {
                    "type": "string",
                    "description": "Payment method (COD, UPI, Card)",
                    "enum": ["COD", "UPI", "Card"]
                }
            },
            required=["delivery_address", "phone", "payment_method"]
        )
    
    @staticmethod
    def create_order_schema() -> Dict[str, Any]:
        """Create schema for order-related tools"""
        return SchemaGenerator.generate_minimal_schema(
            fields={
                "order_id": {
                    "type": "string",
                    "description": "Order ID"
                },
                "action": {
                    "type": "string",
                    "description": "Action to perform",
                    "enum": ["view", "track", "cancel"]
                }
            },
            required=["order_id"]
        )


# Export convenience functions
def generate_mcp_schema(base_schema: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """Generate MCP-compatible schema from base schema"""
    return SchemaGenerator.generate_mcp_schema(base_schema, **kwargs)


def generate_minimal_schema(fields: Dict[str, Dict[str, Any]], **kwargs) -> Dict[str, Any]:
    """Generate minimal Langflow-compatible schema"""
    return SchemaGenerator.generate_minimal_schema(fields, **kwargs)


def merge_schemas(*schemas: Dict[str, Any]) -> Dict[str, Any]:
    """Merge multiple schemas"""
    return SchemaGenerator.merge_schemas(*schemas)