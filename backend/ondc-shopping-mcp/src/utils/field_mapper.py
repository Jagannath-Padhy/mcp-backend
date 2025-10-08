"""
Field Mapper Utility
====================

Handles field name transformations between different system components.
Specifically addresses the _raw vs raw field naming conflict between
backend requirements and Langflow restrictions.
"""

from typing import Dict, Any, List, Union, Optional
import copy


class FieldMapper:
    """Smart field mapping for system compatibility"""
    
    # Field mappings: MCP/Langflow name -> Backend name
    FIELD_MAPPINGS = {
        "raw": "_raw",  # Langflow doesn't allow leading underscores
    }
    
    # Reverse mappings: Backend name -> MCP/Langflow name
    REVERSE_MAPPINGS = {v: k for k, v in FIELD_MAPPINGS.items()}
    
    @staticmethod
    def to_backend(data: Union[Dict[str, Any], List[Any]]) -> Union[Dict[str, Any], List[Any]]:
        """
        Convert field names from MCP/Langflow format to backend format
        
        Args:
            data: Data with MCP/Langflow field names
            
        Returns:
            Data with backend-compatible field names
        """
        if isinstance(data, list):
            return [FieldMapper.to_backend(item) for item in data]
        
        if not isinstance(data, dict):
            return data
        
        # Deep copy to avoid modifying original
        result = {}
        
        for key, value in data.items():
            # Check if this key should be mapped
            if key in FieldMapper.FIELD_MAPPINGS:
                # Use the mapped key instead
                backend_key = FieldMapper.FIELD_MAPPINGS[key]
            else:
                # Keep the original key
                backend_key = key
            
            # Recursively map nested structures
            if isinstance(value, dict):
                result[backend_key] = FieldMapper.to_backend(value)
            elif isinstance(value, list):
                result[backend_key] = [
                    FieldMapper.to_backend(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                result[backend_key] = value
        
        return result
    
    @staticmethod
    def from_backend(data: Union[Dict[str, Any], List[Any]]) -> Union[Dict[str, Any], List[Any]]:
        """
        Convert field names from backend format to MCP/Langflow format
        
        Args:
            data: Data with backend field names
            
        Returns:
            Data with MCP/Langflow-compatible field names
        """
        if isinstance(data, list):
            return [FieldMapper.from_backend(item) for item in data]
        
        if not isinstance(data, dict):
            return data
        
        result = {}
        
        for key, value in data.items():
            # Map field name if needed
            mcp_key = FieldMapper.REVERSE_MAPPINGS.get(key, key)
            
            # Recursively map nested structures
            if isinstance(value, dict):
                result[mcp_key] = FieldMapper.from_backend(value)
            elif isinstance(value, list):
                result[mcp_key] = [
                    FieldMapper.from_backend(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                result[mcp_key] = value
        
        return result
    
    @staticmethod
    def preserve_raw_data(item: Dict[str, Any], raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preserve raw backend data in item for future operations
        
        Args:
            item: Item to enhance
            raw_data: Raw backend data to preserve
            
        Returns:
            Item with preserved raw data
        """
        # Use the backend field name internally
        item_with_raw = item.copy()
        item_with_raw["_raw"] = raw_data
        return item_with_raw
    
    @staticmethod
    def extract_raw_data(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Extract preserved raw data from item
        
        Args:
            item: Item potentially containing raw data
            
        Returns:
            Raw data if present, None otherwise
        """
        # Check both field names for compatibility
        return item.get("_raw") or item.get("raw")
    
    @staticmethod
    def apply_provider_location_fix(data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fix provider location structure for backend compatibility
        
        Backend expects locations as array of objects with id and local_id,
        but sometimes we have just string IDs that need conversion.
        
        Args:
            data: Data potentially containing provider structure
            
        Returns:
            Data with fixed provider location structure
        """
        if not isinstance(data, dict):
            return data
        
        result = copy.deepcopy(data)
        
        # Fix provider locations if present
        if "provider" in result and isinstance(result["provider"], dict):
            provider = result["provider"]
            
            if "locations" in provider and isinstance(provider["locations"], list):
                fixed_locations = []
                
                for location in provider["locations"]:
                    if isinstance(location, str):
                        # Convert string to proper object structure
                        fixed_locations.append({
                            "id": location if "_" in location else f"{provider.get('id', '')}_{location}",
                            "local_id": location.split("_")[-1] if "_" in location else location
                        })
                    elif isinstance(location, dict):
                        # Already in correct format
                        fixed_locations.append(location)
                
                provider["locations"] = fixed_locations
        
        # Recursively fix nested structures
        for key, value in result.items():
            if key != "provider" and isinstance(value, dict):
                result[key] = FieldMapper.apply_provider_location_fix(value)
            elif isinstance(value, list):
                result[key] = [
                    FieldMapper.apply_provider_location_fix(item) if isinstance(item, dict) else item
                    for item in value
                ]
        
        return result


class BackendPayloadEnhancer:
    """Enhance payloads for backend compatibility"""
    
    @staticmethod
    def enhance_cart_item(item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance cart item with all required backend fields
        
        Args:
            item: Basic cart item
            
        Returns:
            Enhanced cart item ready for backend
        """
        from ..utils.himira_provider_constants import (
            HIMIRA_BPP_ID,
            HIMIRA_BPP_URI,
            enrich_cart_item_with_provider
        )
        
        # Apply field mapping first
        backend_item = FieldMapper.to_backend(item)
        
        # Apply minimal required transformations only
        
        return backend_item
    
    @staticmethod
    def enhance_search_result(result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance search result for MCP response
        
        Args:
            result: Raw search result from backend
            
        Returns:
            Enhanced result for MCP
        """
        # Convert backend fields to MCP format
        mcp_result = FieldMapper.from_backend(result)
        
        # Preserve original data for cart operations
        # Temporarily commented out to test if cart works without raw field duplication
        # if "raw" not in mcp_result and "_raw" not in mcp_result:
        #     mcp_result["raw"] = result
        
        return mcp_result


# Export convenience functions
def to_backend(data: Union[Dict[str, Any], List[Any]]) -> Union[Dict[str, Any], List[Any]]:
    """Convert to backend format"""
    return FieldMapper.to_backend(data)


def from_backend(data: Union[Dict[str, Any], List[Any]]) -> Union[Dict[str, Any], List[Any]]:
    """Convert from backend format"""
    return FieldMapper.from_backend(data)


def enhance_for_backend(item: Dict[str, Any]) -> Dict[str, Any]:
    """Enhance item for backend compatibility"""
    return BackendPayloadEnhancer.enhance_cart_item(item)


def enhance_for_mcp(result: Dict[str, Any]) -> Dict[str, Any]:
    """Enhance result for MCP response"""
    return BackendPayloadEnhancer.enhance_search_result(result)