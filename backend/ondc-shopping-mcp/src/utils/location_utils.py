"""
Location Utilities - DRY Provider Location Transformations
===========================================================

Centralized utilities for handling ONDC location transformations.
Eliminates duplicate code for provider location handling.

Date: 2025-09-15
Purpose: Single source of truth for location transformations
"""

from typing import List, Dict, Any, Optional, Union
from .ondc_constants import (
    create_full_location_id,
    HIMIRA_LOCATION_LOCAL_ID
)

def transform_provider_locations(
    locations: List[Union[str, Dict]], 
    format_type: str = "full_object"
) -> List[Union[str, Dict]]:
    """
    Transform provider locations to required format.
    
    Args:
        locations: List of location data (strings or dicts)
        format_type: Output format type
            - "full_object": {id: full_id, local_id: local_id}
            - "simple_object": {id: local_id}
            - "string_array": ["local_id1", "local_id2"]
            - "full_id_object": {id: full_ondc_id}
    
    Returns:
        Transformed locations in requested format
    """
    transformed = []
    
    for loc in locations:
        if format_type == "string_array":
            # Extract local_id as string
            if isinstance(loc, dict):
                local_id = loc.get('local_id', loc.get('id', ''))
            else:
                local_id = str(loc)
            transformed.append(local_id)
            
        elif format_type == "simple_object":
            # Simple object with just local_id
            if isinstance(loc, dict):
                local_id = loc.get('local_id', loc.get('id', ''))
            else:
                local_id = str(loc)
            transformed.append({'id': local_id})
            
        elif format_type == "full_id_object":
            # Object with full ONDC location ID AND local_id (required by Himira backend)
            if isinstance(loc, dict):
                local_id = loc.get('local_id', loc.get('id', ''))
                full_id = loc.get('id') if '_' in loc.get('id', '') else create_full_location_id(local_id)
            else:
                local_id = str(loc)
                full_id = create_full_location_id(local_id)
            # Include both id and local_id as per Himira SELECT API requirements
            transformed.append({'id': full_id, 'local_id': local_id})
            
        else:  # "full_object" (default)
            # Full object with both id and local_id
            if isinstance(loc, dict):
                local_id = loc.get('local_id', loc.get('id', ''))
                full_id = loc.get('id') if '_' in loc.get('id', '') else create_full_location_id(local_id)
            else:
                local_id = str(loc)
                full_id = create_full_location_id(local_id)
            
            transformed.append({
                'id': full_id,
                'local_id': local_id
            })
    
    return transformed

def extract_location_ids(provider_info: Dict) -> List[str]:
    """
    Extract location IDs from provider info.
    
    Args:
        provider_info: Provider dictionary with locations
        
    Returns:
        List of location ID strings
    """
    location_ids = []
    
    if not provider_info or 'locations' not in provider_info:
        return location_ids
        
    for loc in provider_info.get('locations', []):
        if isinstance(loc, dict):
            # Try local_id first, then id
            loc_id = loc.get('local_id', loc.get('id', ''))
        else:
            loc_id = str(loc)
            
        if loc_id:
            location_ids.append(loc_id)
            
    return location_ids

def create_provider_for_context(
    provider_info: Dict,
    context: str = "item",
    include_descriptor: bool = False
) -> Optional[Dict]:
    """
    Create provider object based on context (item/cart/order).
    Single source of truth for provider creation.
    
    Args:
        provider_info: Base provider information
        context: Context for provider ("item", "cart", "order")
        include_descriptor: Whether to include descriptor field
        
    Returns:
        Provider dict formatted for the specific context
    """
    if not provider_info:
        return None
        
    provider = {
        'id': provider_info.get('id'),
        'local_id': provider_info.get('local_id')
    }
    
    # Handle locations based on context
    locations = provider_info.get('locations', [])
    
    if context == "item":
        # Item level: Full objects with both id and local_id
        provider['locations'] = transform_provider_locations(locations, "full_object")
        
    elif context == "cart":
        # Cart level: Full ONDC IDs in simple objects
        provider['locations'] = transform_provider_locations(locations, "full_id_object")
        
    elif context == "order":
        # Order level: Simple objects with local_id
        provider['locations'] = transform_provider_locations(locations, "simple_object")
        
    else:
        # Default: string array
        provider['locations'] = transform_provider_locations(locations, "string_array")
    
    # Add descriptor if requested and available
    if include_descriptor and provider_info.get('descriptor'):
        provider['descriptor'] = provider_info.get('descriptor')
    
    # Remove None values
    provider = {k: v for k, v in provider.items() if v is not None}
    
    return provider

def normalize_location_id(location_id: Any) -> str:
    """
    Normalize a location ID to local_id string format.
    
    Args:
        location_id: Location ID in any format
        
    Returns:
        Normalized local ID string
    """
    if isinstance(location_id, dict):
        # Extract from dict
        return location_id.get('local_id', location_id.get('id', ''))
    elif isinstance(location_id, str):
        # If it's a full ONDC ID, extract the last part
        if '_' in location_id:
            parts = location_id.split('_')
            return parts[-1] if parts else location_id
        return location_id
    else:
        return str(location_id) if location_id else ''

def build_location_objects(
    location_set: set,
    provider_info: Optional[Dict] = None
) -> List[Dict]:
    """
    Build location objects from a set of location IDs.
    
    Args:
        location_set: Set of location ID strings
        provider_info: Optional provider info for additional locations
        
    Returns:
        List of location objects with full ONDC IDs
    """
    location_objs = []
    
    # First add locations from provider_info if available
    if provider_info and provider_info.get('locations'):
        location_objs.extend(
            transform_provider_locations(
                provider_info['locations'], 
                "full_object"
            )
        )
    
    # Then add any additional locations from location_set
    for loc_id in location_set:
        # Check if this location is already added
        normalized_id = normalize_location_id(loc_id)
        if not any(obj.get('local_id') == normalized_id for obj in location_objs):
            location_objs.append({
                'id': create_full_location_id(normalized_id),
                'local_id': normalized_id
            })
    
    return location_objs

def validate_provider_locations(provider: Dict) -> bool:
    """
    Validate that provider has proper location structure.
    
    Args:
        provider: Provider dictionary to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not provider or not isinstance(provider, dict):
        return False
        
    if 'locations' not in provider:
        return False
        
    locations = provider.get('locations', [])
    if not locations or not isinstance(locations, list):
        return False
        
    # Check each location has required fields
    for loc in locations:
        if isinstance(loc, dict):
            # Must have either 'id' or 'local_id'
            if not (loc.get('id') or loc.get('local_id')):
                return False
        elif not isinstance(loc, str):
            return False
            
    return True

# Export all utilities
__all__ = [
    'transform_provider_locations',
    'extract_location_ids',
    'create_provider_for_context',
    'normalize_location_id',
    'build_location_objects',
    'validate_provider_locations'
]