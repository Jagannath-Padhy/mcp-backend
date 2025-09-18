"""
Himira Provider Constants - ONDC Compatible
===========================================

Provider data structures extracted from Himira Order Postman collection.
These constants match the exact format expected by the Himira BIAP backend.

Source: Himira Order.postman_collection (1).json
Date: 2025-09-14
Purpose: Ensure cart items have proper ONDC provider structure for SELECT/INIT/CONFIRM flow
"""

from typing import Dict, Any, List

# Default Himira Store Provider Data (from Postman collection)
# This matches the exact structure used in successful ONDC transactions
HIMIRA_DEFAULT_PROVIDER = {
    "id": "hp-seller-preprod.himira.co.in_ONDC:RET10_d871c2ae-bf3f-4d3c-963f-f85f94848e8c",
    "local_id": "d871c2ae-bf3f-4d3c-963f-f85f94848e8c",
    "descriptor": {
        "name": "Himira Store",
        "symbol": "https://storage.googleapis.com/hp-seller-preprod/logo?GoogleAccessId=hp-ondc-sa%40hp-gov.iam.gserviceaccount.com&Expires=2104115225&Signature=6Q7Ulph%2BOoO4%2FNOb0VGC3tJF%2FfMe9jCfIh%2FvBUiiIFtzG3BzI%2BY%2BBdCgQaeqaxWg%2B3%2FCCm6%2Bt15nFHEVpkNl6ZkL9Uu%2BgYJJVzCRAXCCtrG6loG6xf1kZZuy7Ml8JL1JSId8EuQED5dEM1BKAtjTWIxMj7p4tIg5ngARQs%2FkCFQZrwFGOKXajUURDed1Y2MUz1L4gCeKZetlniWM53kIX8J5BQU2ZCTPJ7yLW0tYFLnZto9wNuwVdMTjM%2BseEESkrEiAB1zsDfpeC6IEWTpiPMF2%2B3tJFDQd%2BxcBq6ZMqUm7l8x5gZgaiX7pIi7C4m7lGp3r%2FBuPPwveDmBF%2BZhGXQ%3D%3D",
        "short_desc": "Himira Store",
        "long_desc": "Himira Store",
        "images": ["https://storage.googleapis.com/hp-seller-preprod/logo?GoogleAccessId=hp-ondc-sa%40hp-gov.iam.gserviceaccount.com&Expires=2104115225&Signature=6Q7Ulph%2BOoO4%2FNOb0VGC3tJF%2FfMe9jCfIh%2FvBUiiIFtzG3BzI%2BY%2BBdCgQaeqaxWg%2B3%2FCCm6%2Bt15nFHEVpkNl6ZkL9Uu%2BgYJJVzCRAXCCtrG6loG6xf1kZZuy7Ml8JL1JSId8EuQED5dEM1BKAtjTWIxMj7p4tIg5ngARQs%2FkCFQZrwFGOKXajUURDed1Y2MUz1L4gCeKZetlniWM53kIX8J5BQU2ZCTPJ7yLW0tYFLnZto9wNuwVdMTjM%2BseEESkrEiAB1zsDfpeC6IEWTpiPMF2%2B3tJFDQd%2BxcBq6ZMqUm7l8x5gZgaiX7pIi7C4m7lGp3r%2FBuPPwveDmBF%2BZhGXQ%3D%3D"]
    },
    "locations": [{
        "id": "hp-seller-preprod.himira.co.in_ONDC:RET10_d871c2ae-bf3f-4d3c-963f-f85f94848e8c_d871c2ae-bf3f-4d3c-963f-f85f94848e8c",
        "local_id": "d871c2ae-bf3f-4d3c-963f-f85f94848e8c",
        "gps": "0,0",  # Provider location coordinates
        "address": {
            "city": "Mohali", 
            "state": "Punjab",
            "area_code": "140301",
            "street": "Eco Floors 1",
            "locality": "Sector 125"
        },
        "time": {
            "label": "enable",
            "timestamp": "2025-08-27T09:00:46.743Z",
            "days": "1,2,3,4,5,6,7",
            "schedule": {"holidays": ["2027-10-23"]},
            "range": {"start": "0000", "end": "2359"}
        },
        "circle": {
            "gps": "0,0",
            "radius": {"unit": "km", "value": "3"}
        }
    }],
    "time": {
        "label": "enable", 
        "timestamp": "2025-08-27T09:00:46.743Z"
    },
    "@ondc/org/fssai_license_no": "12345678901234",
    "ttl": "PT24H",
    "fulfillments": [{
        "id": "1",
        "type": "Delivery",
        "contact": {
            "email": "naman.pawar@thewitslab.com",
            "phone": "7505390422"
        }
    }],
    "tags": [
        {
            "code": "serviceability",
            "list": [
                {"code": "location", "value": "d871c2ae-bf3f-4d3c-963f-f85f94848e8c"},
                {"code": "category", "value": "Tinned and Processed Food"},
                {"code": "type", "value": "12"},
                {"code": "unit", "value": "country"},
                {"code": "val", "value": "IND"}
            ]
        },
        {
            "code": "serviceability", 
            "list": [
                {"code": "location", "value": "d871c2ae-bf3f-4d3c-963f-f85f94848e8c"},
                {"code": "category", "value": "Salt, Sugar and Jaggery"},
                {"code": "type", "value": "12"},
                {"code": "unit", "value": "country"},
                {"code": "val", "value": "IND"}
            ]
        },
        {
            "code": "serviceability",
            "list": [
                {"code": "location", "value": "d871c2ae-bf3f-4d3c-963f-f85f94848e8c"},
                {"code": "category", "value": "Oil & Ghee"},
                {"code": "type", "value": "12"},
                {"code": "unit", "value": "country"},
                {"code": "val", "value": "IND"}
            ]
        },
        {
            "code": "serviceability",
            "list": [
                {"code": "location", "value": "d871c2ae-bf3f-4d3c-963f-f85f94848e8c"},
                {"code": "category", "value": "Pickles and Chutney"},
                {"code": "type", "value": "12"},
                {"code": "unit", "value": "country"},
                {"code": "val", "value": "IND"}
            ]
        },
        {
            "code": "serviceability", 
            "list": [
                {"code": "location", "value": "d871c2ae-bf3f-4d3c-963f-f85f94848e8c"},
                {"code": "category", "value": "Frozen Vegetables"},
                {"code": "type", "value": "12"},
                {"code": "unit", "value": "country"},
                {"code": "val", "value": "IND"}
            ]
        },
        {
            "code": "serviceability",
            "list": [
                {"code": "location", "value": "d871c2ae-bf3f-4d3c-963f-f85f94848e8c"},
                {"code": "category", "value": "Masala & Seasoning"},
                {"code": "type", "value": "12"},
                {"code": "unit", "value": "country"},
                {"code": "val", "value": "IND"}
            ]
        },
        {
            "code": "serviceability",
            "list": [
                {"code": "location", "value": "d871c2ae-bf3f-4d3c-963f-f85f94848e8c"},
                {"code": "category", "value": "Fruits and Vegetables"},
                {"code": "type", "value": "12"},
                {"code": "unit", "value": "country"},
                {"code": "val", "value": "IND"}
            ]
        },
        {
            "code": "timing",
            "list": [
                {"code": "type", "value": "All"},
                {"code": "location", "value": "d871c2ae-bf3f-4d3c-963f-f85f94848e8c"},
                {"code": "day_from", "value": "1"},
                {"code": "day_to", "value": "7"},
                {"code": "time_from", "value": "0000"},
                {"code": "time_to", "value": "2359"}
            ]
        }
    ],
    "slug": "Himira-Store-4c6840"
}

# Core Provider IDs (extracted from Postman collection)
HIMIRA_PROVIDER_ID = "hp-seller-preprod.himira.co.in_ONDC:RET10_d871c2ae-bf3f-4d3c-963f-f85f94848e8c"
HIMIRA_PROVIDER_LOCAL_ID = "d871c2ae-bf3f-4d3c-963f-f85f94848e8c"
HIMIRA_LOCATION_ID = "hp-seller-preprod.himira.co.in_ONDC:RET10_d871c2ae-bf3f-4d3c-963f-f85f94848e8c_d871c2ae-bf3f-4d3c-963f-f85f94848e8c"
HIMIRA_LOCATION_LOCAL_ID = "d871c2ae-bf3f-4d3c-963f-f85f94848e8c"

# BPP Information (Business Platform Provider)
HIMIRA_BPP_ID = "hp-seller-preprod.himira.co.in"
HIMIRA_BPP_URI = "https://hp-seller-preprod.himira.co.in/api/v2"
HIMIRA_DOMAIN = "ONDC:RET10"

# Common ONDC Item ID Pattern Generator
def generate_himira_item_id(local_item_id: str) -> str:
    """
    Generate ONDC-compliant item ID for Himira products
    
    Args:
        local_item_id: Local product ID (UUID format)
        
    Returns:
        Full ONDC item ID matching Postman collection pattern
    """
    return f"{HIMIRA_BPP_ID}_{HIMIRA_DOMAIN}_{HIMIRA_PROVIDER_LOCAL_ID}_{local_item_id}"

def create_enriched_provider_data(category: str = "Tinned and Processed Food") -> Dict[str, Any]:
    """
    Create provider data with category-specific serviceability
    
    Args:
        category: Product category for serviceability configuration
        
    Returns:
        Provider dictionary with proper ONDC structure
    """
    provider_data = HIMIRA_DEFAULT_PROVIDER.copy()
    
    # Update serviceability for specific category
    if category and category not in ["Tinned and Processed Food", "Salt, Sugar and Jaggery", 
                                   "Oil & Ghee", "Pickles and Chutney", "Frozen Vegetables", 
                                   "Masala & Seasoning", "Fruits and Vegetables"]:
        # Add serviceability for new category
        provider_data["tags"].append({
            "code": "serviceability",
            "list": [
                {"code": "location", "value": HIMIRA_LOCATION_LOCAL_ID},
                {"code": "category", "value": category},
                {"code": "type", "value": "12"},
                {"code": "unit", "value": "country"},
                {"code": "val", "value": "IND"}
            ]
        })
    
    return provider_data

def create_minimal_provider_for_cart() -> Dict[str, Any]:
    """
    Create minimal provider structure for cart items
    Contains only essential fields needed for ONDC SELECT/INIT/CONFIRM
    
    Returns:
        Minimal provider structure matching backend expected format
    """
    return {
        "id": HIMIRA_PROVIDER_ID,
        "local_id": HIMIRA_PROVIDER_LOCAL_ID,
        # CRITICAL: Backend expects locations as array of strings, not objects!
        # Backend will transform these to {id: location} objects internally
        "locations": [HIMIRA_LOCATION_LOCAL_ID]  # Just the string ID
    }

def enrich_cart_item_with_provider(item_data: Dict[str, Any], category: str = "Tinned and Processed Food") -> Dict[str, Any]:
    """
    Enrich a cart item with proper Himira provider data
    
    Args:
        item_data: Basic item data (name, price, etc.)
        category: Product category for serviceability
        
    Returns:
        Enriched item data with proper ONDC provider structure
    """
    # Extract local_id from full ONDC ID format if needed
    import uuid
    
    if item_data.get("local_id"):
        local_id = item_data["local_id"]
    elif item_data.get("id"):
        # ID format: hp-seller-preprod.himira.co.in_ONDC:RET10_provider-uuid_item-uuid
        # Extract the last UUID part as local_id
        id_parts = item_data["id"].split("_")
        if len(id_parts) >= 4:
            local_id = id_parts[-1]  # Last UUID is the item's local ID
        else:
            local_id = item_data["id"]  # Use as-is if not in expected format
    else:
        # Only generate UUID as last resort
        local_id = str(uuid.uuid4())
    
    enriched_item = {
        "id": generate_himira_item_id(local_id),
        "local_id": local_id,
        "name": item_data.get("name", "Unknown Product"),
        "price": item_data.get("price", 0),
        "category": category,
        "description": item_data.get("description", ""),
        "image_url": item_data.get("image_url", ""),
        
        # ONDC Required Fields
        "bpp_id": HIMIRA_BPP_ID,
        "bpp_uri": HIMIRA_BPP_URI,
        "domain": HIMIRA_DOMAIN,
        "contextCity": "std:0172",  # Kharar/Mohali area code
        
        # Provider Structure (from Postman collection)
        "provider": create_minimal_provider_for_cart(),
        
        # Location and Fulfillment
        "location_id": HIMIRA_LOCATION_LOCAL_ID,
        "fulfillment_id": "1",
        
        # ONDC Compliance Tags
        "tags": [
            {
                "code": "origin",
                "list": [{"code": "country", "value": "IND"}]
            },
            {
                "code": "type", 
                "list": [{"code": "type", "value": "item"}]
            },
            {
                "code": "veg_nonveg",
                "list": [{"code": "veg", "value": "yes"}]
            }
        ],
        
        # Additional ONDC fields
        "customisations": None,
        "hasCustomisations": False,
        "customisationState": {}
    }
    
    return enriched_item

# Export constants for easy access
__all__ = [
    'HIMIRA_DEFAULT_PROVIDER',
    'HIMIRA_PROVIDER_ID', 
    'HIMIRA_PROVIDER_LOCAL_ID',
    'HIMIRA_LOCATION_ID',
    'HIMIRA_LOCATION_LOCAL_ID',
    'HIMIRA_BPP_ID',
    'HIMIRA_BPP_URI', 
    'HIMIRA_DOMAIN',
    'generate_himira_item_id',
    'create_enriched_provider_data',
    'create_minimal_provider_for_cart',
    'enrich_cart_item_with_provider'
]