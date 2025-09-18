"""
ONDC Constants - Centralized Constants Module
=============================================

Single source of truth for all ONDC-related constants used across the codebase.
This module consolidates hardcoded values to follow the DRY principle.

Date: 2025-09-15
Purpose: Centralize all ONDC constants to reduce code duplication
"""

# ===========================
# PROVIDER & LOCATION IDS
# ===========================

# Himira Provider Core IDs
HIMIRA_BPP_ID = "hp-seller-preprod.himira.co.in"
HIMIRA_BPP_URI = "https://hp-seller-preprod.himira.co.in/api/v2"
HIMIRA_DOMAIN = "ONDC:RET10"

# Provider IDs
HIMIRA_PROVIDER_LOCAL_ID = "d871c2ae-bf3f-4d3c-963f-f85f94848e8c"
HIMIRA_PROVIDER_ID = f"{HIMIRA_BPP_ID}_{HIMIRA_DOMAIN}_{HIMIRA_PROVIDER_LOCAL_ID}"

# Location IDs  
HIMIRA_LOCATION_LOCAL_ID = "d871c2ae-bf3f-4d3c-963f-f85f94848e8c"
HIMIRA_LOCATION_ID = f"{HIMIRA_PROVIDER_ID}_{HIMIRA_LOCATION_LOCAL_ID}"

# ===========================
# BUYER APP (BAP) CONSTANTS
# ===========================

BAP_ID = "hp-buyer-preprod.himira.co.in"
BAP_URI = "https://hp-buyer-backend-preprod.himira.co.in/protocol/v1"

# ===========================
# CITY CODES & GPS COORDINATES
# ===========================

# Common city codes (ONDC standard format: std:XXX)
CITY_CODES = {
    "bangalore": "std:080",
    "bengaluru": "std:080",
    "mohali": "std:0172",
    "kharar": "std:0172", 
    "chandigarh": "std:0172",
    "delhi": "std:011",
    "mumbai": "std:022",
    "pune": "std:020",
    "hyderabad": "std:040",
    "chennai": "std:044",
    "kolkata": "std:033"
}

# Default city code (Bangalore)
DEFAULT_CITY_CODE = "std:080"

# City GPS coordinates (latitude,longitude format)
CITY_GPS_COORDINATES = {
    "mohali": "30.7046,76.7179",
    "kharar": "30.7419,76.6457",
    "chandigarh": "30.7333,76.7794",
    "delhi": "28.6139,77.2090",
    "bangalore": "12.9716,77.5946",
    "bengaluru": "12.9716,77.5946",
    "mumbai": "19.0760,72.8777",
    "pune": "18.5204,73.8567",
    "hyderabad": "17.3850,78.4867",
    "chennai": "13.0827,80.2707",
    "kolkata": "22.5726,88.3639"
}

# Default GPS (Mohali)
DEFAULT_GPS = "30.7046,76.7179"

# ===========================
# ONDC PROTOCOL CONSTANTS
# ===========================

# Protocol version
CORE_VERSION = "1.2.0"

# Default TTL
DEFAULT_TTL = "PT30S"
CATALOG_TTL = "PT24H"

# Actions
ONDC_ACTIONS = {
    "SEARCH": "search",
    "SELECT": "select",
    "INIT": "init",
    "CONFIRM": "confirm",
    "STATUS": "status",
    "TRACK": "track",
    "CANCEL": "cancel",
    "RATING": "rating",
    "SUPPORT": "support"
}

# ===========================
# PROVIDER CONSTANTS
# ===========================

# Provider location defaults
PROVIDER_GPS = "0,0"  # Default provider GPS
PROVIDER_AREA_CODE = "140301"  # Mohali area code
PROVIDER_CITY = "Mohali"
PROVIDER_STATE = "Punjab"

# FSSAI License (required for food items)
FSSAI_LICENSE_NO = "12345678901234"

# ===========================
# FULFILLMENT CONSTANTS
# ===========================

# Fulfillment types
FULFILLMENT_TYPES = {
    "DELIVERY": "Delivery",
    "PICKUP": "Self-Pickup",
    "DELIVERY_AND_PICKUP": "Delivery and Self-Pickup"
}

# Default fulfillment ID
DEFAULT_FULFILLMENT_ID = "1"

# ===========================
# PAYMENT CONSTANTS
# ===========================

# Payment types
PAYMENT_TYPES = {
    "ON_ORDER": "ON-ORDER",
    "POST_FULFILLMENT": "POST-FULFILLMENT",
    "ON_FULFILLMENT": "ON-FULFILLMENT"
}

# Default payment type
DEFAULT_PAYMENT_TYPE = "ON-ORDER"

# ===========================
# PRODUCT CATEGORIES
# ===========================

PRODUCT_CATEGORIES = [
    "Tinned and Processed Food",
    "Salt, Sugar and Jaggery",
    "Oil & Ghee", 
    "Pickles and Chutney",
    "Frozen Vegetables",
    "Masala & Seasoning",
    "Fruits and Vegetables"
]

# ===========================
# ERROR MESSAGES
# ===========================

ERROR_MESSAGES = {
    "EMPTY_CART": "Cart is empty",
    "EMPTY_ORDER": "Empty order received",
    "INVALID_PROVIDER": "Invalid provider information",
    "DELIVERY_NOT_AVAILABLE": "Delivery not available in this location",
    "PAYMENT_FAILED": "Payment processing failed",
    "VALIDATION_ERROR": "Request validation failed"
}

# ===========================
# HELPER FUNCTIONS
# ===========================

def get_city_code(city: str) -> str:
    """Get ONDC city code for a city name"""
    return CITY_CODES.get(city.lower(), DEFAULT_CITY_CODE)

def get_city_gps(city: str) -> str:
    """Get GPS coordinates for a city"""
    return CITY_GPS_COORDINATES.get(city.lower(), DEFAULT_GPS)

def create_full_provider_id(provider_local_id: str = None) -> str:
    """Create full ONDC provider ID"""
    local_id = provider_local_id or HIMIRA_PROVIDER_LOCAL_ID
    return f"{HIMIRA_BPP_ID}_{HIMIRA_DOMAIN}_{local_id}"

def create_full_location_id(location_local_id: str = None, provider_local_id: str = None) -> str:
    """Create full ONDC location ID"""
    provider_id = create_full_provider_id(provider_local_id)
    location_id = location_local_id or HIMIRA_LOCATION_LOCAL_ID
    return f"{provider_id}_{location_id}"

def create_full_item_id(item_local_id: str, provider_local_id: str = None) -> str:
    """Create full ONDC item ID"""
    provider_id = create_full_provider_id(provider_local_id)
    return f"{provider_id}_{item_local_id}"

# Export all constants
__all__ = [
    # Provider & Location IDs
    'HIMIRA_BPP_ID', 'HIMIRA_BPP_URI', 'HIMIRA_DOMAIN',
    'HIMIRA_PROVIDER_LOCAL_ID', 'HIMIRA_PROVIDER_ID',
    'HIMIRA_LOCATION_LOCAL_ID', 'HIMIRA_LOCATION_ID',
    
    # BAP Constants
    'BAP_ID', 'BAP_URI',
    
    # City & GPS
    'CITY_CODES', 'DEFAULT_CITY_CODE',
    'CITY_GPS_COORDINATES', 'DEFAULT_GPS',
    
    # Protocol Constants
    'CORE_VERSION', 'DEFAULT_TTL', 'CATALOG_TTL', 'ONDC_ACTIONS',
    
    # Provider Constants
    'PROVIDER_GPS', 'PROVIDER_AREA_CODE', 'PROVIDER_CITY', 
    'PROVIDER_STATE', 'FSSAI_LICENSE_NO',
    
    # Fulfillment & Payment
    'FULFILLMENT_TYPES', 'DEFAULT_FULFILLMENT_ID',
    'PAYMENT_TYPES', 'DEFAULT_PAYMENT_TYPE',
    
    # Categories & Errors
    'PRODUCT_CATEGORIES', 'ERROR_MESSAGES',
    
    # Helper Functions
    'get_city_code', 'get_city_gps',
    'create_full_provider_id', 'create_full_location_id', 'create_full_item_id'
]