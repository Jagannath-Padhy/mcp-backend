"""City Code Mapping Service - BIAP Compatible"""

from typing import Optional, Dict


# Area code mapping - subset of BIAP AreaCodeMap for common cities
AREA_CODE_MAP = [
    # Delhi
    {"pincode": 110001, "city": "CENTRAL", "state": "DELHI", "std_code": "011"},
    {"pincode": 110002, "city": "NEW DELHI", "state": "DELHI", "std_code": "011"},
    {"pincode": 110003, "city": "NEW DELHI", "state": "DELHI", "std_code": "011"},
    
    # Mumbai
    {"pincode": 400001, "city": "MUMBAI", "state": "MAHARASHTRA", "std_code": "022"},
    {"pincode": 400002, "city": "MUMBAI", "state": "MAHARASHTRA", "std_code": "022"},
    {"pincode": 400003, "city": "MUMBAI", "state": "MAHARASHTRA", "std_code": "022"},
    
    # Bangalore
    {"pincode": 560001, "city": "BANGALORE", "state": "KARNATAKA", "std_code": "080"},
    {"pincode": 560002, "city": "BANGALORE", "state": "KARNATAKA", "std_code": "080"},
    {"pincode": 560034, "city": "BANGALORE", "state": "KARNATAKA", "std_code": "080"},
    
    # Chennai  
    {"pincode": 600001, "city": "CHENNAI", "state": "TAMIL NADU", "std_code": "044"},
    {"pincode": 600002, "city": "CHENNAI", "state": "TAMIL NADU", "std_code": "044"},
    
    # Hyderabad
    {"pincode": 500001, "city": "HYDERABAD", "state": "TELANGANA", "std_code": "040"},
    {"pincode": 500002, "city": "HYDERABAD", "state": "TELANGANA", "std_code": "040"},
    
    # Pune
    {"pincode": 411001, "city": "PUNE", "state": "MAHARASHTRA", "std_code": "020"},
    {"pincode": 411002, "city": "PUNE", "state": "MAHARASHTRA", "std_code": "020"},
    
    # Kharar (from user's example)
    {"pincode": 140301, "city": "KHARAR", "state": "PUNJAB", "std_code": "0172"},
]


def get_city_code_by_pincode(pincode: Optional[str]) -> str:
    """
    Get city code in ONDC format from pincode
    Matches BIAP getCityCode function
    
    Args:
        pincode: Pincode string or number
        
    Returns:
        City code in format "std:XXX" (defaults to "std:080" for Bangalore)
    """
    if not pincode:
        return "std:080"  # Default to Bangalore like BIAP
    
    try:
        # Convert to int for lookup
        pincode_int = int(str(pincode))
        
        # Find matching area code
        area_code = next(
            (item for item in AREA_CODE_MAP if item["pincode"] == pincode_int), 
            None
        )
        
        if area_code:
            return f"std:{area_code['std_code']}"
        else:
            return "std:080"  # Default to Bangalore
            
    except (ValueError, TypeError):
        return "std:080"  # Default to Bangalore


def get_city_info_by_pincode(pincode: Optional[str]) -> Optional[Dict]:
    """
    Get complete city information by pincode
    
    Args:
        pincode: Pincode string or number
        
    Returns:
        Dictionary with city, state, std_code or None if not found
    """
    if not pincode:
        return None
        
    try:
        pincode_int = int(str(pincode))
        return next(
            (item for item in AREA_CODE_MAP if item["pincode"] == pincode_int),
            None
        )
    except (ValueError, TypeError):
        return None


def create_delivery_info_from_billing_info(billing_info: Dict) -> Dict:
    """
    Create delivery info from billing info
    Matches BIAP createDeliveryInfoFromBillingInfo function
    
    Args:
        billing_info: Dictionary with address, email, name, phone, lat, lng
        
    Returns:
        BIAP-compatible delivery info structure
    """
    address = billing_info.get("address", {})
    email = billing_info.get("email", "")
    name = billing_info.get("name", "")
    phone = billing_info.get("phone", "")
    lat = address.get("lat", billing_info.get("lat", "12.9716"))  # Default to Bangalore
    lng = address.get("lng", billing_info.get("lng", "77.5946"))  # Default to Bangalore
    
    delivery_info = {
        "email": email,
        "location": {
            "address": {**address},
            "gps": f"{lat},{lng}"
        },
        "name": name,
        "phone": phone,
        "type": "Delivery"
    }
    
    return delivery_info


# Common city mappings for quick reference
COMMON_CITY_CODES = {
    "delhi": "std:011",
    "mumbai": "std:022", 
    "bangalore": "std:080",
    "chennai": "std:044",
    "hyderabad": "std:040",
    "pune": "std:020",
    "kharar": "std:0172"
}


def get_city_code_by_name(city_name: str) -> str:
    """
    Get city code by city name (fallback method)
    
    Args:
        city_name: City name
        
    Returns:
        City code in format "std:XXX"
    """
    if not city_name:
        return "std:080"
        
    city_lower = city_name.lower().strip()
    return COMMON_CITY_CODES.get(city_lower, "std:080")


# Pincode to GPS coordinates mapping for major areas
PINCODE_COORDINATES = {
    # Delhi NCR
    "110001": {"lat": 28.6333, "lng": 77.2167, "city": "Delhi"},
    "110002": {"lat": 28.6369, "lng": 77.2183, "city": "Delhi"},
    "110003": {"lat": 28.6517, "lng": 77.2219, "city": "Delhi"},
    
    # Mumbai
    "400001": {"lat": 18.9388, "lng": 72.8354, "city": "Mumbai"},
    "400002": {"lat": 18.9484, "lng": 72.8327, "city": "Mumbai"},
    "400003": {"lat": 18.9547, "lng": 72.8302, "city": "Mumbai"},
    
    # Bangalore
    "560001": {"lat": 12.9719, "lng": 77.5937, "city": "Bangalore"},
    "560002": {"lat": 12.9634, "lng": 77.5855, "city": "Bangalore"},
    "560034": {"lat": 12.9565, "lng": 77.7004, "city": "Bangalore"},
    
    # Chennai
    "600001": {"lat": 13.0827, "lng": 80.2707, "city": "Chennai"},
    "600002": {"lat": 13.0878, "lng": 80.2785, "city": "Chennai"},
    
    # Hyderabad
    "500001": {"lat": 17.3850, "lng": 78.4867, "city": "Hyderabad"},
    "500002": {"lat": 17.3616, "lng": 78.4747, "city": "Hyderabad"},
    
    # Pune
    "411001": {"lat": 18.5204, "lng": 73.8567, "city": "Pune"},
    "411002": {"lat": 18.5074, "lng": 73.8907, "city": "Pune"},
    
    # Kharar (Punjab) - User's example
    "140301": {"lat": 30.7455, "lng": 76.6357, "city": "Kharar"},
    
    # Chandigarh
    "160001": {"lat": 30.7333, "lng": 76.7794, "city": "Chandigarh"},
    "160002": {"lat": 30.7370, "lng": 76.7880, "city": "Chandigarh"},
}


def get_coordinates_by_pincode(pincode: Optional[str]) -> Dict[str, float]:
    """
    Get GPS coordinates from pincode
    
    Args:
        pincode: 6-digit Indian pincode
        
    Returns:
        Dictionary with 'latitude' and 'longitude' keys
    """
    if not pincode:
        # Default to Bangalore coordinates
        return {"latitude": 12.9719, "longitude": 77.5937}
    
    # Clean pincode
    pincode_clean = str(pincode).strip()
    
    # Look up exact match first
    if pincode_clean in PINCODE_COORDINATES:
        coords = PINCODE_COORDINATES[pincode_clean]
        return {"latitude": coords["lat"], "longitude": coords["lng"]}
    
    # Try to match by first 3 digits (city/district level)
    prefix = pincode_clean[:3] if len(pincode_clean) >= 3 else None
    if prefix:
        # Map common prefixes to major city coordinates
        prefix_map = {
            "110": {"latitude": 28.6139, "longitude": 77.2090},  # Delhi
            "400": {"latitude": 19.0760, "longitude": 72.8777},  # Mumbai
            "560": {"latitude": 12.9716, "longitude": 77.5946},  # Bangalore
            "600": {"latitude": 13.0827, "longitude": 80.2707},  # Chennai
            "500": {"latitude": 17.3850, "longitude": 78.4867},  # Hyderabad
            "411": {"latitude": 18.5204, "longitude": 73.8567},  # Pune
            "140": {"latitude": 30.7455, "longitude": 76.6357},  # Kharar/Mohali
            "160": {"latitude": 30.7333, "longitude": 76.7794},  # Chandigarh
        }
        
        if prefix in prefix_map:
            return prefix_map[prefix]
    
    # Default to Bangalore if no match found
    return {"latitude": 12.9719, "longitude": 77.5937}