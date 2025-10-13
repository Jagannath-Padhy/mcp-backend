"""ONDC Data Models and Generators"""

from typing import Dict, List, Optional
from datetime import datetime
import uuid


class ONDCDataGenerator:
    """Generator for ONDC-compliant data structures"""
    
    @staticmethod
    def generate_transaction_id() -> str:
        """Generate ONDC transaction ID"""
        return f"T{uuid.uuid4().hex[:8].upper()}"
    
    @staticmethod
    def generate_message_id() -> str:
        """Generate ONDC message ID"""
        return f"M{uuid.uuid4().hex[:8].upper()}"
    
    @staticmethod
    def generate_ondc_context(action: str, transaction_id: Optional[str] = None) -> Dict:
        """
        Generate ONDC context structure
        
        Args:
            action: ONDC action (search, select, init, confirm, etc.)
            transaction_id: Optional transaction ID to use
            
        Returns:
            Dict containing ONDC context
        """
        return {
            "domain": "ONDC:RET10",
            "country": "IND",
            "city": "std:0160",
            "action": action,
            "core_version": "1.2.0",
            "bap_id": "buyer-app.ondc.org",
            "bap_uri": "https://buyer-app.ondc.org/protocol/v1",
            "transaction_id": transaction_id or ONDCDataGenerator.generate_transaction_id(),
            "message_id": ONDCDataGenerator.generate_message_id(),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "ttl": "PT30S"
        }
    
    @staticmethod
    def create_item_structure_from_simple(item: Dict) -> Dict:
        """
        Create proper search item structure from simplified item data
        
        Args:
            item: Simple item dictionary with basic fields
            
        Returns:
            Properly structured ONDC item
        """
        return {
            "item_details": {
                "id": item.get("id", f"item_{uuid.uuid4().hex[:8]}"),
                "descriptor": {
                    "name": item.get("name", "Unknown Product"),
                    "short_desc": item.get("description", ""),
                    "long_desc": item.get("long_description", ""),
                    "images": item.get("images", [])
                },
                "price": {
                    "currency": item.get("currency", "INR"),
                    "value": str(item.get("price", 0))
                },
                "category_id": item.get("category", ""),
                "@ondc/org/available_on_cod": item.get("cod_available", False),
                "@ondc/org/returnable": item.get("returnable", False),
                "@ondc/org/return_window": item.get("return_window", ""),
                "@ondc/org/contact_details_consumer_care": item.get("contact", "")
            },
            "provider_details": {
                "id": item.get("provider_id", f"provider_{uuid.uuid4().hex[:8]}"),
                "descriptor": {
                    "name": item.get("seller", item.get("provider_name", "Unknown Provider"))
                }
            },
            "attributes": item  # Store all original attributes
        }
    
    @staticmethod
    def convert_search_item_to_ondc_item(search_item: Dict, quantity: int = 1) -> Dict:
        """
        Convert search result item to ONDC cart item format
        
        Args:
            search_item: Item from search results
            quantity: Quantity to add
            
        Returns:
            ONDC-compliant cart item
        """
        # Handle simplified item structure
        if "item_details" not in search_item and "name" in search_item:
            search_item = ONDCDataGenerator.create_item_structure_from_simple(search_item)
        
        item_details = search_item.get("item_details", {})
        descriptor = item_details.get("descriptor", {})
        price = item_details.get("price", {})
        provider_details = search_item.get("provider_details", {})
        
        # Get product name with fallbacks
        product_name = (descriptor.get("name") or 
                       search_item.get("name") or 
                       search_item.get("attributes", {}).get("name") or
                       "Unknown Product")
        
        return {
            "id": item_details.get("id", f"item_{uuid.uuid4().hex[:8]}"),
            "quantity": {
                "count": quantity
            },
            "provider": {
                "id": provider_details.get("id", f"provider_{uuid.uuid4().hex[:8]}"),
                "descriptor": {
                    "name": provider_details.get("descriptor", {}).get("name", "Unknown Provider")
                }
            },
            "product": {
                "id": item_details.get("id", f"product_{uuid.uuid4().hex[:8]}"),
                "descriptor": {
                    "name": product_name,
                    "short_desc": descriptor.get("short_desc", ""),
                    "long_desc": descriptor.get("long_desc", ""),
                    "images": descriptor.get("images", [])
                },
                "price": {
                    "currency": price.get("currency", "INR"),
                    "value": str(price.get("value", search_item.get("price", "0")))
                }
            },
            "customisations": search_item.get("customisations", []),
            "tags": search_item.get("tags", [])
        }
    
    @staticmethod
    def generate_fulfillment_info(delivery_address: str, phone: str, email: str) -> Dict:
        """Generate ONDC fulfillment information"""
        return {
            "id": f"F{uuid.uuid4().hex[:8]}",
            "type": "Delivery",
            "end": {
                "location": {
                    "gps": "30.7333,76.7794",  # Chandigarh coordinates
                    "address": {
                        "name": "Delivery Address",
                        "building": delivery_address,
                        "locality": "Chandigarh",
                        "city": "Chandigarh",
                        "state": "Punjab",
                        "country": "IND",
                        "area_code": "160001"
                    }
                },
                "contact": {
                    "phone": phone,
                    "email": email
                }
            }
        }
    
    @staticmethod
    def generate_billing_info(address: str, phone: str, email: str) -> Dict:
        """Generate ONDC billing information"""
        return {
            "name": "Customer",
            "address": {
                "name": "Billing Address",
                "building": address,
                "locality": "Chandigarh", 
                "city": "Chandigarh",
                "state": "Punjab",
                "country": "IND",
                "area_code": "160001"
            },
            "phone": phone,
            "email": email,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "updated_at": datetime.utcnow().isoformat() + "Z"
        }
    
    @staticmethod
    def generate_payment_info(payment_method: str) -> Dict:
        """Generate ONDC payment information"""
        payment_map = {
            "COD": "ON-FULFILLMENT",
            "UPI": "PRE-FULFILLMENT", 
            "Card": "PRE-FULFILLMENT"
        }
        
        return {
            "uri": "https://razorpay.com/",
            "tl_method": "http/get",
            "params": {
                "amount": "0.00",
                "currency": "INR",
                "transaction_id": ONDCDataGenerator.generate_transaction_id()
            },
            "type": payment_map.get(payment_method, "ON-FULFILLMENT"),
            "status": "PAID" if payment_method != "COD" else "NOT-PAID",
            "time": {
                "label": "PAYMENT_TIMESTAMP",
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        }
    
    @staticmethod
    def get_placeholder_addresses() -> List[Dict]:
        """Get placeholder delivery addresses for testing"""
        return [
            {
                "id": "1",
                "address": "123 Main Street, Sector 17",
                "city": "Chandigarh",
                "pincode": "160017",
                "full_address": "123 Main Street, Sector 17, Chandigarh - 160017"
            },
            {
                "id": "2", 
                "address": "456 Park Avenue, MG Road",
                "city": "Bangalore",
                "pincode": "560001",
                "full_address": "456 Park Avenue, MG Road, Bangalore - 560001"
            },
            {
                "id": "3",
                "address": "789 Marine Drive, Colaba",
                "city": "Mumbai", 
                "pincode": "400005",
                "full_address": "789 Marine Drive, Colaba, Mumbai - 400005"
            }
        ]
    
    @staticmethod
    def get_placeholder_contacts() -> List[Dict]:
        """Get placeholder contact details for testing"""
        return [
            {
                "id": "1",
                "phone": "9876543210",
                "email": "test@example.com",
                "display": "9876543210, test@example.com"
            },
            {
                "id": "2",
                "phone": "8765432109", 
                "email": "user@ondc.com",
                "display": "8765432109, user@ondc.com"
            },
            {
                "id": "3",
                "phone": "7654321098",
                "email": "buyer@shop.com", 
                "display": "7654321098, buyer@shop.com"
            }
        ]
    
    @staticmethod
    def get_payment_options() -> List[Dict]:
        """Get payment method options"""
        return [
            {
                "id": "1",
                "method": "COD",
                "display": "Cash on Delivery (COD)",
                "description": "Pay when you receive your order"
            },
            {
                "id": "2",
                "method": "UPI",
                "display": "UPI Payment", 
                "description": "Pay using Google Pay, PhonePe, Paytm, etc."
            },
            {
                "id": "3",
                "method": "Card",
                "display": "Credit/Debit Card",
                "description": "Pay using Visa, Mastercard, Rupay cards"
            }
        ]
    
