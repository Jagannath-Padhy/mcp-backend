"""
MOCK PAYMENT SERVICE - FOR TESTING ONLY
========================================

This service provides mock Razorpay payment responses for testing purposes.
All mock values are clearly labeled and sourced from the Himira Order Postman collection.

IMPORTANT: This is a mock service and should not be used in production.
Mock values are based on tested data from the Himira Order.postman_collection.

Author: AI Assistant
Created: 2025-01-14
Purpose: Client-side payment mocking to work with pre-prod backend without modifications
"""

import os
import logging
import uuid
from datetime import datetime
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class PaymentMockService:
    """
    MOCK SERVICE - Provides simulated Razorpay payment responses
    
    This service simulates the Razorpay payment creation and verification
    process using test values from the Himira Order Postman collection.
    """
    
    def __init__(self):
        """Initialize mock payment service with configuration from environment"""
        # MOCK CONFIGURATION - Environment driven
        self.mock_mode = os.getenv('PAYMENT_MOCK_MODE', 'true').lower() == 'true'
        self.debug_logs = os.getenv('MCP_DEBUG_PAYMENT_LOGS', 'true').lower() == 'true'
        
        # MOCK VALUES - Source: Himira Order Postman Collection
        self.mock_payment_id = os.getenv('MOCK_RAZORPAY_PAYMENT_ID', 'pay_RFWPuAV50T2Qnj')
        self.mock_payment_status = os.getenv('MOCK_PAYMENT_STATUS', 'PAID')
        
        # MOCK ONDC Settlement Values - ONDC Protocol Compliant
        self.mock_settlement_basis = os.getenv('MOCK_SETTLEMENT_BASIS', 'delivery')
        self.mock_settlement_window = os.getenv('MOCK_SETTLEMENT_WINDOW', 'P1D')  # 1 day
        self.mock_withholding_amount = os.getenv('MOCK_WITHHOLDING_AMOUNT', '0.00')
        
        if self.debug_logs:
            logger.info(f"[MOCK PAYMENT SERVICE] Initialized with payment ID: {self.mock_payment_id}")
            logger.info(f"[MOCK PAYMENT SERVICE] Mock mode: {self.mock_mode}")
    
    def create_mock_razorpay_order(self, amount: float, currency: str = "INR", transaction_id: str = None) -> Dict[str, Any]:
        """
        MOCK IMPLEMENTATION - Simulates Razorpay order creation
        
        Args:
            amount: Order amount in INR
            currency: Currency code (default: INR)
            transaction_id: Optional transaction ID
            
        Returns:
            Mock Razorpay order response matching Himira backend format
        """
        if not self.mock_mode:
            raise ValueError("Mock service called when PAYMENT_MOCK_MODE is disabled")
        
        # Generate mock order ID following Razorpay pattern
        mock_order_id = f"order_{uuid.uuid4().hex[:14]}"
        
        mock_order = {
            # MOCK VALUES - Razorpay format
            "id": mock_order_id,  # MOCK: Generated order ID
            "entity": "order",
            "amount": int(amount * 100),  # Razorpay uses paise (amount * 100)
            "amount_paid": 0,
            "amount_due": int(amount * 100),
            "currency": currency,
            "receipt": transaction_id or f"receipt_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "status": "created",
            "created_at": int(datetime.now().timestamp()),
            
            # MOCK METADATA - Clear indicators
            "_mock_source": "himira_postman_collection",
            "_mock_service": "payment_mock_service",
            "_mock_timestamp": datetime.now().isoformat()
        }
        
        if self.debug_logs:
            logger.info(f"[MOCK RAZORPAY ORDER] Created mock order: {mock_order_id}")
            logger.info(f"[MOCK RAZORPAY ORDER] Amount: {amount} {currency}")
        
        return mock_order
    
    def create_mock_payment(self, amount: float, currency: str = "INR", order_id: str = None) -> Dict[str, Any]:
        """
        MOCK IMPLEMENTATION - Simulates successful Razorpay payment
        
        Args:
            amount: Payment amount in INR
            currency: Currency code (default: INR)
            order_id: Associated order ID
            
        Returns:
            Mock payment response with test values from Postman collection
        """
        if not self.mock_mode:
            raise ValueError("Mock service called when PAYMENT_MOCK_MODE is disabled")
        
        mock_payment = {
            # MOCK VALUES - From Himira Order Postman Collection
            "razorpayPaymentId": self.mock_payment_id,  # MOCK: pay_RFWPuAV50T2Qnj
            "status": self.mock_payment_status,  # MOCK: PAID
            "amount": amount,
            "currency": currency,
            "order_id": order_id,
            
            # Additional Razorpay fields for completeness
            "method": "card",  # MOCK: Simulated card payment
            "captured": True,
            "created_at": int(datetime.now().timestamp()),
            
            # MOCK METADATA - Clear indicators
            "_mock_source": "himira_postman_collection",
            "_mock_payment_id_source": "postman_pay_RFWPuAV50T2Qnj",
            "_mock_service": "payment_mock_service",
            "_mock_timestamp": datetime.now().isoformat()
        }
        
        if self.debug_logs:
            logger.info(f"[MOCK PAYMENT] Created mock payment: {self.mock_payment_id}")
            logger.info(f"[MOCK PAYMENT] Status: {self.mock_payment_status}")
            logger.info(f"[MOCK PAYMENT] Amount: {amount} {currency}")
        
        return mock_payment
    
    def get_mock_payment_id(self) -> str:
        """
        Get the mock Razorpay payment ID from Postman collection
        
        Returns:
            Mock payment ID: pay_RFWPuAV50T2Qnj (from Himira Order Postman collection)
        """
        if self.debug_logs:
            logger.info(f"[MOCK PAYMENT ID] Returning: {self.mock_payment_id}")
        
        return self.mock_payment_id
    
    def create_biap_payment_object(self, total_amount: float) -> Dict[str, Any]:
        """
        Create BIAP-compliant payment object with mock values
        
        Args:
            total_amount: Total order amount
            
        Returns:
            Payment object matching BIAP/ONDC specifications with mock values
        """
        if not self.mock_mode:
            raise ValueError("Mock service called when PAYMENT_MOCK_MODE is disabled")
        
        payment_obj = {
            # BIAP Standard Fields
            'type': 'ON-ORDER',  # Always ON-ORDER for Razorpay payments
            'collected_by': 'BAP',  # Always BAP for ON-ORDER payments
            'razorpayPaymentId': self.mock_payment_id,  # MOCK: From Postman collection
            'paid_amount': total_amount,
            'status': self.mock_payment_status,  # MOCK: PAID
            
            # ONDC Settlement Fields with Mock Placeholder Values
            '@ondc/org/settlement_basis': self.mock_settlement_basis,    # MOCK: 'delivery'
            '@ondc/org/settlement_window': self.mock_settlement_window,  # MOCK: 'P1D' (1 day)
            '@ondc/org/withholding_amount': self.mock_withholding_amount, # MOCK: '0.00'
            
            # Mock Indicators - Clear labeling
            '_payment_mode': 'MOCK_TESTING',  # CLEAR MOCK INDICATOR
            '_mock_source': 'himira_postman_collection',
            '_mock_settlement_values': True,
            '_mock_timestamp': datetime.now().isoformat()
        }
        
        if self.debug_logs:
            logger.info(f"[MOCK BIAP PAYMENT] Created payment object for amount: {total_amount}")
            logger.info(f"[MOCK BIAP PAYMENT] Payment ID: {self.mock_payment_id}")
            logger.info(f"[MOCK BIAP PAYMENT] Settlement basis: {self.mock_settlement_basis}")
            logger.info(f"[MOCK BIAP PAYMENT] Settlement window: {self.mock_settlement_window}")
        
        return payment_obj
    
    def verify_mock_payment(self, payment_id: str, signature: str = None) -> Dict[str, Any]:
        """
        MOCK IMPLEMENTATION - Simulates payment verification
        
        Args:
            payment_id: Payment ID to verify
            signature: Payment signature (ignored in mock)
            
        Returns:
            Mock verification response
        """
        if not self.mock_mode:
            raise ValueError("Mock service called when PAYMENT_MOCK_MODE is disabled")
        
        # For mock, always return success if payment ID matches
        is_valid = payment_id == self.mock_payment_id
        
        verification_result = {
            "verified": is_valid,
            "payment_id": payment_id,
            "status": "verified" if is_valid else "failed",
            "_mock_verification": True,
            "_mock_expected_id": self.mock_payment_id,
            "_mock_timestamp": datetime.now().isoformat()
        }
        
        if self.debug_logs:
            logger.info(f"[MOCK PAYMENT VERIFICATION] Payment ID: {payment_id}")
            logger.info(f"[MOCK PAYMENT VERIFICATION] Verified: {is_valid}")
        
        return verification_result
    
    def is_mock_mode_enabled(self) -> bool:
        """Check if mock mode is enabled"""
        return self.mock_mode
    
    def get_mock_configuration(self) -> Dict[str, Any]:
        """
        Get current mock configuration for debugging
        
        Returns:
            Dictionary of mock configuration values
        """
        return {
            "mock_mode": self.mock_mode,
            "mock_payment_id": self.mock_payment_id,
            "mock_payment_status": self.mock_payment_status,
            "mock_settlement_basis": self.mock_settlement_basis,
            "mock_settlement_window": self.mock_settlement_window,
            "mock_withholding_amount": self.mock_withholding_amount,
            "debug_logs": self.debug_logs,
            "source": "himira_postman_collection"
        }


# Global mock service instance
mock_payment_service = PaymentMockService()

# Convenience functions for easy import
def get_mock_payment_id() -> str:
    """Get mock payment ID from Postman collection"""
    return mock_payment_service.get_mock_payment_id()

def create_mock_payment(amount: float, currency: str = "INR") -> Dict[str, Any]:
    """Create mock payment response"""
    return mock_payment_service.create_mock_payment(amount, currency)

def create_biap_payment_object(total_amount: float) -> Dict[str, Any]:
    """Create BIAP-compliant payment object with mock values"""
    return mock_payment_service.create_biap_payment_object(total_amount)

def is_payment_mock_enabled() -> bool:
    """Check if payment mock mode is enabled"""
    return mock_payment_service.is_mock_mode_enabled()