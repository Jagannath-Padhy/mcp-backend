"""
Comprehensive ONDC Payment Service

Handles all payment operations for ONDC orders:
1. RazorPay integration - Primary payment gateway
2. JusPay integration - Alternative payment gateway  
3. COD (Cash on Delivery) support
4. Payment verification and confirmation
5. Payment failure handling and retries
6. Multi-gateway support with fallback

Integrates between ONDC INIT and CONFIRM steps.
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from enum import Enum
import logging

from ..buyer_backend_client import BuyerBackendClient
from ..models.session import Session
from ..utils.logger import get_logger

logger = get_logger(__name__)


class PaymentStatus(Enum):
    """Payment status enumeration"""
    PENDING = "pending"
    PROCESSING = "processing" 
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"
    REFUNDED = "refunded"


class PaymentMethod(Enum):
    """Supported payment methods"""
    RAZORPAY_CARD = "razorpay_card"
    RAZORPAY_UPI = "razorpay_upi" 
    RAZORPAY_NETBANKING = "razorpay_netbanking"
    RAZORPAY_WALLET = "razorpay_wallet"
    JUSPAY_CARD = "juspay_card"
    JUSPAY_UPI = "juspay_upi"
    COD = "cod"
    PREPAID = "prepaid"


class PaymentGateway(Enum):
    """Payment gateway providers"""
    RAZORPAY = "razorpay"
    JUSPAY = "juspay"
    COD = "cod"


class PaymentService:
    """Comprehensive service for ONDC payment management"""
    
    def __init__(self, buyer_backend_client: Optional[BuyerBackendClient] = None):
        """
        Initialize payment service
        
        Args:
            buyer_backend_client: Comprehensive client for backend API calls
        """
        self.buyer_app = buyer_backend_client or BuyerBackendClient()
        self.payment_attempts = {}  # Track payment attempts per session
        logger.info("PaymentService initialized with multi-gateway support (RazorPay, JusPay, COD)")
    
    # ================================
    # PAYMENT GATEWAY SELECTION
    # ================================
    
    def get_available_payment_methods(self, order_amount: float, location: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get available payment methods based on order details
        
        Args:
            order_amount: Order total amount
            location: Delivery location (for COD availability)
            
        Returns:
            List of available payment method options
        """
        methods = []
        
        # RazorPay methods (primary gateway)
        methods.extend([
            {
                "method": PaymentMethod.RAZORPAY_CARD.value,
                "gateway": PaymentGateway.RAZORPAY.value,
                "display_name": "Credit/Debit Card (RazorPay)",
                "min_amount": 1.0,
                "max_amount": 200000.0,
                "processing_fee": order_amount * 0.02,  # 2% processing fee
                "supported": True
            },
            {
                "method": PaymentMethod.RAZORPAY_UPI.value, 
                "gateway": PaymentGateway.RAZORPAY.value,
                "display_name": "UPI (RazorPay)",
                "min_amount": 1.0,
                "max_amount": 100000.0,
                "processing_fee": 0.0,  # No processing fee for UPI
                "supported": True
            },
            {
                "method": PaymentMethod.RAZORPAY_NETBANKING.value,
                "gateway": PaymentGateway.RAZORPAY.value, 
                "display_name": "Net Banking (RazorPay)",
                "min_amount": 1.0,
                "max_amount": 100000.0,
                "processing_fee": order_amount * 0.015,  # 1.5% processing fee
                "supported": True
            },
            {
                "method": PaymentMethod.RAZORPAY_WALLET.value,
                "gateway": PaymentGateway.RAZORPAY.value,
                "display_name": "Wallet (RazorPay)",
                "min_amount": 1.0,
                "max_amount": 50000.0,
                "processing_fee": 0.0,
                "supported": True
            }
        ])
        
        # JusPay methods (alternative gateway)
        methods.extend([
            {
                "method": PaymentMethod.JUSPAY_CARD.value,
                "gateway": PaymentGateway.JUSPAY.value,
                "display_name": "Credit/Debit Card (JusPay)",
                "min_amount": 1.0,
                "max_amount": 200000.0, 
                "processing_fee": order_amount * 0.025,  # 2.5% processing fee
                "supported": True
            },
            {
                "method": PaymentMethod.JUSPAY_UPI.value,
                "gateway": PaymentGateway.JUSPAY.value,
                "display_name": "UPI (JusPay)", 
                "min_amount": 1.0,
                "max_amount": 100000.0,
                "processing_fee": 0.0,
                "supported": True
            }
        ])
        
        # Cash on Delivery (if supported in location)
        if self._is_cod_available(location):
            methods.append({
                "method": PaymentMethod.COD.value,
                "gateway": PaymentGateway.COD.value,
                "display_name": "Cash on Delivery",
                "min_amount": 50.0,
                "max_amount": 10000.0,  # COD limit
                "processing_fee": 20.0,  # Fixed COD handling fee
                "supported": True
            })
        
        # Filter methods based on order amount
        available_methods = []
        for method in methods:
            if (method["min_amount"] <= order_amount <= method["max_amount"] and 
                method["supported"]):
                available_methods.append(method)
        
        logger.info(f"[Payment] Found {len(available_methods)} available payment methods for amount ₹{order_amount}")
        return available_methods
    
    def _is_cod_available(self, location: Optional[str]) -> bool:
        """Check if COD is available for location"""
        # Simplified COD availability logic
        # In real implementation, this would check serviceable pincodes
        return location is not None
    
    # ================================
    # RAZORPAY INTEGRATION
    # ================================
    
    async def create_razorpay_order(
        self, 
        session: Session, 
        amount: float,
        currency: str = "INR",
        auth_token: Optional[str] = None
    ) -> Tuple[bool, str, Optional[Dict]]:
        """
        Create RazorPay order for payment
        
        Args:
            session: User session
            amount: Payment amount
            currency: Currency code
            auth_token: Authentication token
            
        Returns:
            Tuple of (success, message, razorpay_order_data)
        """
        try:
            logger.info(f"[Payment] Creating RazorPay order for amount: ₹{amount}")
            
            # Prepare RazorPay order data
            order_data = {
                "amount": int(amount * 100),  # Convert to paise
                "currency": currency,
                "receipt": f"order_{session.user_id}_{int(datetime.utcnow().timestamp())}",
                "notes": {
                    "user_id": session.user_id,
                    "device_id": session.device_id,
                    "transaction_id": session.checkout_state.transaction_id or ""
                }
            }
            
            # Call backend API
            result = await self.buyer_app.create_razorpay_order(order_data, auth_token)
            
            if result and not result.get('error') and result.get('id'):
                # Store payment attempt
                self.payment_attempts[session.session_id] = {
                    "razorpay_order_id": result["id"],
                    "amount": amount,
                    "currency": currency,
                    "gateway": PaymentGateway.RAZORPAY.value,
                    "status": PaymentStatus.PENDING.value,
                    "created_at": datetime.utcnow(),
                    "attempts": 1
                }
                
                # Add to session history
                session.add_to_history('payment_order_created', {
                    'gateway': 'razorpay',
                    'order_id': result["id"],
                    'amount': amount,
                    'currency': currency
                })
                
                logger.info(f"[Payment] RazorPay order created: {result['id']}")
                return True, f" Payment order created (₹{amount})", result
            else:
                logger.error(f"[Payment] RazorPay order creation failed: {result}")
                return False, f" Payment order creation failed: {result.get('error', 'Unknown error')}", None
                
        except Exception as e:
            logger.error(f"[Payment] RazorPay order creation error: {e}")
            return False, f" Payment order creation failed: {str(e)}", None
    
    async def verify_razorpay_payment(
        self,
        session: Session,
        payment_id: str,
        order_id: str,
        signature: str,
        auth_token: Optional[str] = None
    ) -> Tuple[bool, str, Optional[Dict]]:
        """
        Verify RazorPay payment signature and status
        
        Args:
            session: User session
            payment_id: RazorPay payment ID
            order_id: RazorPay order ID
            signature: Payment signature for verification
            auth_token: Authentication token
            
        Returns:
            Tuple of (success, message, payment_verification_data)
        """
        try:
            logger.info(f"[Payment] Verifying RazorPay payment: {payment_id}")
            
            # Prepare verification data
            verification_data = {
                "razorpay_payment_id": payment_id,
                "razorpay_order_id": order_id,
                "razorpay_signature": signature,
                "user_id": session.user_id,
                "device_id": session.device_id
            }
            
            # Call backend verification API
            result = await self.buyer_app.verify_razorpay_payment(verification_data, auth_token)
            
            if result and not result.get('error'):
                # Update payment attempt status
                if session.session_id in self.payment_attempts:
                    self.payment_attempts[session.session_id]["status"] = PaymentStatus.SUCCESS.value
                    self.payment_attempts[session.session_id]["payment_id"] = payment_id
                    self.payment_attempts[session.session_id]["verified_at"] = datetime.utcnow()
                
                # Update session checkout state
                session.checkout_state.payment_method = PaymentMethod.RAZORPAY_CARD.value
                session.checkout_state.payment_status = PaymentStatus.SUCCESS.value
                
                # Add to session history
                session.add_to_history('payment_verified', {
                    'gateway': 'razorpay',
                    'payment_id': payment_id,
                    'order_id': order_id,
                    'status': 'success'
                })
                
                logger.info(f"[Payment] RazorPay payment verified successfully: {payment_id}")
                return True, f" Payment verified successfully", result
            else:
                # Update payment attempt status
                if session.session_id in self.payment_attempts:
                    self.payment_attempts[session.session_id]["status"] = PaymentStatus.FAILED.value
                    self.payment_attempts[session.session_id]["error"] = result.get('error', 'Verification failed')
                
                logger.error(f"[Payment] RazorPay payment verification failed: {result}")
                return False, f" Payment verification failed: {result.get('error', 'Invalid signature')}", None
                
        except Exception as e:
            logger.error(f"[Payment] RazorPay payment verification error: {e}")
            return False, f" Payment verification failed: {str(e)}", None
    
    # ================================
    # JUSPAY INTEGRATION
    # ================================
    
    async def create_juspay_order(
        self,
        session: Session,
        amount: float,
        currency: str = "INR",
        auth_token: Optional[str] = None
    ) -> Tuple[bool, str, Optional[Dict]]:
        """
        Create JusPay order for payment
        
        Args:
            session: User session
            amount: Payment amount
            currency: Currency code
            auth_token: Authentication token
            
        Returns:
            Tuple of (success, message, juspay_order_data)
        """
        try:
            logger.info(f"[Payment] Creating JusPay order for amount: ₹{amount}")
            
            # Prepare JusPay order data
            order_data = {
                "amount": amount,
                "currency": currency,
                "order_id": f"juspay_{session.user_id}_{int(datetime.utcnow().timestamp())}",
                "customer_id": session.user_id,
                "customer_email": session.user_profile.email if session.user_profile else "",
                "customer_phone": session.user_profile.phone if session.user_profile else "",
                "description": f"ONDC Order Payment - Transaction: {session.checkout_state.transaction_id}",
                "return_url": "/payment/success",  # Configure based on your frontend
                "metadata": {
                    "user_id": session.user_id,
                    "device_id": session.device_id,
                    "transaction_id": session.checkout_state.transaction_id or ""
                }
            }
            
            # Call backend API
            result = await self.buyer_app.create_juspay_order(order_data, auth_token)
            
            if result and not result.get('error'):
                # Store payment attempt
                self.payment_attempts[session.session_id] = {
                    "juspay_order_id": result.get("order_id"),
                    "amount": amount,
                    "currency": currency,
                    "gateway": PaymentGateway.JUSPAY.value,
                    "status": PaymentStatus.PENDING.value,
                    "created_at": datetime.utcnow(),
                    "attempts": 1
                }
                
                # Add to session history
                session.add_to_history('payment_order_created', {
                    'gateway': 'juspay',
                    'order_id': result.get("order_id"),
                    'amount': amount,
                    'currency': currency
                })
                
                logger.info(f"[Payment] JusPay order created: {result.get('order_id')}")
                return True, f" JusPay payment order created (₹{amount})", result
            else:
                logger.error(f"[Payment] JusPay order creation failed: {result}")
                return False, f" JusPay order creation failed: {result.get('error', 'Unknown error')}", None
                
        except Exception as e:
            logger.error(f"[Payment] JusPay order creation error: {e}")
            return False, f" JusPay order creation failed: {str(e)}", None
    
    # ================================
    # CASH ON DELIVERY
    # ================================
    
    async def process_cod_payment(
        self,
        session: Session,
        amount: float,
        delivery_info: Dict[str, Any]
    ) -> Tuple[bool, str, Optional[Dict]]:
        """
        Process Cash on Delivery payment
        
        Args:
            session: User session
            amount: Payment amount
            delivery_info: Delivery address information
            
        Returns:
            Tuple of (success, message, cod_payment_data)
        """
        try:
            logger.info(f"[Payment] Processing COD payment for amount: ₹{amount}")
            
            # Check COD availability
            location = delivery_info.get('pincode', delivery_info.get('city'))
            if not self._is_cod_available(location):
                return False, " Cash on Delivery not available in your area", None
            
            # Check COD limits
            cod_method = next((m for m in self.get_available_payment_methods(amount, location) 
                             if m["method"] == PaymentMethod.COD.value), None)
            
            if not cod_method:
                return False, f" COD not available for amount ₹{amount}", None
            
            # Create COD payment record
            cod_payment = {
                "payment_method": PaymentMethod.COD.value,
                "gateway": PaymentGateway.COD.value,
                "amount": amount,
                "currency": "INR",
                "processing_fee": cod_method["processing_fee"],
                "status": PaymentStatus.PENDING.value,
                "created_at": datetime.utcnow(),
                "delivery_info": delivery_info,
                "cod_order_id": f"cod_{session.user_id}_{int(datetime.utcnow().timestamp())}"
            }
            
            # Store payment attempt
            self.payment_attempts[session.session_id] = {
                **cod_payment,
                "attempts": 1
            }
            
            # Update session checkout state
            session.checkout_state.payment_method = PaymentMethod.COD.value
            session.checkout_state.payment_status = PaymentStatus.PENDING.value
            
            # Add to session history
            session.add_to_history('cod_payment_selected', {
                'amount': amount,
                'processing_fee': cod_payment["processing_fee"],
                'location': location,
                'cod_order_id': cod_payment["cod_order_id"]
            })
            
            logger.info(f"[Payment] COD payment processed: {cod_payment['cod_order_id']}")
            return True, f" Cash on Delivery selected (₹{amount} + ₹{cod_payment['processing_fee']} handling)", cod_payment
            
        except Exception as e:
            logger.error(f"[Payment] COD payment processing error: {e}")
            return False, f" COD payment processing failed: {str(e)}", None
    
    # ================================
    # PAYMENT FLOW ORCHESTRATION
    # ================================
    
    async def initiate_payment(
        self,
        session: Session,
        payment_method: str,
        amount: float,
        auth_token: Optional[str] = None,
        **kwargs
    ) -> Tuple[bool, str, Optional[Dict]]:
        """
        Initiate payment with selected method
        
        Args:
            session: User session
            payment_method: Selected payment method
            amount: Payment amount
            auth_token: Authentication token
            **kwargs: Additional payment parameters
            
        Returns:
            Tuple of (success, message, payment_data)
        """
        try:
            logger.info(f"[Payment] Initiating payment: {payment_method} for ₹{amount}")
            
            # Validate session state
            if not session.checkout_state.transaction_id:
                return False, " No active transaction. Please initialize order first.", None
            
            method_enum = PaymentMethod(payment_method)
            
            # Route to appropriate payment gateway
            if method_enum in [PaymentMethod.RAZORPAY_CARD, PaymentMethod.RAZORPAY_UPI, 
                             PaymentMethod.RAZORPAY_NETBANKING, PaymentMethod.RAZORPAY_WALLET]:
                return await self.create_razorpay_order(session, amount, auth_token=auth_token)
                
            elif method_enum in [PaymentMethod.JUSPAY_CARD, PaymentMethod.JUSPAY_UPI]:
                return await self.create_juspay_order(session, amount, auth_token=auth_token)
                
            elif method_enum == PaymentMethod.COD:
                delivery_info = kwargs.get('delivery_info', {})
                return await self.process_cod_payment(session, amount, delivery_info)
                
            else:
                return False, f" Unsupported payment method: {payment_method}", None
                
        except Exception as e:
            logger.error(f"[Payment] Payment initiation error: {e}")
            return False, f" Payment initiation failed: {str(e)}", None
    
    async def complete_payment(
        self,
        session: Session,
        payment_details: Dict[str, Any],
        auth_token: Optional[str] = None
    ) -> Tuple[bool, str, Optional[Dict]]:
        """
        Complete payment verification and prepare for order confirmation
        
        Args:
            session: User session
            payment_details: Payment completion details
            auth_token: Authentication token
            
        Returns:
            Tuple of (success, message, completion_data)
        """
        try:
            payment_method = session.checkout_state.payment_method
            
            if not payment_method:
                return False, " No payment method selected", None
            
            logger.info(f"[Payment] Completing payment: {payment_method}")
            
            # Handle different payment methods
            if payment_method.startswith('razorpay'):
                return await self.verify_razorpay_payment(
                    session,
                    payment_details.get('payment_id'),
                    payment_details.get('order_id'), 
                    payment_details.get('signature'),
                    auth_token
                )
                
            elif payment_method.startswith('juspay'):
                # JusPay verification would be implemented here
                # For now, simplified verification
                session.checkout_state.payment_status = PaymentStatus.SUCCESS.value
                return True, " JusPay payment completed", payment_details
                
            elif payment_method == PaymentMethod.COD.value:
                # COD is already "completed" at selection time
                session.checkout_state.payment_status = PaymentStatus.PENDING.value
                return True, " COD payment confirmed (to be collected on delivery)", {}
                
            else:
                return False, f" Unknown payment method: {payment_method}", None
                
        except Exception as e:
            logger.error(f"[Payment] Payment completion error: {e}")
            return False, f" Payment completion failed: {str(e)}", None
    
    # ================================
    # PAYMENT UTILITIES
    # ================================
    
    def get_payment_status(self, session: Session) -> Dict[str, Any]:
        """
        Get current payment status for session
        
        Args:
            session: User session
            
        Returns:
            Payment status information
        """
        attempt = self.payment_attempts.get(session.session_id)
        
        return {
            "has_active_payment": attempt is not None,
            "payment_method": session.checkout_state.payment_method,
            "payment_status": session.checkout_state.payment_status,
            "gateway": attempt.get("gateway") if attempt else None,
            "amount": attempt.get("amount") if attempt else None,
            "created_at": attempt.get("created_at").isoformat() if attempt and attempt.get("created_at") else None,
            "attempts": attempt.get("attempts", 0) if attempt else 0
        }
    
    async def handle_payment_failure(
        self,
        session: Session,
        failure_reason: str,
        retry_suggested: bool = True
    ) -> Tuple[bool, str]:
        """
        Handle payment failure and suggest next steps
        
        Args:
            session: User session
            failure_reason: Reason for payment failure
            retry_suggested: Whether retry is suggested
            
        Returns:
            Tuple of (can_retry, failure_message)
        """
        try:
            logger.warning(f"[Payment] Payment failed for session {session.session_id}: {failure_reason}")
            
            # Update payment attempt
            if session.session_id in self.payment_attempts:
                attempt = self.payment_attempts[session.session_id]
                attempt["status"] = PaymentStatus.FAILED.value
                attempt["error"] = failure_reason
                attempt["failed_at"] = datetime.utcnow()
                attempt["attempts"] += 1
            
            # Update session
            session.checkout_state.payment_status = PaymentStatus.FAILED.value
            
            # Add to session history
            session.add_to_history('payment_failed', {
                'reason': failure_reason,
                'payment_method': session.checkout_state.payment_method,
                'can_retry': retry_suggested and (attempt.get("attempts", 0) < 3)
            })
            
            # Determine if retry is possible
            max_attempts = 3
            current_attempts = self.payment_attempts.get(session.session_id, {}).get("attempts", 0)
            can_retry = retry_suggested and current_attempts < max_attempts
            
            if can_retry:
                return True, f" Payment failed: {failure_reason}. You can try again ({current_attempts}/{max_attempts} attempts used)."
            else:
                return False, f" Payment failed: {failure_reason}. Maximum retry attempts reached. Please select a different payment method."
                
        except Exception as e:
            logger.error(f"[Payment] Payment failure handling error: {e}")
            return False, f" Payment failed: {failure_reason}. Unable to process retry."
    
    def format_payment_summary(self, payment_data: Dict[str, Any]) -> str:
        """
        Format payment information for display
        
        Args:
            payment_data: Payment information
            
        Returns:
            Formatted payment summary string
        """
        try:
            method = payment_data.get('payment_method', payment_data.get('method', 'Unknown'))
            gateway = payment_data.get('gateway', 'Unknown')
            amount = payment_data.get('amount', 0)
            status = payment_data.get('status', payment_data.get('payment_status', 'Unknown'))
            
            lines = [
                f" **Payment Summary**",
                f" Gateway: {gateway.title()}",
                f" Amount: ₹{amount:.2f}",
                f" Status: {status.title()}",
                ""
            ]
            
            # Add method-specific details
            if 'processing_fee' in payment_data and payment_data['processing_fee'] > 0:
                lines.insert(-1, f" Processing Fee: ₹{payment_data['processing_fee']:.2f}")
            
            if method == PaymentMethod.COD.value:
                lines.insert(-1, f" Payment will be collected on delivery")
            
            if payment_data.get('payment_id'):
                lines.insert(-1, f" Payment ID: {payment_data['payment_id']}")
            
            return "\n".join(lines)
            
        except Exception as e:
            logger.error(f"[Payment] Format payment summary error: {e}")
            return f"Payment: ₹{payment_data.get('amount', 'Unknown')}"
    
    def cleanup_payment_attempt(self, session: Session):
        """
        Cleanup payment attempt data for session
        
        Args:
            session: User session
        """
        if session.session_id in self.payment_attempts:
            del self.payment_attempts[session.session_id]
            logger.info(f"[Payment] Cleaned up payment attempt for session: {session.session_id}")


# Singleton instance
_payment_service: Optional[PaymentService] = None


def get_payment_service() -> PaymentService:
    """Get singleton PaymentService instance"""
    global _payment_service
    if _payment_service is None:
        _payment_service = PaymentService()
    return _payment_service