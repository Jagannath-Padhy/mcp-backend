"""
User Authentication Service for ONDC MCP Server

Handles user authentication with real backend integration only.
Demo users use real phone-based authentication with the backend.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import time

from ..buyer_backend_client import BuyerBackendClient
from ..models.session import Session
from ..utils.logger import get_logger
from ..utils.device_id import get_or_create_device_id
from .firebase_service import get_firebase_service

logger = get_logger(__name__)


class UserService:
    """User authentication and profile management service with real backend"""
    
    def __init__(self, buyer_backend_client: Optional[BuyerBackendClient] = None):
        """
        Initialize user service
        
        Args:
            buyer_backend_client: Client for backend API calls
        """
        self.buyer_app = buyer_backend_client or BuyerBackendClient()
        self.firebase_service = get_firebase_service()  # Firebase authentication service
        
        # Store OTP state temporarily (in production, use Redis/DB)
        self._otp_state = {}
        
        # Store current auth session
        self._current_auth = None
        
        logger.info("UserService initialized with Firebase and backend authentication")
    
    # ================================
    # AUTHENTICATION METHODS
    # ================================
    
    async def send_otp(self, phone: str = None) -> Tuple[bool, str, Optional[Dict]]:
        """
        Send OTP to phone number
        
        Args:
            phone: Phone number (uses default if not provided)
            
        Returns:
            Tuple of (success, message, data)
        """
        try:
            if not phone:
                return False, " Phone number is required for OTP authentication", None
            
            # Normalize phone number
            if not phone.startswith('+'):
                if phone.startswith('91') and len(phone) == 12:
                    phone = f"+{phone}"
                elif len(phone) == 10:
                    phone = f"+91{phone}"
            
            logger.info(f"[Auth] Requesting OTP for: {phone}")
            
            # Try Firebase authentication first (real SMS OTP)
            firebase_success, firebase_msg, firebase_data = await self.firebase_service.send_otp(phone)
            
            if firebase_success:
                # Store phone in OTP state with Firebase data
                self._otp_state[phone] = {
                    "status": "pending", 
                    "timestamp": time.time(),
                    "provider": "firebase",
                    "session_data": firebase_data
                }
                
                logger.info(f"[Auth] Firebase OTP sent successfully to: {phone}")
                return True, firebase_msg, firebase_data
            
            # Fallback to backend API if Firebase fails
            logger.warning(f"Firebase OTP failed, trying backend fallback: {firebase_msg}")
            result = await self.buyer_app.signup({"phone": phone})
            
            if result and result.get("success"):
                # Store phone in OTP state with backend data
                self._otp_state[phone] = {
                    "status": "pending", 
                    "timestamp": time.time(),
                    "provider": "backend"
                }
                
                logger.info(f"[Auth] Backend OTP sent successfully to: {phone}")
                return True, f" OTP sent to {phone}. Please enter the 6-digit OTP you received.", {
                    "phone": phone,
                    "otp_requested": True,
                    "provider": "backend"
                }
            else:
                error_msg = result.get("message", "Failed to send OTP") if result else "Backend connection failed"
                logger.error(f"[Auth] Both Firebase and backend OTP failed for {phone}")
                return False, f" OTP sending failed: {error_msg}", None
                
        except Exception as e:
            logger.error(f"[Auth] Send OTP error: {e}")
            return False, f" Error: {str(e)}", None
    
    async def verify_otp(self, phone: str, otp: str) -> Tuple[bool, str, Optional[Dict]]:
        """
        Verify OTP and complete login
        
        Args:
            phone: Phone number
            otp: OTP code (or password for now)
            
        Returns:
            Tuple of (success, message, auth_data)
        """
        try:
            if not phone:
                return False, " Phone number is required for OTP verification", None
            
            # Normalize phone number
            if not phone.startswith('+'):
                if phone.startswith('91') and len(phone) == 12:
                    phone = f"+{phone}"
                elif len(phone) == 10:
                    phone = f"+91{phone}"
            
            logger.info(f"[Auth] Verifying OTP for: {phone}")
            
            # Check if we have an OTP session and which provider was used
            otp_session = self._otp_state.get(phone)
            provider = otp_session.get("provider", "backend") if otp_session else "backend"
            
            # Try Firebase verification first if Firebase was used for OTP
            if provider == "firebase":
                firebase_success, firebase_msg, firebase_data = await self.firebase_service.verify_otp(phone, otp)
                
                if firebase_success:
                    logger.info(f"[Auth] Firebase OTP verification successful for: {phone}")
                    
                    # Clear OTP state
                    if phone in self._otp_state:
                        del self._otp_state[phone]
                    
                    # Create user session with Firebase auth
                    self._current_auth = {
                        "token": firebase_data.get("token"),
                        "user": {
                            "phone": phone,
                            "provider": "firebase"
                        },
                        "phone": phone,
                        "authenticated": True,
                        "provider": "firebase"
                    }
                    
                    # Also try to get/create user profile from backend
                    try:
                        profile_result = await self._get_or_create_user_profile(phone)
                        if profile_result:
                            self._current_auth["user"].update(profile_result)
                    except Exception as e:
                        logger.warning(f"Failed to get user profile from backend: {e}")
                    
                    return True, f" {firebase_msg}", self._current_auth
                else:
                    logger.warning(f"Firebase OTP verification failed: {firebase_msg}")
                    # Fall through to backend verification
            
            # Verify OTP via backend API - fallback or primary method
            try:
                result = await self.buyer_app.verify_otp({
                    "phone": phone,
                    "otp": otp
                })
                
                # Backend returns success response with token and user data
                if result and (result.get("type") == "success" or result.get("success")):
                    auth_data = result.get("data", {})
                    token = auth_data.get("token")
                    user_id = auth_data.get("userId")
                    
                    if token and user_id:
                        logger.info(f"[Auth] OTP verification successful for: {phone}")
                        
                        # Clear OTP state
                        if phone in self._otp_state:
                            del self._otp_state[phone]
                        
                        # Store auth for session - matches frontend structure
                        self._current_auth = {
                            "token": token,
                            "user": {
                                "userId": user_id,
                                "phone": phone
                            },
                            "user_id": user_id,
                            "phone": phone,
                            "authenticated": True
                        }
                        
                        # Create/update user profile in backend - matches frontend flow
                        profile_success = await self._create_user_profile(token, user_id, phone)
                        if profile_success:
                            return True, " Login successful!", self._current_auth
                        else:
                            return True, " OTP verified - profile setup needed", self._current_auth
                    else:
                        logger.warning(f"[Auth] Incomplete auth data for: {phone}")
                        return False, " Authentication incomplete. Please try again.", None
                else:
                    error_msg = result.get("message", "Invalid OTP") if result else "Verification failed"
                    logger.warning(f"[Auth] OTP verification failed for: {phone} - {error_msg}")
                    return False, f" {error_msg}", None
                    
            except Exception as e:
                logger.error(f"[Auth] Backend OTP verification error for {phone}: {e}")
                return False, f" Verification failed: {str(e)}", None
                
        except Exception as e:
            logger.error(f"[Auth] Verify OTP error: {e}")
            return False, f" Error: {str(e)}", None
    
    async def login(self, username: str = None, password: str = None, use_otp: bool = True) -> Tuple[bool, str, Optional[Dict]]:
        """
        User login - initiates OTP flow by default
        
        Args:
            username: Phone number (uses default if not provided)
            password: Not used for OTP flow
            use_otp: If True, use OTP flow (default)
            
        Returns:
            Tuple of (success, message, auth_data)
        """
        try:
            if not username:
                return False, " Phone number is required for login", None
            phone = username
            
            if use_otp:
                # Initiate OTP flow
                logger.info(f"[Auth] Initiating OTP flow for: {phone}")
                return await self.send_otp(phone)
            else:
                # Direct OTP verification  
                if not password:
                    return False, " OTP is required for verification", None
                logger.info(f"[Auth] Using direct OTP verification for: {phone}")
                return await self.verify_otp(phone, password)
                
        except Exception as e:
            logger.error(f"[Auth] Login error: {e}")
            return False, f" Login error: {str(e)}", None
    
    async def signup(self, email: str, password: str, name: str, phone: str, demo_mode: bool = False) -> Tuple[bool, str, Optional[Dict]]:
        """
        User signup - always uses real backend
        
        Args:
            email: User email
            password: User password
            name: Full name
            phone: Phone number
            demo_mode: Ignored (kept for compatibility)
            
        Returns:
            Tuple of (success, message, auth_data)
        """
        try:
            logger.info(f"[Auth] Signup attempt for: {phone}")
            
            # Normalize phone number
            if not phone.startswith('+'):
                if phone.startswith('91') and len(phone) == 12:
                    phone = f"+{phone}"
                elif len(phone) == 10:
                    phone = f"+91{phone}"
            
            # Try to create user via backend
            try:
                # First try regular signup
                result = await self.buyer_app.signup({
                    "email": email,
                    "password": password,
                    "name": name,
                    "phone": phone
                })
                
                if result and not result.get("error"):
                    logger.info(f"[Auth] Signup successful for: {phone}")
                    user_data = result.get("user", {})
                    user_id = user_data.get("userId") or user_data.get("_id") or phone
                    
                    return True, " Account created successfully", {
                        "token": result.get("token"),
                        "user": user_data,
                        "user_id": user_id,
                        "phone": phone,
                        "is_demo": False
                    }
                else:
                    # If signup fails, try login (user might exist)
                    login_result = await self.buyer_app.login_with_phone({
                        "phone": phone,
                        "password": password
                    })
                    
                    if login_result and login_result.get("token"):
                        logger.info(f"[Auth] User exists, logged in: {phone}")
                        user_data = login_result.get("user", {})
                        user_id = user_data.get("userId") or user_data.get("_id") or phone
                        
                        return True, " User already exists. Logged in successfully.", {
                            "token": login_result["token"],
                            "user": user_data,
                            "user_id": user_id,
                            "phone": phone,
                            "is_demo": False
                        }
                    else:
                        return False, " Signup failed. Please try again.", None
                    
            except Exception as e:
                logger.error(f"[Auth] Signup error: {e}")
                return False, f" Backend connection error: {str(e)}", None
                
        except Exception as e:
            logger.error(f"[Auth] Signup error: {e}")
            return False, f" Signup failed: {str(e)}", None
    
    async def verify_token(self, auth_token: str, demo_mode: bool = False) -> Tuple[bool, str, Optional[Dict]]:
        """
        Verify authentication token with real backend
        
        Args:
            auth_token: Authentication token to verify
            demo_mode: Ignored (kept for compatibility)
            
        Returns:
            Tuple of (is_valid, message, user_data)
        """
        try:
            # Always verify with real backend
            result = await self.buyer_app.get_user_profile(auth_token)
            if result and not result.get("error"):
                logger.info("[Auth] Token verified successfully")
                return True, " Token valid", result
            else:
                logger.warning("[Auth] Token verification failed")
                return False, " Invalid or expired token", None
                
        except Exception as e:
            logger.error(f"[Auth] Token verification error: {e}")
            return False, f" Token verification failed: {str(e)}", None
    
    async def _get_or_create_user_profile(self, phone: str) -> Optional[Dict]:
        """
        Get or create user profile from backend
        
        Args:
            phone: Phone number
            
        Returns:
            User profile data or None
        """
        try:
            # Try to get user by phone via backend
            result = await self.buyer_app.login_with_phone({"phone": phone})
            
            if result and result.get("success"):
                user_data = result.get("user", {})
                return {
                    "userId": user_data.get("_id"),
                    "userName": user_data.get("userName"),
                    "email": user_data.get("email"),
                    "phone": user_data.get("phone"),
                    "tracking_id": user_data.get("tracking_id")
                }
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to get user profile for {phone}: {e}")
            return None
    
    # ================================
    # PROFILE MANAGEMENT
    # ================================
    
    async def get_user_profile(self, auth_token: str) -> Tuple[bool, str, Optional[Dict]]:
        """
        Get user profile from backend
        
        Args:
            auth_token: Authentication token
            
        Returns:
            Tuple of (success, message, profile_data)
        """
        try:
            # Get profile from backend
            result = await self.buyer_app.get_user_profile(auth_token)
            
            if result and not result.get("error"):
                return True, " Profile retrieved", result
            else:
                return False, " Failed to get profile", None
                
        except Exception as e:
            logger.error(f"[Profile] Get profile error: {e}")
            return False, f" Failed to get profile: {str(e)}", None
    
    async def update_user_profile(self, auth_token: str, profile_data: Dict[str, Any]) -> Tuple[bool, str, Optional[Dict]]:
        """
        Update user profile in backend
        
        Args:
            auth_token: Authentication token
            profile_data: Updated profile data
            
        Returns:
            Tuple of (success, message, updated_profile)
        """
        try:
            # Update profile in backend
            result = await self.buyer_app.update_user_profile(profile_data, auth_token)
            
            if result and not result.get("error"):
                return True, " Profile updated successfully", result
            else:
                return False, f" Profile update failed: {result.get('error', 'Unknown error')}", None
                
        except Exception as e:
            logger.error(f"[Profile] Update profile error: {e}")
            return False, f" Profile update failed: {str(e)}", None
    
    # ================================
    # UTILITY METHODS
    # ================================
    
    
    def format_user_profile(self, user_data: Dict[str, Any]) -> str:
        """
        Format user profile for display
        
        Args:
            user_data: User profile data
            
        Returns:
            Formatted profile string
        """
        try:
            name = user_data.get('userName') or user_data.get('name') or 'Unknown'
            email = user_data.get('email') or 'No email'
            phone = user_data.get('phone') or 'No phone'
            address = user_data.get('address') or 'No address saved'
            user_id = user_data.get('userId') or user_data.get('_id') or 'Unknown'
            
            lines = [
                f" **{name}**",
                f" Email: {email}",
                f" Phone: {phone}",
                f" Address: {address}",
                f" User ID: {user_id}"
            ]
            
            
            return "\n".join(lines)
            
        except Exception as e:
            logger.error(f"[Profile] Format profile error: {e}")
            return f"User Profile: {user_data.get('name', 'Unknown')}"
    
    def is_authenticated(self, session: Session) -> bool:
        """
        Check if session is authenticated
        
        Args:
            session: User session
            
        Returns:
            True if authenticated
        """
        return session.user_authenticated and session.auth_token is not None
    
    async def _create_user_profile(self, auth_token: str, user_id: str, phone: str) -> bool:
        """
        Create/update user profile in backend - matches frontend handleUserLogin
        
        Args:
            auth_token: Backend JWT token
            user_id: User ID from OTP verification
            phone: User's phone number
            
        Returns:
            True if profile created/updated successfully
        """
        try:
            # Get/create device ID - matches frontend pattern
            device_id = await get_or_create_device_id()
            
            # Prepare payload matching frontend structure
            payload = {
                "userId": user_id,
                "deviceId": device_id,
                "phone": phone,
                "signup_with": "phone"
            }
            
            logger.info(f"[Profile] Creating user profile for {user_id} with phone {phone}")
            
            # Call backend userProfile API
            result = await self.buyer_app.update_user_profile(payload, auth_token)
            
            if result and result.get("success"):
                user_data = result.get("data", {})
                logger.info(f"[Profile] User profile created successfully for {user_id}")
                
                # Update current auth with complete user data
                if self._current_auth and self._current_auth.get("user"):
                    self._current_auth["user"].update({
                        "userName": user_data.get("userName"),
                        "email": user_data.get("email"),
                        "deviceId": device_id
                    })
                
                return True
            else:
                error_msg = result.get("message", "Unknown error") if result else "No response"
                logger.warning(f"[Profile] Failed to create profile for {user_id}: {error_msg}")
                return False
                
        except Exception as e:
            logger.error(f"[Profile] Error creating user profile for {user_id}: {e}")
            return False


# Singleton instance
_user_service: Optional[UserService] = None


def get_user_service() -> UserService:
    """Get singleton UserService instance"""
    global _user_service
    if _user_service is None:
        _user_service = UserService()
    return _user_service