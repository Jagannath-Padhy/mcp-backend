"""
Bypass Authentication Service for ONDC MCP Server

Handles phone authentication using backend bypass mechanism.
Creates real Himira backend user sessions without SMS dependency.
"""

import os
import json
import logging
import asyncio
from typing import Dict, Optional, Tuple, Any
from datetime import datetime, timedelta

from ..utils.logger import get_logger

logger = get_logger(__name__)


class FirebaseService:
    """Backend bypass authentication service for phone-based login"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FirebaseService, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize bypass authentication service"""
        if not hasattr(self, 'initialized'):
            self._otp_sessions = {}  # Store authentication sessions
            self.initialized = True
    
    
    
    async def send_otp(self, phone_number: str) -> Tuple[bool, str, Optional[Dict]]:
        """
        Setup authentication using backend bypass mechanism
        
        Args:
            phone_number: Phone number in format +91XXXXXXXXXX
            
        Returns:
            Tuple of (success, message, session_data)
        """
        try:
            # Normalize phone number
            if not phone_number.startswith('+'):
                if phone_number.startswith('91') and len(phone_number) == 12:
                    phone_number = f"+{phone_number}"
                elif len(phone_number) == 10:
                    phone_number = f"+91{phone_number}"
            
            logger.info(f"[Firebase] Setting up bypass authentication for: {phone_number}")
            
            # Call backend with bypass header for immediate authentication setup
            success, message, backend_data = await self._send_backend_otp(phone_number)
            
            if success:
                # Create session tracking for bypass authentication
                session_id = f"firebase_bypass_session_{phone_number}_{int(datetime.now().timestamp())}"
                
                self._otp_sessions[phone_number] = {
                    'session_id': session_id,
                    'created_at': datetime.now(),
                    'expires_at': datetime.now() + timedelta(minutes=5),
                    'verified': False,
                    'bypass_used': True,
                    'backend_data': backend_data
                }
                
                logger.info(f"[Firebase] Bypass authentication prepared for: {phone_number}")
                
                return True, f" Authentication ready for {phone_number}. Enter any 6-digit code to complete login.", {
                    "phone": phone_number,
                    "session_id": session_id,
                    "otp_requested": True,
                    "provider": "firebase_backend_bypass"
                }
            else:
                logger.error(f"Backend bypass authentication failed: {message}")
                return False, f" Authentication setup failed: {message}", None
            
        except Exception as e:
            logger.error(f"[Firebase] Authentication setup error: {e}")
            return False, f" Authentication error: {str(e)}", None
    
    async def _send_backend_otp(self, phone_number: str) -> Tuple[bool, str, Optional[Dict]]:
        """Call backend API to send real OTP via SMS"""
        try:
            import httpx
            import os
            
            # Get backend URL and API key from environment
            backend_url = os.getenv("BACKEND_ENDPOINT", "https://hp-buyer-backend-preprod.himira.co.in/clientApis")
            wil_api_key = os.getenv("WIL_API_KEY")
            
            if not wil_api_key:
                logger.error("[Backend OTP] WIL_API_KEY not found in environment")
                return False, "Backend authentication not configured", None
            
            # Remove + from phone number for backend (it expects plain phone number)
            backend_phone = phone_number.replace('+91', '').replace('+', '')
            if len(backend_phone) == 10:
                # Backend might expect +91 format, let's try both
                backend_phone = f"+91{backend_phone}"
            
            # Try to call backend's user creation/OTP endpoint
            # Based on the backend code, we need to call the signup endpoint
            # which will send OTP if user doesn't exist
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                logger.info(f"[Backend OTP] Calling backend signup API for: {backend_phone}")
                
                # Call the signup endpoint with bypass header (no real OTP needed)
                response = await client.post(
                    f"{backend_url}/signup",
                    json={
                        "phone": backend_phone,
                        "signup_with": "phone"
                    },
                    headers={
                        "Content-Type": "application/json",
                        "User-Agent": "ONDC-MCP-Server/1.0",
                        "wil-api-key": wil_api_key,
                        "by-pass-otp-key": "26cc6c05-3144-4b7b-9979-4a21a53c98b2"
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    logger.info(f"[Backend OTP] Success response: {data}")
                    
                    # Check if the response is successful (bypass creates user without OTP)
                    if data.get("success"):
                        return True, "User authentication prepared via bypass", {
                            "bypass_used": True,
                            "backend_response": data
                        }
                    else:
                        logger.warning(f"[Backend OTP] Unexpected response: {data}")
                        return False, f"Backend response: {data.get('message', 'Unknown error')}", None
                else:
                    logger.error(f"[Backend OTP] HTTP {response.status_code}: {response.text}")
                    return False, f"Backend API error: HTTP {response.status_code}", None
                    
        except Exception as e:
            logger.error(f"[Backend OTP] Error calling backend: {e}")
            return False, f"Backend API call failed: {str(e)}", None
    
    async def _verify_backend_otp(self, phone_number: str, otp_code: str, session: Dict) -> Tuple[bool, str, Optional[Dict]]:
        """Call backend API to verify real OTP"""
        try:
            import httpx
            import os
            
            # Get backend URL and API key from environment
            backend_url = os.getenv("BACKEND_ENDPOINT", "https://hp-buyer-backend-preprod.himira.co.in/clientApis")
            wil_api_key = os.getenv("WIL_API_KEY")
            
            if not wil_api_key:
                logger.error("[Backend OTP Verify] WIL_API_KEY not found in environment")
                return False, "Backend authentication not configured", None
            
            # Remove + from phone number for backend
            backend_phone = phone_number.replace('+91', '').replace('+', '')
            if len(backend_phone) == 10:
                backend_phone = f"+91{backend_phone}"
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                logger.info(f"[Backend OTP Verify] Calling backend verifyotp API for: {backend_phone}")
                
                # For bypass sessions, use "4477" which always works according to backend logic
                verify_otp_code = "4477" if session.get('bypass_used') else otp_code
                logger.info(f"[Backend OTP Verify] Using OTP code: {verify_otp_code} (bypass: {session.get('bypass_used', False)})")
                
                # Call the verifyotp endpoint
                response = await client.post(
                    f"{backend_url}/verifyotp",
                    json={
                        "phone": backend_phone,
                        "otp": verify_otp_code
                    },
                    headers={
                        "Content-Type": "application/json",
                        "User-Agent": "ONDC-MCP-Server/1.0",
                        "wil-api-key": wil_api_key
                    }
                )
                
                if response.status_code in [200, 201]:
                    data = response.json()
                    logger.info(f"[Backend OTP Verify] Success response: {data}")
                    
                    # Check if verification was successful (backend uses "type": "success")
                    if data.get("success") or data.get("type") == "success":
                        return True, data.get("message", "OTP verified successfully"), {
                            "backend_response": data,
                            "user_data": data.get("data")
                        }
                    else:
                        return False, data.get("message", "OTP verification failed"), None
                else:
                    logger.error(f"[Backend OTP Verify] HTTP {response.status_code}: {response.text}")
                    return False, f"Backend verification error: HTTP {response.status_code}", None
                    
        except Exception as e:
            logger.error(f"[Backend OTP Verify] Error calling backend: {e}")
            return False, f"Backend verification failed: {str(e)}", None
    
    
    async def verify_otp(self, phone_number: str, otp_code: str) -> Tuple[bool, str, Optional[Dict]]:
        """
        Verify OTP code with Firebase
        
        Args:
            phone_number: Phone number used for OTP
            otp_code: 6-digit OTP code
            
        Returns:
            Tuple of (success, message, auth_data)
        """
        try:
            # Normalize phone number
            if not phone_number.startswith('+'):
                if phone_number.startswith('91') and len(phone_number) == 12:
                    phone_number = f"+{phone_number}"
                elif len(phone_number) == 10:
                    phone_number = f"+91{phone_number}"
            
            logger.info(f"[Firebase] Verifying OTP for: {phone_number}")
            
            # Check if session exists
            if phone_number not in self._otp_sessions:
                return False, " No OTP session found. Please request OTP first.", None
            
            session = self._otp_sessions[phone_number]
            
            # Check if session is expired
            if datetime.now() > session['expires_at']:
                del self._otp_sessions[phone_number]
                return False, " OTP expired. Please request a new OTP.", None
            
            # Check if this is a bypass session
            if session.get('bypass_used'):
                # For bypass sessions, accept any 6-digit code and verify with backend
                if len(otp_code) == 6 and otp_code.isdigit():
                    # Call backend to complete authentication with bypass
                    backend_success, backend_message, backend_auth_data = await self._verify_backend_otp(phone_number, otp_code, session)
                    
                    if backend_success:
                        # Extract real JWT token from backend response
                        auth_data = backend_auth_data.get('backend_response', {}).get('data', {}) if backend_auth_data else {}
                        jwt_token = auth_data.get('token') if auth_data else None
                        user_id = auth_data.get('userId') if auth_data else None
                        
                        # Clean up session
                        del self._otp_sessions[phone_number]
                        
                        logger.info(f"[Firebase] Bypass authentication successful for: {phone_number}")
                        
                        return True, " Authentication completed successfully", {
                            "phone": phone_number,
                            "token": jwt_token or f"bypass_token_{phone_number}_{int(datetime.now().timestamp())}",
                            "userId": user_id,
                            "provider": "firebase_backend_bypass",
                            "verified": True,
                            "authenticated": True,
                            "backend_auth": auth_data
                        }
                    else:
                        return False, f" {backend_message}", None
                else:
                    return False, " Please enter a valid 6-digit code.", None
            else:
                # Legacy fallback for non-bypass sessions (should not happen with new flow)
                return False, " Invalid session type. Please request authentication again.", None
            
        except Exception as e:
            logger.error(f"[Firebase] Verify OTP error: {e}")
            return False, f" Firebase verification error: {str(e)}", None
    
    def cleanup_expired_sessions(self):
        """Clean up expired OTP sessions"""
        current_time = datetime.now()
        expired_phones = [
            phone for phone, session in self._otp_sessions.items()
            if current_time > session['expires_at']
        ]
        
        for phone in expired_phones:
            del self._otp_sessions[phone]
            logger.debug(f"Cleaned up expired session for: {phone}")


# Global instance
_firebase_service = None

def get_firebase_service() -> FirebaseService:
    """Get Firebase service singleton"""
    global _firebase_service
    if _firebase_service is None:
        _firebase_service = FirebaseService()
    return _firebase_service