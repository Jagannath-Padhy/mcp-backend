"""
Device ID Generation Utility - Matches Frontend Pattern
"""

import uuid
from typing import Optional
from ..utils.logger import get_logger

logger = get_logger(__name__)

# Global device ID storage (in production, this would be session-based)
_device_id: Optional[str] = None


async def get_or_create_device_id() -> str:
    """
    Generate or retrieve device ID - matches frontend pattern
    
    Returns:
        Unique device ID string
    """
    global _device_id
    
    if not _device_id:
        # Generate new device ID using UUID4 - matches frontend pattern
        _device_id = str(uuid.uuid4())
        logger.info(f"[DeviceID] Generated new device ID: {_device_id}")
    
    return _device_id


def get_device_id() -> Optional[str]:
    """
    Get current device ID without creating new one
    
    Returns:
        Current device ID or None if not set
    """
    return _device_id


def set_device_id(device_id: str) -> None:
    """
    Set device ID manually
    
    Args:
        device_id: Device ID to set
    """
    global _device_id
    _device_id = device_id
    logger.info(f"[DeviceID] Device ID set to: {device_id}")


def reset_device_id() -> None:
    """
    Reset device ID (for testing or session cleanup)
    """
    global _device_id
    _device_id = None
    logger.info("[DeviceID] Device ID reset")