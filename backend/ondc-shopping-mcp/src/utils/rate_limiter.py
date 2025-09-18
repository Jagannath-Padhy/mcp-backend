"""Rate limiter for API calls"""

import asyncio
import time
from typing import Optional
from collections import deque
import logging

from .logger import get_logger

logger = get_logger(__name__)


class RateLimiter:
    """
    Token bucket rate limiter for API calls.
    
    Allows burst traffic up to max_tokens while maintaining
    an average rate of tokens_per_second.
    """
    
    def __init__(self, tokens_per_second: float, max_tokens: Optional[int] = None):
        """
        Initialize rate limiter.
        
        Args:
            tokens_per_second: Rate at which tokens are replenished
            max_tokens: Maximum tokens in bucket (defaults to tokens_per_second)
        """
        self.rate = tokens_per_second
        self.max_tokens = max_tokens or int(tokens_per_second)
        self.tokens = float(self.max_tokens)
        self.last_update = time.monotonic()
        self._lock = asyncio.Lock()
    
    async def acquire(self, tokens: int = 1) -> float:
        """
        Acquire tokens from the bucket.
        
        Args:
            tokens: Number of tokens to acquire
            
        Returns:
            Time waited (0 if no wait required)
        """
        async with self._lock:
            wait_time = await self._acquire_tokens(tokens)
            if wait_time > 0:
                logger.debug(f"Rate limit: waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
            return wait_time
    
    async def _acquire_tokens(self, tokens: int) -> float:
        """Internal method to calculate wait time"""
        now = time.monotonic()
        
        # Replenish tokens based on elapsed time
        elapsed = now - self.last_update
        self.tokens = min(self.max_tokens, self.tokens + elapsed * self.rate)
        self.last_update = now
        
        # Check if we have enough tokens
        if self.tokens >= tokens:
            self.tokens -= tokens
            return 0.0
        
        # Calculate wait time
        needed = tokens - self.tokens
        wait_time = needed / self.rate
        
        # Update state for after the wait
        self.tokens = 0
        return wait_time


class RequestTracker:
    """
    Track recent requests for monitoring and debugging.
    """
    
    def __init__(self, max_history: int = 100):
        """
        Initialize request tracker.
        
        Args:
            max_history: Maximum number of requests to track
        """
        self.history = deque(maxlen=max_history)
        self._lock = asyncio.Lock()
    
    async def log_request(self, method: str, path: str, duration: float, status: str):
        """Log a request"""
        async with self._lock:
            self.history.append({
                "timestamp": time.time(),
                "method": method,
                "path": path,
                "duration": duration,
                "status": status
            })
    
    async def get_stats(self) -> dict:
        """Get request statistics"""
        async with self._lock:
            if not self.history:
                return {"total": 0, "success_rate": 0.0, "avg_duration": 0.0}
            
            total = len(self.history)
            success = sum(1 for r in self.history if r["status"] == "success")
            avg_duration = sum(r["duration"] for r in self.history) / total
            
            return {
                "total": total,
                "success_rate": success / total,
                "avg_duration": avg_duration,
                "recent_errors": [r for r in list(self.history)[-10:] if r["status"] != "success"]
            }