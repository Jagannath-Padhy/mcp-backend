"""
Image formatter for MCP content

Handles image fetching, resizing, and base64 encoding for MCP display.
"""

import base64
from io import BytesIO
from typing import Optional, Dict
import httpx

# Optional Pillow support for image resizing
try:
    from PIL import Image
except ImportError:
    Image = None


class ImageFormatter:
    """Handles image formatting for MCP content"""
    
    def __init__(self, max_size_mb: float = 0.8):
        """
        Initialize image formatter.
        
        Args:
            max_size_mb: Maximum image size in MB for optimization
        """
        self.max_size_mb = max_size_mb
        self.max_final_size_mb = 1.0  # MCP limit
    
    async def fetch_as_base64(self, image_url: str) -> Optional[Dict]:
        """
        Fetch image from URL and convert to base64 MCP content with size optimization.
        
        Args:
            image_url: URL of the image to fetch
            
        Returns:
            MCP image content dict or None if failed
        """
        if not image_url or not image_url.startswith(('http://', 'https://')):
            return None
            
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(image_url)
                response.raise_for_status()
                
                # Check content type
                content_type = response.headers.get('content-type', '').lower()
                if not content_type.startswith('image/'):
                    return None
                
                image_data = response.content
                original_size_mb = len(image_data) / (1024 * 1024)
                
                # If image is too large and PIL is available, resize it
                if original_size_mb > self.max_size_mb and Image:
                    try:
                        img = Image.open(BytesIO(image_data))
                        
                        # Calculate resize dimensions to stay under limit
                        scale_factor = (self.max_size_mb / original_size_mb) ** 0.5
                        new_width = int(img.width * scale_factor)
                        new_height = int(img.height * scale_factor)
                        
                        # Resize and optimize
                        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                        
                        # Convert to RGB if needed and save as JPEG for better compression
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        
                        buffer = BytesIO()
                        img.save(buffer, format='JPEG', quality=85, optimize=True)
                        image_data = buffer.getvalue()
                        content_type = 'image/jpeg'
                        
                    except Exception:
                        # If resizing fails, check if original is under final limit
                        if original_size_mb > self.max_final_size_mb:
                            return None
                
                # Final size check
                final_size_mb = len(image_data) / (1024 * 1024)
                if final_size_mb > self.max_final_size_mb:
                    return None
                
                # Encode to base64
                base64_data = base64.b64encode(image_data).decode('utf-8')
                
                return {
                    "type": "image",
                    "data": base64_data,
                    "mimeType": content_type
                }
                
        except Exception:
            # Silently fail - we'll just show text without image
            return None
    
    async def validate_url(self, image_url: str) -> bool:
        """
        Validate if image URL is accessible and is an actual image.
        
        Args:
            image_url: URL to validate
            
        Returns:
            True if valid image URL, False otherwise
        """
        if not image_url or not image_url.startswith(('http://', 'https://')):
            return False
            
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                # Use HEAD request to check without downloading full image
                response = await client.head(image_url)
                response.raise_for_status()
                
                content_type = response.headers.get('content-type', '').lower()
                return content_type.startswith('image/')
        except:
            return False