#!/usr/bin/env python3
"""
Environment validation script for ONDC MCP Backend
Ensures all required environment variables are set before starting services
"""

import os
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_env():
    """Validate required environment variables"""
    
    required_vars = {
        # API Keys
        "GEMINI_API_KEY": "Google Gemini API key for AI embeddings",
        "WIL_API_KEY": "Backend API authentication key",
        
        # Backend Configuration
        "BACKEND_ENDPOINT": "ONDC backend API URL",
        
        # Database Configuration
        "QDRANT_HOST": "Qdrant vector database host",
        "QDRANT_PORT": "Qdrant vector database port",
    }
    
    optional_vars = {
        "FIREBASE_API_KEY": "Firebase API key for SMS OTP (optional)",
        "LOG_LEVEL": "Logging level (default: INFO)",
        "SESSION_TTL_HOURS": "Session timeout in hours (default: 24)",
    }
    
    errors = []
    warnings = []
    
    # Check required variables
    for var, description in required_vars.items():
        value = os.getenv(var)
        if not value:
            errors.append(f"❌ {var}: {description} - NOT SET")
        else:
            # Mask sensitive data in output
            if "KEY" in var or "URI" in var:
                masked_value = value[:8] + "..." if len(value) > 8 else "***"
                logger.info(f"✅ {var}: {masked_value}")
            else:
                logger.info(f"✅ {var}: {value}")
    
    # Check optional variables
    for var, description in optional_vars.items():
        value = os.getenv(var)
        if not value:
            warnings.append(f"⚠️  {var}: {description} - NOT SET (using default)")
        else:
            logger.info(f"✅ {var}: Set")
    
    # Report results
    if errors:
        logger.error("\n=== VALIDATION FAILED ===")
        for error in errors:
            logger.error(error)
        logger.error("\nPlease set the required environment variables in your .env file")
        logger.error("Copy .env.example to .env and fill in the values")
        return False
    
    if warnings:
        logger.warning("\n=== WARNINGS ===")
        for warning in warnings:
            logger.warning(warning)
    
    # Additional validation
    logger.info("\n=== ADDITIONAL CHECKS ===")
    
    # Check Qdrant port is numeric
    qdrant_port = os.getenv("QDRANT_PORT", "")
    if qdrant_port and not qdrant_port.isdigit():
        logger.error("❌ QDRANT_PORT must be a number")
        return False
    
    # Check backend endpoint is a valid URL
    backend_endpoint = os.getenv("BACKEND_ENDPOINT", "")
    if backend_endpoint and not (backend_endpoint.startswith("http://") or backend_endpoint.startswith("https://")):
        logger.error("❌ BACKEND_ENDPOINT must start with 'http://' or 'https://'")
        return False
    
    logger.info("\n✅ Environment validation successful!")
    return True

def main():
    """Main function"""
    if not validate_env():
        sys.exit(1)
    
    logger.info("\n🚀 Environment is ready for ONDC MCP Backend")
    logger.info("You can now start the services with: docker-compose up")

if __name__ == "__main__":
    main()