#!/usr/bin/env python3
"""
Generate mcp_agent.config.yaml from environment variables
This script reads from environment and creates the runtime config file
"""

import os
import yaml
from pathlib import Path

def generate_config():
    """Generate MCP agent config from environment variables"""
    
    # Get environment variables with defaults for non-sensitive values
    backend_endpoint = os.getenv('BACKEND_ENDPOINT', 'https://hp-buyer-backend-preprod.himira.co.in/clientApis')
    wil_api_key = os.getenv('WIL_API_KEY')
    gemini_api_key = os.getenv('GEMINI_API_KEY')
    
    # Validate required keys
    if not wil_api_key:
        raise ValueError("WIL_API_KEY environment variable is required")
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY environment variable is required")
    
    # Build config structure
    config = {
        'logger': {
            'type': 'console',
            'level': 'info'
        },
        'mcp': {
            'servers': {
                'ondc-shopping': {
                    'command': 'python',
                    'args': ['/app/ondc-shopping-mcp/run_mcp_server.py'],
                    'env': {
                        # Transport
                        'MCP_TRANSPORT': 'stdio',
                        'PYTHONUNBUFFERED': '1',
                        
                        # Backend configuration from environment
                        'BACKEND_ENDPOINT': backend_endpoint,
                        'WIL_API_KEY': wil_api_key,
                        'GEMINI_API_KEY': gemini_api_key,
                        
                        # Vector Search
                        'VECTOR_SEARCH_ENABLED': os.getenv('VECTOR_SEARCH_ENABLED', 'true'),
                        'QDRANT_HOST': os.getenv('QDRANT_HOST', 'qdrant'),
                        'QDRANT_PORT': os.getenv('QDRANT_PORT', '6333'),
                        'QDRANT_COLLECTION': os.getenv('QDRANT_COLLECTION', 'himira_products'),
                        'VECTOR_SIMILARITY_THRESHOLD': '0.3',
                        'VECTOR_MAX_RESULTS': '20',
                        
                        # Session Management
                        'SESSION_STORE': 'file',
                        'SESSION_STORE_PATH': '/app/sessions',
                        'SESSION_TIMEOUT_MINUTES': '30',
                        
                        # Performance & Caching
                        'CACHE_ENABLED': 'true',
                        'CACHE_TTL_SECONDS': '300',
                        'CONCURRENT_SEARCHES': '5',
                        
                        # Logging
                        'LOG_LEVEL': os.getenv('LOG_LEVEL', 'INFO'),
                        'LOG_FILE': '/app/logs/mcp_operations.log',
                        'DEBUG_CURL_LOGGING': 'true',
                        'MCP_DEBUG_PAYMENT_LOGS': 'false'
                    }
                }
            }
        }
    }
    
    # Write config to file
    config_path = Path('/app/mcp_agent.config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"✅ Generated config at {config_path}")
    print(f"   Backend: {backend_endpoint}")
    print(f"   WIL API Key: {'***' + wil_api_key[-4:] if wil_api_key else 'NOT SET'}")
    print(f"   Gemini API Key: {'***' + gemini_api_key[-4:] if gemini_api_key else 'NOT SET'}")

if __name__ == '__main__':
    try:
        generate_config()
    except Exception as e:
        print(f"❌ Failed to generate config: {e}")
        exit(1)