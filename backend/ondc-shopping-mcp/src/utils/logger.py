"""Logging utilities for MCP Server"""

import logging
import sys
from typing import Optional


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Get a configured logger instance
    
    Args:
        name: Logger name (usually __name__)
        level: Optional log level override
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Set level if provided
    if level:
        log_level = getattr(logging, level.upper(), logging.INFO)
        logger.setLevel(log_level)
    
    return logger


def setup_mcp_logging(debug: bool = False):
    """
    Setup logging appropriate for MCP server
    
    MCP servers should only output JSON-RPC messages to stdout,
    so we redirect all logging to stderr.
    
    Args:
        debug: Enable debug logging
    """
    # Configure root logger - respect LOG_LEVEL environment variable
    import os
    log_level_str = os.getenv('LOG_LEVEL', 'INFO').upper()
    log_level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    log_level = log_level_map.get(log_level_str, logging.INFO)
    
    # Override with debug flag if provided
    if debug:
        log_level = logging.DEBUG
    
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear any existing handlers
    root_logger.handlers.clear()
    
    # Create stderr handler (so logs don't interfere with JSON-RPC on stdout)
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(log_level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    stderr_handler.setFormatter(formatter)
    
    # Add handler
    root_logger.addHandler(stderr_handler)
    
    # Suppress noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    
    return root_logger