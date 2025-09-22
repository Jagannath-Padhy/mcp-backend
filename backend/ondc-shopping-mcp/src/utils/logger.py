"""Logging utilities for MCP Server"""

import logging
import sys
import json
import os
import time
from datetime import datetime
from typing import Optional, Dict, Any, List
from functools import wraps


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


class MCPOperationsLogger:
    """Enhanced logger for MCP operations with structured JSON logging"""
    
    def __init__(self):
        self.debug_level = os.getenv('MCP_DEBUG_LEVEL', 'BASIC').upper()
        self.max_log_size = int(os.getenv('MCP_LOG_MAX_SIZE', '5000'))
        self.include_vectors = os.getenv('MCP_LOG_INCLUDE_VECTORS', 'false').lower() == 'true'
        self.log_file = '/app/logs/mcp_operations.log'
        
        # Create logger for MCP operations
        self.logger = logging.getLogger('mcp_operations')
        self.logger.setLevel(logging.DEBUG)
        
        # Setup file handler if not already exists
        if not self.logger.handlers:
            self._setup_file_handler()
    
    def _setup_file_handler(self):
        """Setup file handler for MCP operations logging"""
        try:
            # Ensure log directory exists
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
            
            # Create file handler
            file_handler = logging.FileHandler(self.log_file)
            file_handler.setLevel(logging.DEBUG)
            
            # Create formatter for structured logging
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            
            self.logger.addHandler(file_handler)
            self.logger.info(f"MCP Operations logger initialized with debug_level={self.debug_level}")
            
        except Exception as e:
            # Fallback to stderr if file logging fails
            stderr_handler = logging.StreamHandler(sys.stderr)
            stderr_handler.setFormatter(formatter)
            self.logger.addHandler(stderr_handler)
            self.logger.error(f"Failed to setup file logging, using stderr: {e}")
    
    def _truncate_data(self, data: Any) -> Any:
        """Truncate large data structures for logging"""
        if self.debug_level == 'RAW':
            return data
            
        json_str = json.dumps(data, default=str)
        if len(json_str) <= self.max_log_size:
            return data
        
        # Truncate and add indicator
        truncated_str = json_str[:self.max_log_size] + '...[TRUNCATED]'
        return {"_truncated": True, "_size": len(json_str), "_data": truncated_str}
    
    def _filter_vector_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Remove vector data from logs if not included"""
        if self.include_vectors:
            return data
            
        filtered_data = data.copy()
        
        # Remove common vector fields
        vector_fields = ['vector', 'vectors', 'embedding', 'embeddings', 'query_vector', 'search_results']
        for field in vector_fields:
            if field in filtered_data:
                if isinstance(filtered_data[field], list):
                    filtered_data[field] = f"[{len(filtered_data[field])} items hidden]"
                else:
                    filtered_data[field] = "[hidden]"
        
        # Filter vector data from nested structures
        if 'products' in filtered_data and isinstance(filtered_data['products'], list):
            for product in filtered_data['products']:
                if isinstance(product, dict):
                    for vf in vector_fields:
                        if vf in product:
                            product[vf] = "[hidden]"
        
        return filtered_data
    
    def log_tool_request(self, tool_name: str, session_id: str, request_data: Dict[str, Any]):
        """Log MCP tool request"""
        if self.debug_level == 'BASIC':
            return
            
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "event_type": "mcp_request",
            "tool": tool_name,
            "session_id": session_id[:16] + "..." if len(session_id) > 16 else session_id,
            "request": self._truncate_data(self._filter_vector_data(request_data))
        }
        
        self.logger.info(f"[REQUEST] {json.dumps(log_entry, separators=(',', ':'))}")
    
    def log_tool_response(self, tool_name: str, session_id: str, response_data: Dict[str, Any], 
                         execution_time_ms: float, backend_calls: List[str] = None, status: str = "success"):
        """Log MCP tool response with execution metrics"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "event_type": "mcp_response", 
            "tool": tool_name,
            "session_id": session_id[:16] + "..." if len(session_id) > 16 else session_id,
            "execution_time_ms": round(execution_time_ms, 2),
            "status": status,
            "backend_calls": backend_calls or [],
            "response": self._truncate_data(self._filter_vector_data(response_data)) if self.debug_level == 'FULL' else {"success": response_data.get("success", True), "message": response_data.get("message", "")[:200]}
        }
        
        self.logger.info(f"[RESPONSE] {json.dumps(log_entry, separators=(',', ':'))}")
    
    def log_tool_error(self, tool_name: str, session_id: str, error: Exception, execution_time_ms: float):
        """Log MCP tool error"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "event_type": "mcp_error",
            "tool": tool_name, 
            "session_id": session_id[:16] + "..." if len(session_id) > 16 else session_id,
            "execution_time_ms": round(execution_time_ms, 2),
            "status": "error",
            "error": {
                "type": type(error).__name__,
                "message": str(error)[:500]
            }
        }
        
        self.logger.error(f"[ERROR] {json.dumps(log_entry, separators=(',', ':'))}")


# Global instance
_mcp_operations_logger = None

def get_mcp_operations_logger() -> MCPOperationsLogger:
    """Get global MCP operations logger instance"""
    global _mcp_operations_logger
    if _mcp_operations_logger is None:
        _mcp_operations_logger = MCPOperationsLogger()
    return _mcp_operations_logger