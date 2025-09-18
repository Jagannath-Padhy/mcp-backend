"""
MCP Protocol Error Handling

Implements all standard JSON-RPC and MCP-specific error codes.
Provides robust error handling for the protocol layer.
"""

from typing import Optional, Any, Dict
from enum import IntEnum


class ErrorCode(IntEnum):
    """Standard JSON-RPC 2.0 and MCP error codes"""
    
    # JSON-RPC 2.0 Standard Errors
    PARSE_ERROR = -32700          # Invalid JSON was received
    INVALID_REQUEST = -32600       # The JSON sent is not a valid Request object
    METHOD_NOT_FOUND = -32601      # The method does not exist or is not available
    INVALID_PARAMS = -32602        # Invalid method parameter(s)
    INTERNAL_ERROR = -32603        # Internal JSON-RPC error
    
    # JSON-RPC 2.0 Implementation-defined errors (-32000 to -32099)
    RESOURCE_NOT_FOUND = -32002   # MCP: Requested resource doesn't exist
    
    # MCP-specific errors (-32800 to -32899)
    REQUEST_CANCELLED = -32800    # The request was cancelled
    CONTENT_TOO_LARGE = -32801    # The content is too large


class MCPError(Exception):
    """Base class for all MCP protocol errors"""
    
    def __init__(
        self,
        code: int,
        message: str,
        data: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize MCP error
        
        Args:
            code: Error code from ErrorCode enum
            message: Human-readable error message
            data: Optional additional error data
        """
        self.code = code
        self.message = message
        self.data = data or {}
        super().__init__(message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to JSON-RPC error format"""
        error_dict = {
            "code": self.code,
            "message": self.message
        }
        if self.data:
            error_dict["data"] = self.data
        return error_dict
    
    def to_response(self, request_id: Optional[Any] = None) -> Dict[str, Any]:
        """Create complete JSON-RPC error response"""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": self.to_dict()
        }


class ParseError(MCPError):
    """Invalid JSON was received"""
    
    def __init__(self, message: str = "Parse error", data: Optional[Dict] = None):
        super().__init__(
            ErrorCode.PARSE_ERROR,
            f"Parse error: {message}",
            data
        )


class InvalidRequest(MCPError):
    """The JSON sent is not a valid Request object"""
    
    def __init__(self, message: str = "Invalid request", data: Optional[Dict] = None):
        super().__init__(
            ErrorCode.INVALID_REQUEST,
            f"Invalid request: {message}",
            data
        )


class MethodNotFound(MCPError):
    """The method does not exist or is not available"""
    
    def __init__(self, method: str, data: Optional[Dict] = None):
        super().__init__(
            ErrorCode.METHOD_NOT_FOUND,
            f"Method not found: {method}",
            data or {"method": method}
        )


class InvalidParams(MCPError):
    """Invalid method parameter(s)"""
    
    def __init__(self, message: str = "Invalid parameters", data: Optional[Dict] = None):
        super().__init__(
            ErrorCode.INVALID_PARAMS,
            f"Invalid params: {message}",
            data
        )


class InternalError(MCPError):
    """Internal JSON-RPC error"""
    
    def __init__(self, message: str = "Internal error", data: Optional[Dict] = None):
        super().__init__(
            ErrorCode.INTERNAL_ERROR,
            f"Internal error: {message}",
            data
        )


class ResourceNotFound(MCPError):
    """Requested resource doesn't exist"""
    
    def __init__(self, resource: str, data: Optional[Dict] = None):
        super().__init__(
            ErrorCode.RESOURCE_NOT_FOUND,
            f"Resource not found: {resource}",
            data or {"resource": resource}
        )


class RequestCancelled(MCPError):
    """The request was cancelled"""
    
    def __init__(self, request_id: Any, data: Optional[Dict] = None):
        super().__init__(
            ErrorCode.REQUEST_CANCELLED,
            f"Request cancelled: {request_id}",
            data or {"request_id": request_id}
        )


class ContentTooLarge(MCPError):
    """The content is too large"""
    
    def __init__(self, size: int, max_size: int, data: Optional[Dict] = None):
        super().__init__(
            ErrorCode.CONTENT_TOO_LARGE,
            f"Content too large: {size} bytes (max: {max_size})",
            data or {"size": size, "max_size": max_size}
        )


class ErrorHandler:
    """Utility class for handling and formatting errors"""
    
    @staticmethod
    def handle_exception(e: Exception, request_id: Optional[Any] = None) -> Dict[str, Any]:
        """
        Convert any exception to proper MCP error response
        
        Args:
            e: Exception to handle
            request_id: Request ID for the response
            
        Returns:
            JSON-RPC error response dict
        """
        if isinstance(e, MCPError):
            # Already an MCP error, just format it
            return e.to_response(request_id)
        
        # Convert generic exceptions to internal errors
        internal_error = InternalError(
            str(e),
            {"exception_type": type(e).__name__}
        )
        return internal_error.to_response(request_id)
    
    @staticmethod
    def validate_request(request: Dict[str, Any]) -> None:
        """
        Validate JSON-RPC request format
        
        Args:
            request: Request dictionary to validate
            
        Raises:
            InvalidRequest: If request is invalid
        """
        # Check required fields
        if not isinstance(request, dict):
            raise InvalidRequest("Request must be a JSON object")
        
        if request.get("jsonrpc") != "2.0":
            raise InvalidRequest("Missing or invalid jsonrpc version")
        
        if "method" not in request:
            raise InvalidRequest("Missing method field")
        
        method = request["method"]
        if not isinstance(method, str) or not method:
            raise InvalidRequest("Method must be a non-empty string")
        
        # ID is optional for notifications
        if "id" in request:
            request_id = request["id"]
            if not isinstance(request_id, (str, int, type(None))):
                raise InvalidRequest("ID must be string, number, or null")
        
        # Params is optional but must be object or array if present
        if "params" in request:
            params = request["params"]
            if not isinstance(params, (dict, list)):
                raise InvalidRequest("Params must be object or array")
    
    @staticmethod
    def validate_params(params: Dict[str, Any], required: List[str]) -> None:
        """
        Validate that required parameters are present
        
        Args:
            params: Parameters dictionary
            required: List of required parameter names
            
        Raises:
            InvalidParams: If required parameters are missing
        """
        missing = [name for name in required if name not in params]
        if missing:
            raise InvalidParams(
                f"Missing required parameters: {', '.join(missing)}",
                {"missing": missing}
            )