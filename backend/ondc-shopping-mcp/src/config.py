"""Configuration management for ONDC Shopping MCP Server"""

import os
import sys
from dataclasses import dataclass
from typing import Optional, Dict, Any
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Note: Logging configuration moved to individual modules to avoid stdout conflicts
# MCP servers must keep stdout clean for JSON-RPC communication

@dataclass
class APIConfig:
    """API configuration"""
    backend_endpoint: str
    wil_api_key: str
    timeout: int = 30
    max_retries: int = 3
    
    @property
    def default_headers(self) -> Dict[str, str]:
        """Default headers for API requests"""
        return {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "wil-api-key": self.wil_api_key
        }


@dataclass 
class VectorConfig:
    """Vector search configuration"""
    enabled: bool
    host: str = "localhost"
    port: int = 6333
    collection: str = "himira_products"
    embedding_model: str = "models/text-embedding-004"  # Must match ETL model
    vector_dimension: int = 768
    gemini_api_key: Optional[str] = None
    similarity_threshold: float = 0.3  # Lower threshold for better recall
    max_results: int = 20
    

@dataclass
class SessionConfig:
    """Session management configuration"""
    timeout_minutes: int = 30
    store_type: str = "memory"  # memory, file, redis
    store_path: str = "./session_store"
    

@dataclass
class PerformanceConfig:
    """Performance optimization configuration"""
    cache_enabled: bool = True
    cache_ttl_seconds: int = 300
    max_image_size_mb: float = 0.8
    concurrent_searches: int = 5
    request_timeout: int = 30
    

@dataclass
class PaymentConfig:
    """Payment Mock Configuration - FOR TESTING ONLY"""
    # Mock Mode Settings
    mock_mode: bool = True
    debug_logs: bool = True
    
    # Mock Values - Source: Himira Order Postman Collection
    mock_razorpay_payment_id: str = "pay_RFWPuAV50T2Qnj"  # MOCK: From Postman
    mock_payment_status: str = "PAID"  # MOCK: Simulated status
    
    # Mock ONDC Settlement Values - ONDC Protocol Compliant
    mock_settlement_basis: str = "delivery"  # MOCK: Standard ONDC value
    mock_settlement_window: str = "P1D"      # MOCK: ISO 8601 duration (1 day)
    mock_withholding_amount: str = "0.00"    # MOCK: String format amount
    
    # Feature Toggles
    enable_cod_payments: bool = False  # COD not supported by Himira backend
    payment_provider: str = "razorpay_mock"  # Mock provider identifier


@dataclass
class GuestConfig:
    """Guest user configuration for consistent guest journeys"""
    user_id: str = "guestUser"
    device_id: str = "d58dc5e2119ae5430b9321602618c878"
    
    
@dataclass
class SearchConfig:
    """Dynamic search configuration for intelligent result sizing"""
    # Default result limits
    min_results: int = 2
    max_results: int = 25
    default_limit: int = 10
    
    # Relevance thresholds
    default_relevance_threshold: float = 0.7
    min_relevance_threshold: float = 0.4
    max_relevance_threshold: float = 0.9
    
    # Feature toggles
    adaptive_sizing: bool = True
    query_analysis_enabled: bool = True
    relevance_filtering: bool = True
    context_aware: bool = True
    
    # Intent-based configurations can be overridden
    enable_intent_analysis: bool = True
    

@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    file: Optional[str] = None
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    

class Config:
    """Main configuration class"""
    
    def __init__(self):
        # API Configuration
        # Use Himira preprod backend as default
        default_endpoint = "https://hp-buyer-backend-preprod.himira.co.in"
        default_api_key = None  # Must be provided via environment variable
        
        self.api = APIConfig(
            backend_endpoint=os.getenv("BACKEND_ENDPOINT", default_endpoint),
            wil_api_key=os.getenv("WIL_API_KEY", default_api_key)
        )
        
        # Vector Search Configuration
        self.vector = VectorConfig(
            enabled=os.getenv("VECTOR_SEARCH_ENABLED", "false").lower() == "true",
            host=os.getenv("QDRANT_HOST", "localhost"),
            port=int(os.getenv("QDRANT_PORT", "6333")),
            collection=os.getenv("QDRANT_COLLECTION", "himira_products"),
            gemini_api_key=os.getenv("GEMINI_API_KEY"),
            similarity_threshold=float(os.getenv("VECTOR_SIMILARITY_THRESHOLD", "0.3")),
            max_results=int(os.getenv("VECTOR_MAX_RESULTS", "20"))
        )
        
        # Session Configuration
        self.session = SessionConfig(
            timeout_minutes=int(os.getenv("SESSION_TIMEOUT_MINUTES", "30")),
            store_type=os.getenv("SESSION_STORE", "file"),  # Default to file for persistence
            store_path=os.path.expanduser(os.getenv("SESSION_STORE_PATH", "~/.ondc-mcp/sessions"))
        )
        
        # Performance Configuration
        self.performance = PerformanceConfig(
            cache_enabled=os.getenv("CACHE_ENABLED", "true").lower() == "true",
            cache_ttl_seconds=int(os.getenv("CACHE_TTL_SECONDS", "300")),
            max_image_size_mb=float(os.getenv("MAX_IMAGE_SIZE_MB", "0.8")),
            concurrent_searches=int(os.getenv("CONCURRENT_SEARCHES", "5"))
        )
        
        # Dynamic Search Configuration
        self.search = SearchConfig(
            min_results=int(os.getenv("SEARCH_MIN_RESULTS", "2")),
            max_results=int(os.getenv("SEARCH_MAX_RESULTS", "25")),
            default_limit=int(os.getenv("SEARCH_DEFAULT_LIMIT", "10")),
            default_relevance_threshold=float(os.getenv("SEARCH_DEFAULT_RELEVANCE", "0.7")),
            min_relevance_threshold=float(os.getenv("SEARCH_MIN_RELEVANCE", "0.4")),
            max_relevance_threshold=float(os.getenv("SEARCH_MAX_RELEVANCE", "0.9")),
            adaptive_sizing=os.getenv("SEARCH_ADAPTIVE_SIZING", "true").lower() == "true",
            query_analysis_enabled=os.getenv("SEARCH_QUERY_ANALYSIS", "true").lower() == "true",
            relevance_filtering=os.getenv("SEARCH_RELEVANCE_FILTERING", "true").lower() == "true",
            context_aware=os.getenv("SEARCH_CONTEXT_AWARE", "true").lower() == "true",
            enable_intent_analysis=os.getenv("SEARCH_INTENT_ANALYSIS", "true").lower() == "true"
        )
        
        # Payment Mock Configuration - FOR TESTING ONLY
        self.payment = PaymentConfig(
            mock_mode=os.getenv("PAYMENT_MOCK_MODE", "true").lower() == "true",
            debug_logs=os.getenv("MCP_DEBUG_PAYMENT_LOGS", "true").lower() == "true",
            mock_razorpay_payment_id=os.getenv("MOCK_RAZORPAY_PAYMENT_ID", "pay_RFWPuAV50T2Qnj"),
            mock_payment_status=os.getenv("MOCK_PAYMENT_STATUS", "PAID"),
            mock_settlement_basis=os.getenv("MOCK_SETTLEMENT_BASIS", "delivery"),
            mock_settlement_window=os.getenv("MOCK_SETTLEMENT_WINDOW", "P1D"),
            mock_withholding_amount=os.getenv("MOCK_WITHHOLDING_AMOUNT", "0.00"),
            enable_cod_payments=os.getenv("ENABLE_COD_PAYMENTS", "false").lower() == "true",
            payment_provider=os.getenv("MCP_PAYMENT_PROVIDER", "razorpay_mock")
        )
        
        # Guest User Configuration
        self.guest = GuestConfig(
            user_id=os.getenv("GUEST_USER_ID", "guestUser"),
            device_id=os.getenv("GUEST_DEVICE_ID", "d58dc5e2119ae5430b9321602618c878")
        )
        
        # Logging Configuration
        # Always log to file for better observability
        self.logging = LoggingConfig(
            level=os.getenv("LOG_LEVEL", "INFO"),
            file=os.getenv("LOG_FILE", "/app/logs/mcp_operations.log")
        )
        
        # Apply logging configuration
        self._configure_logging()
        
    def _configure_logging(self):
        """Configure logging based on settings"""
        log_level = getattr(logging, self.logging.level.upper(), logging.INFO)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Console handler - MUST use stderr for MCP servers
        # stdout is reserved for JSON-RPC communication only
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(logging.Formatter(self.logging.format))
        root_logger.addHandler(console_handler)
        
        # File handler if specified
        if self.logging.file:
            try:
                # Try to create directory if it doesn't exist
                log_dir = os.path.dirname(self.logging.file)
                if log_dir and not os.path.exists(log_dir):
                    os.makedirs(log_dir, exist_ok=True)
                
                file_handler = logging.FileHandler(self.logging.file)
                file_handler.setLevel(log_level)
                file_handler.setFormatter(logging.Formatter(self.logging.format))
                root_logger.addHandler(file_handler)
            except (OSError, PermissionError) as e:
                # If we can't create the log file (e.g., in Docker), just use console
                logging.warning(f"Could not create log file {self.logging.file}: {e}")
    
    def validate(self) -> bool:
        """Validate configuration"""
        errors = []
        
        # Validate API config
        if not self.api.backend_endpoint:
            errors.append("BACKEND_ENDPOINT is required")
        if not self.api.wil_api_key:
            errors.append("WIL_API_KEY is required")
        
        # Validate vector config if enabled
        if self.vector.enabled:
            if not self.vector.gemini_api_key:
                errors.append("GEMINI_API_KEY is required when vector search is enabled")
        
        if errors:
            for error in errors:
                logging.error(f"Configuration error: {error}")
            return False
            
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "api": {
                "backend_endpoint": self.api.backend_endpoint,
                "timeout": self.api.timeout,
                "max_retries": self.api.max_retries
            },
            "vector": {
                "enabled": self.vector.enabled,
                "host": self.vector.host,
                "port": self.vector.port,
                "collection": self.vector.collection,
                "embedding_model": self.vector.embedding_model,
                "vector_dimension": self.vector.vector_dimension
            },
            "session": {
                "timeout_minutes": self.session.timeout_minutes,
                "store_type": self.session.store_type
            },
            "performance": {
                "cache_enabled": self.performance.cache_enabled,
                "cache_ttl_seconds": self.performance.cache_ttl_seconds,
                "max_image_size_mb": self.performance.max_image_size_mb
            },
            "search": {
                "min_results": self.search.min_results,
                "max_results": self.search.max_results,
                "default_limit": self.search.default_limit,
                "default_relevance_threshold": self.search.default_relevance_threshold,
                "adaptive_sizing": self.search.adaptive_sizing,
                "query_analysis_enabled": self.search.query_analysis_enabled,
                "relevance_filtering": self.search.relevance_filtering,
                "context_aware": self.search.context_aware
            },
            "payment": {
                "mock_mode": self.payment.mock_mode,
                "debug_logs": self.payment.debug_logs,
                "mock_payment_id": self.payment.mock_razorpay_payment_id,
                "payment_provider": self.payment.payment_provider,
                "enable_cod": self.payment.enable_cod_payments
            },
            "logging": {
                "level": self.logging.level,
                "file": self.logging.file
            }
        }


# Global configuration instance
config = Config()