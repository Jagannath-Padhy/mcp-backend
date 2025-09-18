# ONDC Shopping MCP - Developer Guide

## Overview

This guide provides comprehensive information for developers contributing to the ONDC Shopping MCP system, including development setup, coding standards, architecture patterns, and extension guidelines.

## Table of Contents

1. [Development Environment Setup](#development-environment-setup)
2. [Project Structure & Organization](#project-structure--organization)
3. [Coding Standards & Best Practices](#coding-standards--best-practices)
4. [Adding New MCP Tools](#adding-new-mcp-tools)
5. [Service Layer Development](#service-layer-development)
6. [Testing Framework](#testing-framework)
7. [Debugging & Troubleshooting](#debugging--troubleshooting)
8. [Performance Profiling](#performance-profiling)
9. [Contributing Guidelines](#contributing-guidelines)
10. [Release Process](#release-process)

---

## Development Environment Setup

### Prerequisites

```bash
# Required tools
- Python 3.11+
- Docker & Docker Compose
- Git
- VS Code (recommended) or PyCharm

# Optional but recommended
- Desktop Client Pro (for testing)
- MCP Inspector (free testing alternative)
- Postman (for API testing)
```

### Local Development Setup

```bash
# 1. Clone repository
git clone <repository-url>
cd ondc-shopping-mcp/

# 2. Create Python virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate    # Windows

# 3. Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# 4. Setup pre-commit hooks
pre-commit install

# 5. Copy environment configuration
cp .env.example .env.dev
# Edit .env.dev with development settings

# 6. Start supporting services (Qdrant for vector search)
docker-compose -f docker-compose.dev.yml up -d qdrant

# 7. Run MCP server in development mode
python run_mcp_server.py --env dev --debug
```

### Development Dependencies

```txt
# requirements-dev.txt
pytest>=7.0.0
pytest-asyncio>=0.21.0
pytest-cov>=4.0.0
black>=23.0.0
isort>=5.12.0
flake8>=6.0.0
mypy>=1.0.0
pre-commit>=3.0.0
bandit>=1.7.0
safety>=2.3.0
```

### IDE Configuration

#### VS Code Settings
```json
// .vscode/settings.json
{
  "python.defaultInterpreterPath": "./venv/bin/python",
  "python.formatting.provider": "black",
  "python.linting.enabled": true,
  "python.linting.flake8Enabled": true,
  "python.linting.mypyEnabled": true,
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": ["tests/"],
  "files.exclude": {
    "**/__pycache__": true,
    "**/*.pyc": true,
    ".mypy_cache": true
  },
  "python.formatting.blackArgs": ["--line-length=88"],
  "isort.args": ["--profile", "black"]
}
```

#### PyCharm Configuration
```python
# Configure in Settings > Tools > Python Integrated Tools
# - Default test runner: pytest
# - Type checker: mypy
# - Code style: Black formatter
# - Import sorter: isort with Black profile
```

---

## Project Structure & Organization

### Directory Layout

```
ondc-shopping-mcp/
 src/                          # Source code
    adapters/                 # MCP adapter modules (6 modules)
       __init__.py           # Package documentation
       utils.py              # Shared utilities
       cart.py               # Cart operations (6 tools)
       search.py             # Search operations (3 tools)
       checkout.py           # ONDC checkout (4 tools)
       auth.py               # Authentication (1 tool)
       session.py            # Session management (2 tools)
       orders.py             # Order management (4 tools)
    services/                 # Business logic layer
       session_service.py    # Session lifecycle
       search_service.py     # Product search & vector
       cart_service.py       # Cart operations
       checkout_service.py   # Consolidated checkout
       order_service.py      # ONDC order flow
       payment_service.py    # Payment processing
    data_models/              # Data structures
       ondc_models.py        # ONDC protocol models
       biap_context_factory.py # BIAP context generation
    utils/                    # Shared utilities
       logger.py             # Logging configuration
       field_mapper.py       # Data transformation
       schema_generators.py  # MCP schema utilities
    mcp_adapters.py           # Main adapter imports
    mcp_server.py             # MCP server implementation
    config.py                 # Configuration management
    buyer_backend_client.py   # Himira backend client
 tests/                        # Test suite
    unit/                     # Unit tests
    integration/              # Integration tests
    performance/              # Load tests
    fixtures/                 # Test data
 docs/                         # Documentation
 scripts/                      # Utility scripts
 docker-compose.*.yml          # Docker configurations
 requirements*.txt             # Python dependencies
 run_mcp_server.py             # Server entry point
```

### Architecture Layers

```python
# Layer responsibilities and dependencies

1. MCP Adapter Layer (src/adapters/)
   - Purpose: MCP protocol interface
   - Dependencies: Service layer, utils
   - Pattern: Thin adapters wrapping service calls

2. Service Layer (src/services/)
   - Purpose: Business logic implementation
   - Dependencies: Data models, backend client, utils
   - Pattern: Single responsibility services

3. Data Model Layer (src/data_models/)
   - Purpose: ONDC/BIAP data structures
   - Dependencies: None (pure data models)
   - Pattern: Immutable data classes

4. Utility Layer (src/utils/)
   - Purpose: Shared functionality
   - Dependencies: None or minimal
   - Pattern: Stateless utility functions

5. Integration Layer (buyer_backend_client.py)
   - Purpose: External API communication
   - Dependencies: Config, utils
   - Pattern: Client facade with retry logic
```

---

## Coding Standards & Best Practices

### Python Style Guide

```python
# Follow PEP 8 with Black formatter configuration
# Line length: 88 characters
# String quotes: Double quotes preferred
# Import organization: isort with Black profile

# Example of proper code structure
from typing import Dict, List, Optional, Any
import asyncio
import logging

from ..utils.logger import get_logger
from ..services.base_service import BaseService

logger = get_logger(__name__)


class ExampleService(BaseService):
    """Example service following coding standards.
    
    This service demonstrates proper:
    - Type hints
    - Docstring format
    - Error handling
    - Async patterns
    """
    
    def __init__(self, config: Config):
        super().__init__(config)
        self.cache: Dict[str, Any] = {}
    
    async def process_data(self, 
                          data: Dict[str, Any], 
                          options: Optional[List[str]] = None) -> Dict[str, Any]:
        """Process data with optional configuration.
        
        Args:
            data: Input data dictionary
            options: Optional processing options
            
        Returns:
            Processed data dictionary
            
        Raises:
            ValueError: If data is invalid
            ProcessingError: If processing fails
        """
        if not data:
            raise ValueError("Data cannot be empty")
        
        try:
            result = await self._internal_process(data, options or [])
            logger.info(f"Successfully processed data: {len(data)} items")
            return result
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            raise ProcessingError(f"Failed to process data: {e}") from e
    
    async def _internal_process(self, data: Dict, options: List[str]) -> Dict:
        """Internal processing logic."""
        # Implementation details
        pass
```

### Async Programming Standards

```python
# Async best practices

# 1. Always use async/await for I/O operations
async def fetch_data(url: str) -> Dict:
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()

# 2. Use asyncio.gather for parallel operations
async def fetch_multiple(urls: List[str]) -> List[Dict]:
    tasks = [fetch_data(url) for url in urls]
    return await asyncio.gather(*tasks, return_exceptions=True)

# 3. Proper exception handling with context
async def safe_operation():
    try:
        result = await risky_operation()
        return result
    except SpecificError as e:
        logger.warning(f"Expected error occurred: {e}")
        return default_result()
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise

# 4. Use asyncio.wait_for for timeouts
async def operation_with_timeout():
    try:
        return await asyncio.wait_for(
            slow_operation(), 
            timeout=30.0
        )
    except asyncio.TimeoutError:
        raise OperationTimeoutError("Operation timed out after 30 seconds")
```

### Error Handling Patterns

```python
# Standardized error handling

from enum import Enum
from typing import Optional

class ErrorType(Enum):
    VALIDATION_ERROR = "validation_error"
    AUTHENTICATION_ERROR = "authentication_error"
    BACKEND_ERROR = "backend_error"
    RATE_LIMIT_ERROR = "rate_limit_error"

class MCPError(Exception):
    """Base exception for MCP operations."""
    
    def __init__(self, 
                 error_type: ErrorType, 
                 message: str,
                 session_id: Optional[str] = None,
                 details: Optional[Dict] = None):
        self.error_type = error_type
        self.message = message
        self.session_id = session_id
        self.details = details or {}
        super().__init__(message)

# Usage in services
async def service_operation(data: Dict) -> Dict:
    try:
        # Validate input
        if not validate_data(data):
            raise MCPError(
                ErrorType.VALIDATION_ERROR,
                "Invalid input data format",
                details={'required_fields': ['id', 'name']}
            )
        
        # Perform operation
        result = await backend_operation(data)
        return result
        
    except MCPError:
        # Re-raise MCP errors as-is
        raise
    except ConnectionError as e:
        # Convert to standardized error
        raise MCPError(
            ErrorType.BACKEND_ERROR,
            "Backend service unavailable",
            details={'original_error': str(e)}
        ) from e
```

### Logging Standards

```python
# Structured logging approach

import logging
import json
from typing import Any, Dict

class StructuredLogger:
    """Structured logger for consistent log formatting."""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
    
    def info(self, message: str, **kwargs):
        self._log(logging.INFO, message, kwargs)
    
    def error(self, message: str, **kwargs):
        self._log(logging.ERROR, message, kwargs)
    
    def _log(self, level: int, message: str, extra: Dict[str, Any]):
        log_data = {
            'message': message,
            'timestamp': time.time(),
            **extra
        }
        self.logger.log(level, json.dumps(log_data))

# Usage patterns
logger = StructuredLogger(__name__)

async def example_operation(session_id: str, data: Dict):
    logger.info(
        "Starting operation",
        session_id=session_id,
        operation="example_operation",
        data_size=len(data)
    )
    
    try:
        result = await process_data(data)
        logger.info(
            "Operation completed successfully",
            session_id=session_id,
            result_size=len(result),
            execution_time_ms=timer.elapsed()
        )
        return result
    except Exception as e:
        logger.error(
            "Operation failed",
            session_id=session_id,
            error_type=type(e).__name__,
            error_message=str(e)
        )
        raise
```

---

## Adding New MCP Tools

### Step-by-Step Process

#### 1. Define the Tool Function

```python
# src/adapters/new_module.py (if creating new module)
# OR add to existing module in src/adapters/

from typing import Dict, Any, Optional
from .utils import (
    get_persistent_session,
    save_persistent_session, 
    extract_session_id,
    format_mcp_response,
    get_services
)
from ..utils.logger import get_logger

logger = get_logger(__name__)

# Get services
services = get_services()
new_service = services['new_service']  # Add to services if needed


async def new_tool_name(
    session: Optional[Any] = None,
    required_param: str = None,
    optional_param: Optional[int] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    MCP adapter for new functionality.
    
    Args:
        session: Session object or ID
        required_param: Description of required parameter
        optional_param: Description of optional parameter
        
    Returns:
        Standardized MCP response dictionary
        
    Raises:
        MCPError: On validation or processing errors
    """
    try:
        # Get enhanced session with conversation tracking
        session_id = extract_session_id(session)
        session_obj, conversation_manager = get_persistent_session(
            session_id, 
            tool_name="new_tool_name", 
            **kwargs
        )
        
        # Validate required parameters
        if not required_param:
            return format_mcp_response(
                False,
                " Required parameter 'required_param' is missing.",
                session_obj.session_id,
                required_fields=["required_param"]
            )
        
        # Business logic using service layer
        result = await new_service.perform_operation(
            session_obj, 
            required_param, 
            optional_param
        )
        
        # Save enhanced session with conversation tracking
        save_persistent_session(session_obj, conversation_manager)
        
        # Return formatted response
        return format_mcp_response(
            result['success'],
            result['message'],
            session_obj.session_id,
            **result.get('data', {})
        )
        
    except Exception as e:
        logger.error(f"Failed to execute new_tool_name: {e}")
        return format_mcp_response(
            False,
            f' Failed to execute operation: {str(e)}',
            extract_session_id(session) or 'unknown'
        )
```

#### 2. Add to Main Adapters File

```python
# src/mcp_adapters.py

# Add import for new tool
from .adapters.new_module import (
    new_tool_name
)

# Add to __all__ list
__all__ = [
    # ... existing tools ...
    'new_tool_name',
]
```

#### 3. Register in MCP Server

```python
# src/mcp_server.py

# Import new tool
from .mcp_adapters import (
    # ... existing imports ...
    new_tool_name
)

class MCPServer:
    def __init__(self):
        # ... existing initialization ...
        self._register_all_tools()
    
    def _register_all_tools(self):
        # ... existing tool registrations ...
        
        # Register new tool
        self._register_tool(
            "new_tool_name",
            "Description of what the new tool does",
            {
                "required_param": {
                    "type": "string", 
                    "required": True,
                    "description": "Description of required parameter"
                },
                "optional_param": {
                    "type": "number", 
                    "required": False,
                    "description": "Description of optional parameter"
                }
            },
            new_tool_name
        )
```

#### 4. Add Service Layer Support (if needed)

```python
# src/services/new_service.py

from typing import Dict, Any
from .base_service import BaseService
from ..utils.logger import get_logger

logger = get_logger(__name__)


class NewService(BaseService):
    """Service for new functionality."""
    
    def __init__(self, config, backend_client):
        super().__init__(config)
        self.backend_client = backend_client
    
    async def perform_operation(self, 
                               session_obj: Any, 
                               required_param: str,
                               optional_param: int = None) -> Dict[str, Any]:
        """Perform the new operation."""
        try:
            # Validate inputs
            if not self._validate_input(required_param):
                return {
                    'success': False,
                    'message': 'Invalid input parameters'
                }
            
            # Perform business logic
            result = await self._execute_operation(
                session_obj,
                required_param,
                optional_param
            )
            
            return {
                'success': True,
                'message': 'Operation completed successfully',
                'data': result
            }
            
        except Exception as e:
            logger.error(f"NewService operation failed: {e}")
            return {
                'success': False,
                'message': f'Operation failed: {str(e)}'
            }
    
    def _validate_input(self, param: str) -> bool:
        """Validate input parameters."""
        return bool(param and param.strip())
    
    async def _execute_operation(self, session_obj, param1, param2):
        """Execute the core operation logic."""
        # Implementation details
        pass

# Add to service factory
# src/services/__init__.py
def get_new_service():
    from .new_service import NewService
    return NewService(config, get_backend_client())
```

#### 5. Add Tests

```python
# tests/unit/test_new_tool.py

import pytest
from unittest.mock import AsyncMock, patch
from src.mcp_adapters import new_tool_name


@pytest.fixture
def mock_session():
    return {
        'session_id': 'test_session_123',
        'user_authenticated': True
    }


@pytest.mark.asyncio
async def test_new_tool_success(mock_session):
    """Test successful execution of new tool."""
    with patch('src.services.new_service.NewService.perform_operation') as mock_service:
        mock_service.return_value = {
            'success': True,
            'message': 'Operation successful',
            'data': {'result': 'test_result'}
        }
        
        result = await new_tool_name(
            session=mock_session['session_id'],
            required_param='test_value'
        )
        
        assert result['success'] is True
        assert 'test_result' in str(result)


@pytest.mark.asyncio 
async def test_new_tool_missing_param(mock_session):
    """Test tool with missing required parameter."""
    result = await new_tool_name(
        session=mock_session['session_id']
        # missing required_param
    )
    
    assert result['success'] is False
    assert 'required_param' in result['message']


@pytest.mark.asyncio
async def test_new_tool_service_error(mock_session):
    """Test tool with service layer error."""
    with patch('src.services.new_service.NewService.perform_operation') as mock_service:
        mock_service.side_effect = Exception("Service error")
        
        result = await new_tool_name(
            session=mock_session['session_id'],
            required_param='test_value'
        )
        
        assert result['success'] is False
        assert 'Failed to execute operation' in result['message']
```

#### 6. Update Documentation

```python
# Add to FEATURES_DOCUMENTATION.md

### X.X `new_tool_name`

**Purpose**: Description of what the new tool does and its use case.

**Parameters**:
- `session` (optional): Session object or ID
- `required_param` (required): Description of required parameter
- `optional_param` (optional): Description of optional parameter

**Returns**:
- Success status and result data
- Error messages with guidance
- Updated session information

**Usage Example**:
```json
{
  "tool": "new_tool_name",
  "parameters": {
    "session": "shop_20250116_143052_abc123",
    "required_param": "example_value",
    "optional_param": 100
  }
}
```

**Response**:
```json
{
  "success": true,
  "message": " Operation completed successfully!",
  "session_id": "shop_20250116_143052_abc123",
  "result_data": {
    "processed_value": "example_value_processed"
  }
}
```

**Key Features**:
- Feature 1 description
- Feature 2 description
- Integration with existing workflow
```

---

## Service Layer Development

### Base Service Pattern

```python
# src/services/base_service.py

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from ..config import Config
from ..utils.logger import get_logger

logger = get_logger(__name__)


class BaseService(ABC):
    """Base class for all service layer implementations."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = get_logger(self.__class__.__name__)
    
    def validate_session(self, session_obj: Any) -> bool:
        """Validate session object."""
        return hasattr(session_obj, 'session_id') and session_obj.session_id
    
    def log_operation(self, operation: str, session_id: str, **kwargs):
        """Log service operations consistently."""
        self.logger.info(
            f"Service operation: {operation}",
            session_id=session_id,
            service=self.__class__.__name__,
            **kwargs
        )
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if service is healthy."""
        pass


class AsyncServiceMixin:
    """Mixin for async service operations."""
    
    async def with_timeout(self, coro, timeout: float = 30.0):
        """Execute coroutine with timeout."""
        try:
            return await asyncio.wait_for(coro, timeout=timeout)
        except asyncio.TimeoutError:
            raise ServiceTimeoutError(f"Operation timed out after {timeout}s")
    
    async def retry_operation(self, operation, max_attempts: int = 3):
        """Retry operation with exponential backoff."""
        for attempt in range(max_attempts):
            try:
                return await operation()
            except Exception as e:
                if attempt == max_attempts - 1:
                    raise
                await asyncio.sleep(2 ** attempt)
```

### Service Integration Pattern

```python
# src/services/example_service.py

import asyncio
from typing import Dict, Any, List, Optional
from .base_service import BaseService, AsyncServiceMixin
from ..buyer_backend_client import BuyerBackendClient
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ExampleService(BaseService, AsyncServiceMixin):
    """Example service demonstrating patterns."""
    
    def __init__(self, config: Config, backend_client: BuyerBackendClient):
        super().__init__(config)
        self.backend_client = backend_client
        self.cache = {}
    
    async def complex_operation(self, 
                               session_obj: Any,
                               data: Dict[str, Any]) -> Dict[str, Any]:
        """Example of complex service operation."""
        self.log_operation("complex_operation", session_obj.session_id)
        
        try:
            # Step 1: Validate inputs
            if not self.validate_session(session_obj):
                raise ValueError("Invalid session")
            
            # Step 2: Check cache
            cache_key = self._generate_cache_key(data)
            cached_result = await self._get_cached_result(cache_key)
            if cached_result:
                return cached_result
            
            # Step 3: Parallel operations
            backend_tasks = [
                self.backend_client.fetch_data(endpoint1, data),
                self.backend_client.fetch_data(endpoint2, data)
            ]
            
            results = await self.with_timeout(
                asyncio.gather(*backend_tasks, return_exceptions=True),
                timeout=30.0
            )
            
            # Step 4: Process results
            processed_result = await self._process_results(results)
            
            # Step 5: Cache result
            await self._cache_result(cache_key, processed_result)
            
            return {
                'success': True,
                'message': 'Operation completed successfully',
                'data': processed_result
            }
            
        except Exception as e:
            self.logger.error(f"Complex operation failed: {e}")
            return {
                'success': False,
                'message': f'Operation failed: {str(e)}'
            }
    
    async def health_check(self) -> bool:
        """Check service health."""
        try:
            # Test backend connectivity
            test_result = await self.backend_client.health_check()
            return test_result
        except Exception:
            return False
    
    def _generate_cache_key(self, data: Dict) -> str:
        """Generate cache key from data."""
        import hashlib
        import json
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.md5(data_str.encode()).hexdigest()
    
    async def _get_cached_result(self, cache_key: str) -> Optional[Dict]:
        """Get result from cache."""
        # Implementation depends on cache backend
        return self.cache.get(cache_key)
    
    async def _cache_result(self, cache_key: str, result: Dict):
        """Cache result."""
        self.cache[cache_key] = result
    
    async def _process_results(self, results: List[Any]) -> Dict:
        """Process backend results."""
        # Implementation specific to operation
        return {'processed': True, 'count': len(results)}
```

---

## Testing Framework

### Test Organization

```python
# tests/conftest.py - Shared test configuration

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from src.config import Config
from src.services import get_all_services


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_config():
    """Test configuration."""
    config = Config()
    config.debug_mode = True
    config.backend_endpoint = "http://test-backend"
    config.wil_api_key = "test_api_key"
    return config


@pytest.fixture
def mock_session():
    """Mock session object."""
    session = MagicMock()
    session.session_id = "test_session_123"
    session.user_authenticated = True
    session.auth_token = "test_token"
    session.cart = MagicMock()
    session.cart.is_empty.return_value = False
    session.cart.items = []
    return session


@pytest.fixture
def mock_backend_client():
    """Mock backend client."""
    client = AsyncMock()
    client.search_products.return_value = {
        'success': True,
        'products': [
            {
                'id': 'test_product',
                'name': 'Test Product',
                'price': 100.0
            }
        ]
    }
    return client


@pytest.fixture
def mock_services(mock_backend_client):
    """Mock service layer."""
    services = {}
    for service_name in ['search_service', 'cart_service', 'order_service']:
        services[service_name] = AsyncMock()
    return services
```

### Unit Test Examples

```python
# tests/unit/test_search_adapter.py

import pytest
from unittest.mock import patch, AsyncMock
from src.mcp_adapters import search_products


@pytest.mark.asyncio
async def test_search_products_success(mock_session, mock_services):
    """Test successful product search."""
    # Mock service response
    mock_services['search_service'].search_products.return_value = {
        'success': True,
        'products': [
            {
                'id': 'prod_1',
                'name': 'Organic Honey',
                'price': 250.0,
                'rating': 4.5
            }
        ],
        'total_results': 1
    }
    
    with patch('src.adapters.search.get_services', return_value=mock_services):
        result = await search_products(
            session=mock_session.session_id,
            query='organic honey',
            limit=10
        )
    
    assert result['success'] is True
    assert len(result['products']) == 1
    assert result['products'][0]['name'] == 'Organic Honey'
    assert 'Found 1 products' in result['message']


@pytest.mark.asyncio
async def test_search_products_empty_query(mock_session):
    """Test search with empty query."""
    result = await search_products(
        session=mock_session.session_id,
        query='',
        limit=10
    )
    
    assert result['success'] is False
    assert 'search query is required' in result['message'].lower()


@pytest.mark.asyncio
async def test_search_products_service_error(mock_session, mock_services):
    """Test search with service error."""
    # Mock service exception
    mock_services['search_service'].search_products.side_effect = Exception("Service error")
    
    with patch('src.adapters.search.get_services', return_value=mock_services):
        result = await search_products(
            session=mock_session.session_id,
            query='test query'
        )
    
    assert result['success'] is False
    assert 'Failed to search products' in result['message']
```

### Integration Test Examples

```python
# tests/integration/test_shopping_flow.py

import pytest
from src.mcp_adapters import *


@pytest.mark.integration
@pytest.mark.asyncio
async def test_complete_shopping_flow():
    """Test complete shopping workflow integration."""
    
    # 1. Initialize shopping session
    session_result = await initialize_shopping()
    assert session_result['success']
    session_id = session_result['session_id']
    
    # 2. Search for products
    search_result = await search_products(
        session=session_id,
        query='test product'
    )
    assert search_result['success']
    
    # 3. Add product to cart (if products found)
    if search_result.get('products'):
        product = search_result['products'][0]
        cart_result = await add_to_cart(
            session=session_id,
            item=product,
            quantity=1
        )
        assert cart_result['success']
        
        # 4. View cart
        view_result = await view_cart(session=session_id)
        assert view_result['success']
        assert view_result['cart_contents']['items']
        
        # 5. Get cart total
        total_result = await get_cart_total(session=session_id)
        assert total_result['success']
        assert total_result['totals']['final_total'] > 0


@pytest.mark.integration
@pytest.mark.asyncio 
async def test_authentication_flow():
    """Test authentication integration."""
    
    # Initialize session
    session_result = await initialize_shopping()
    session_id = session_result['session_id']
    
    # Test phone login
    auth_result = await phone_login(
        phone='9876543210',
        session=session_id
    )
    
    # Should succeed with mock backend
    assert auth_result['success']
    assert 'authenticated' in auth_result.get('user_profile', {})
```

### Performance Test Examples

```python
# tests/performance/test_load.py

import asyncio
import time
import statistics
from concurrent.futures import ThreadPoolExecutor
from src.mcp_adapters import search_products, initialize_shopping


async def single_search_test():
    """Single search operation test."""
    start_time = time.time()
    
    session_result = await initialize_shopping()
    search_result = await search_products(
        session=session_result['session_id'],
        query='performance test'
    )
    
    end_time = time.time()
    return {
        'duration': end_time - start_time,
        'success': search_result['success']
    }


async def concurrent_search_test(concurrent_users: int = 10):
    """Test concurrent search operations."""
    tasks = [single_search_test() for _ in range(concurrent_users)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    successful_results = [r for r in results if isinstance(r, dict) and r['success']]
    durations = [r['duration'] for r in successful_results]
    
    if durations:
        return {
            'total_requests': len(results),
            'successful_requests': len(successful_results),
            'success_rate': len(successful_results) / len(results),
            'avg_duration': statistics.mean(durations),
            'p95_duration': statistics.quantiles(durations, n=20)[18] if len(durations) > 5 else max(durations),
            'max_duration': max(durations),
            'min_duration': min(durations)
        }
    else:
        return {'error': 'No successful requests'}


@pytest.mark.performance
@pytest.mark.asyncio
async def test_search_performance():
    """Test search performance under load."""
    result = await concurrent_search_test(concurrent_users=5)
    
    # Performance assertions
    assert result['success_rate'] >= 0.8  # 80% success rate
    assert result['avg_duration'] < 2.0   # Average under 2 seconds
    assert result['p95_duration'] < 5.0   # 95th percentile under 5 seconds
    
    print(f"Performance test results: {result}")


@pytest.mark.performance
@pytest.mark.asyncio
async def test_session_creation_performance():
    """Test session creation performance."""
    session_count = 100
    start_time = time.time()
    
    tasks = [initialize_shopping() for _ in range(session_count)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    end_time = time.time()
    successful_sessions = [r for r in results if isinstance(r, dict) and r.get('success')]
    
    total_time = end_time - start_time
    sessions_per_second = len(successful_sessions) / total_time
    
    assert len(successful_sessions) >= session_count * 0.9  # 90% success rate
    assert sessions_per_second >= 10  # At least 10 sessions per second
    
    print(f"Created {len(successful_sessions)} sessions in {total_time:.2f}s ({sessions_per_second:.1f}/s)")
```

---

## Debugging & Troubleshooting

### Debugging Setup

```python
# Enable debug mode in development
# .env.dev
DEBUG_MODE=true
LOG_LEVEL=DEBUG

# Detailed logging configuration
# src/utils/debug_logger.py

import logging
import sys
from typing import Any, Dict

class DebugLogger:
    """Enhanced logger for debugging."""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.setup_debug_logging()
    
    def setup_debug_logging(self):
        """Setup detailed debug logging."""
        self.logger.setLevel(logging.DEBUG)
        
        # Console handler with detailed format
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - '
            '%(filename)s:%(lineno)d - %(funcName)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(console_handler)
    
    def trace_function_call(self, func_name: str, args: tuple, kwargs: dict):
        """Trace function calls."""
        self.logger.debug(
            f"TRACE: Calling {func_name} with args={args}, kwargs={kwargs}"
        )
    
    def trace_function_result(self, func_name: str, result: Any):
        """Trace function results.""" 
        self.logger.debug(f"TRACE: {func_name} returned {type(result).__name__}")

# Debugging decorator
def debug_trace(logger: DebugLogger):
    """Decorator to trace function calls."""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            logger.trace_function_call(func.__name__, args, kwargs)
            try:
                result = await func(*args, **kwargs)
                logger.trace_function_result(func.__name__, result)
                return result
            except Exception as e:
                logger.logger.error(f"TRACE: {func.__name__} raised {type(e).__name__}: {e}")
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            logger.trace_function_call(func.__name__, args, kwargs)
            try:
                result = func(*args, **kwargs)
                logger.trace_function_result(func.__name__, result)
                return result
            except Exception as e:
                logger.logger.error(f"TRACE: {func.__name__} raised {type(e).__name__}: {e}")
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator
```

### Common Debugging Scenarios

```python
# Session debugging
async def debug_session_issue(session_id: str):
    """Debug session-related issues."""
    try:
        # Check session file existence
        session_file = os.path.expanduser(f"~/.ondc-mcp/sessions/{session_id}.json")
        if os.path.exists(session_file):
            with open(session_file, 'r') as f:
                session_data = json.load(f)
            print(f"Session data: {json.dumps(session_data, indent=2)}")
        else:
            print(f"Session file not found: {session_file}")
        
        # Test session loading
        from src.services.session_service import get_session_service
        session_service = get_session_service()
        session_obj = await session_service.get_session(session_id)
        print(f"Session object: {session_obj}")
        
    except Exception as e:
        print(f"Session debug error: {e}")
        import traceback
        traceback.print_exc()

# Backend connectivity debugging  
async def debug_backend_connectivity():
    """Debug backend connectivity issues."""
    from src.buyer_backend_client import BuyerBackendClient
    
    client = BuyerBackendClient()
    
    try:
        # Test basic connectivity
        response = await client.test_connection()
        print(f"Backend connectivity: {response}")
        
        # Test authentication
        auth_response = await client.login_with_phone("+919876543210")
        print(f"Authentication test: {auth_response}")
        
    except Exception as e:
        print(f"Backend debug error: {e}")
        import traceback
        traceback.print_exc()

# Tool execution debugging
async def debug_tool_execution(tool_name: str, **params):
    """Debug specific tool execution."""
    try:
        # Import tool function
        from src import mcp_adapters
        tool_func = getattr(mcp_adapters, tool_name)
        
        print(f"Executing tool: {tool_name}")
        print(f"Parameters: {params}")
        
        # Execute with timing
        start_time = time.time()
        result = await tool_func(**params)
        end_time = time.time()
        
        print(f"Execution time: {end_time - start_time:.3f}s")
        print(f"Result: {json.dumps(result, indent=2)}")
        
        return result
        
    except Exception as e:
        print(f"Tool execution debug error: {e}")
        import traceback
        traceback.print_exc()
```

### Debugging Tools

```bash
# Debug scripts

# scripts/debug_session.py
#!/usr/bin/env python3
"""Debug session issues."""

import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.utils.debug_tools import debug_session_issue

async def main():
    session_id = sys.argv[1] if len(sys.argv) > 1 else input("Enter session ID: ")
    await debug_session_issue(session_id)

if __name__ == "__main__":
    asyncio.run(main())

# scripts/debug_backend.py  
#!/usr/bin/env python3
"""Debug backend connectivity."""

import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.utils.debug_tools import debug_backend_connectivity

if __name__ == "__main__":
    asyncio.run(debug_backend_connectivity())

# scripts/debug_tool.py
#!/usr/bin/env python3
"""Debug tool execution."""

import asyncio
import sys
import json
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.utils.debug_tools import debug_tool_execution

async def main():
    if len(sys.argv) < 2:
        print("Usage: python debug_tool.py <tool_name> [params_json]")
        sys.exit(1)
    
    tool_name = sys.argv[1]
    params = {}
    
    if len(sys.argv) > 2:
        params = json.loads(sys.argv[2])
    
    await debug_tool_execution(tool_name, **params)

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Performance Profiling

### Profiling Tools Setup

```python
# src/utils/profiler.py

import time
import cProfile
import pstats
import functools
from typing import Any, Callable
import asyncio

class PerformanceProfiler:
    """Performance profiling utilities."""
    
    def __init__(self):
        self.timings = {}
        self.call_counts = {}
    
    def time_function(self, func_name: str = None):
        """Decorator to time function execution."""
        def decorator(func: Callable) -> Callable:
            name = func_name or f"{func.__module__}.{func.__name__}"
            
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                start_time = time.perf_counter()
                try:
                    result = await func(*args, **kwargs)
                    return result
                finally:
                    end_time = time.perf_counter()
                    duration = end_time - start_time
                    self._record_timing(name, duration)
            
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                start_time = time.perf_counter()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    end_time = time.perf_counter() 
                    duration = end_time - start_time
                    self._record_timing(name, duration)
            
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        return decorator
    
    def _record_timing(self, func_name: str, duration: float):
        """Record function timing."""
        if func_name not in self.timings:
            self.timings[func_name] = []
            self.call_counts[func_name] = 0
        
        self.timings[func_name].append(duration)
        self.call_counts[func_name] += 1
    
    def get_stats(self) -> dict:
        """Get performance statistics."""
        stats = {}
        for func_name, timings in self.timings.items():
            stats[func_name] = {
                'call_count': self.call_counts[func_name],
                'total_time': sum(timings),
                'avg_time': sum(timings) / len(timings),
                'min_time': min(timings),
                'max_time': max(timings)
            }
        return stats
    
    def print_stats(self):
        """Print performance statistics."""
        stats = self.get_stats()
        print("\n=== Performance Statistics ===")
        for func_name, data in sorted(stats.items(), 
                                    key=lambda x: x[1]['total_time'], 
                                    reverse=True):
            print(f"{func_name}:")
            print(f"  Calls: {data['call_count']}")
            print(f"  Total: {data['total_time']:.4f}s")
            print(f"  Avg: {data['avg_time']:.4f}s")
            print(f"  Min/Max: {data['min_time']:.4f}s / {data['max_time']:.4f}s")
            print()

# Global profiler
profiler = PerformanceProfiler()

# Usage examples
@profiler.time_function("search_products_timed")
async def search_products_profiled(*args, **kwargs):
    return await search_products(*args, **kwargs)
```

### Memory Profiling

```python
# src/utils/memory_profiler.py

import psutil
import tracemalloc
import functools
from typing import Any, Callable

class MemoryProfiler:
    """Memory usage profiling."""
    
    def __init__(self):
        self.snapshots = {}
        tracemalloc.start()
    
    def profile_memory(self, func_name: str = None):
        """Decorator to profile memory usage."""
        def decorator(func: Callable) -> Callable:
            name = func_name or f"{func.__module__}.{func.__name__}"
            
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                # Take snapshot before
                snapshot_before = tracemalloc.take_snapshot()
                process = psutil.Process()
                memory_before = process.memory_info().rss
                
                try:
                    result = await func(*args, **kwargs)
                    return result
                finally:
                    # Take snapshot after
                    snapshot_after = tracemalloc.take_snapshot()
                    memory_after = process.memory_info().rss
                    
                    # Compare snapshots
                    top_stats = snapshot_after.compare_to(
                        snapshot_before, 'lineno'
                    )
                    
                    memory_diff = memory_after - memory_before
                    self._record_memory_usage(name, memory_diff, top_stats[:5])
            
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        return decorator
    
    def _record_memory_usage(self, func_name: str, memory_diff: int, top_stats):
        """Record memory usage."""
        self.snapshots[func_name] = {
            'memory_diff_mb': memory_diff / (1024 * 1024),
            'top_allocations': [str(stat) for stat in top_stats]
        }
    
    def print_memory_stats(self):
        """Print memory statistics."""
        print("\n=== Memory Usage Statistics ===")
        for func_name, data in self.snapshots.items():
            print(f"{func_name}:")
            print(f"  Memory diff: {data['memory_diff_mb']:.2f} MB")
            print("  Top allocations:")
            for allocation in data['top_allocations']:
                print(f"    {allocation}")
            print()

# Global memory profiler
memory_profiler = MemoryProfiler()
```

### Profiling Integration

```python
# Enable profiling in development
# scripts/run_with_profiling.py

import asyncio
import cProfile
import pstats
from src.utils.profiler import profiler, memory_profiler
from src.mcp_adapters import *

async def profiling_test():
    """Run performance test with profiling."""
    
    # Initialize session
    session_result = await initialize_shopping()
    session_id = session_result['session_id']
    
    # Run multiple operations for profiling
    for i in range(10):
        await search_products(session=session_id, query=f'test query {i}')
        await search_products(session=session_id, query='organic honey')
    
    # Print profiling results
    profiler.print_stats()
    memory_profiler.print_memory_stats()

if __name__ == "__main__":
    # Run with cProfile
    pr = cProfile.Profile()
    pr.enable()
    
    asyncio.run(profiling_test())
    
    pr.disable()
    
    # Print cProfile stats
    stats = pstats.Stats(pr)
    stats.sort_stats('cumulative')
    stats.print_stats(20)  # Top 20 functions
```

---

## Contributing Guidelines

### Git Workflow

```bash
# 1. Fork repository and create feature branch
git checkout -b feature/new-mcp-tool

# 2. Make changes with atomic commits
git add src/adapters/new_tool.py
git commit -m "feat: add new_tool MCP adapter

- Implement new_tool functionality
- Add parameter validation
- Include error handling
- Add unit tests"

# 3. Follow conventional commits
# Types: feat, fix, docs, style, refactor, test, chore
# Format: type(scope): description

# 4. Keep commits focused and atomic
git commit -m "feat(adapters): add new_tool adapter"
git commit -m "test(adapters): add tests for new_tool"
git commit -m "docs(features): document new_tool usage"

# 5. Rebase before submitting PR
git rebase main
git push origin feature/new-mcp-tool
```

### Code Review Checklist

```markdown
## Code Review Checklist

### Functionality
- [ ] Code solves the intended problem
- [ ] Edge cases are handled appropriately
- [ ] Error handling is comprehensive
- [ ] Input validation is implemented

### Code Quality
- [ ] Code follows project style guidelines
- [ ] Functions have clear, single responsibilities
- [ ] Variable and function names are descriptive
- [ ] Code is DRY (Don't Repeat Yourself)
- [ ] Complex logic is commented

### Testing
- [ ] Unit tests cover new functionality
- [ ] Integration tests pass
- [ ] Edge cases are tested
- [ ] Test coverage is adequate (>80%)

### Documentation
- [ ] Public APIs are documented
- [ ] README is updated if needed
- [ ] FEATURES_DOCUMENTATION.md is updated
- [ ] Code comments explain why, not what

### Performance
- [ ] No obvious performance bottlenecks
- [ ] Async patterns used appropriately
- [ ] Database queries are optimized
- [ ] Caching is implemented where beneficial

### Security
- [ ] Input sanitization is implemented
- [ ] No sensitive data in logs
- [ ] Authentication is properly handled
- [ ] Rate limiting is considered
```

### Pull Request Template

```markdown
## Description

Brief description of changes and motivation.

## Type of Change

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing

- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Screenshots (if applicable)

Include screenshots for UI changes.

## Checklist

- [ ] My code follows the style guidelines
- [ ] I have performed a self-review of my code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
```

---

## Release Process

### Version Management

```bash
# Follow semantic versioning (MAJOR.MINOR.PATCH)
# MAJOR: Breaking changes
# MINOR: New features (backward compatible)
# PATCH: Bug fixes (backward compatible)

# Version bumping
pip install bump2version

# Bump patch version
bump2version patch

# Bump minor version  
bump2version minor

# Bump major version
bump2version major
```

### Release Checklist

```markdown
## Release Checklist

### Pre-Release
- [ ] All tests pass
- [ ] Documentation is updated
- [ ] CHANGELOG.md is updated
- [ ] Version is bumped appropriately
- [ ] Security scan passes
- [ ] Performance benchmarks meet requirements

### Release
- [ ] Create release branch
- [ ] Tag release version
- [ ] Build and test Docker images
- [ ] Create GitHub release
- [ ] Update deployment documentation

### Post-Release
- [ ] Monitor error rates
- [ ] Verify deployment success
- [ ] Update production documentation
- [ ] Communicate release to stakeholders
```

### Deployment Automation

```yaml
# .github/workflows/release.yml
name: Release

on:
  push:
    tags:
      - 'v*'

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v3
        with:
          python-version: 3.11
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-dev.txt
      - name: Run tests
        run: pytest tests/ --cov=src/
      - name: Security scan
        run: bandit -r src/
  
  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Build Docker image
        run: |
          docker build -t ondc-mcp:${{ github.ref_name }} .
          docker build -t ondc-mcp:latest .
      - name: Push to registry
        run: |
          echo ${{ secrets.DOCKER_PASSWORD }} | docker login -u ${{ secrets.DOCKER_USERNAME }} --password-stdin
          docker push ondc-mcp:${{ github.ref_name }}
          docker push ondc-mcp:latest
  
  release:
    needs: build
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Create GitHub Release
        uses: actions/create-release@v1
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        with:
          tag_name: ${{ github.ref }}
          release_name: Release ${{ github.ref }}
          draft: false
          prerelease: false
```

This comprehensive developer guide provides all the information needed for contributing to and maintaining the ONDC Shopping MCP system.