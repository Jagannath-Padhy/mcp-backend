# Test Suite Documentation

This document describes the organized test suite after the September 2024 cleanup and tool call detection implementation.

## Test Structure

### 🧪 Integration Tests (`tests/integration/`)
**Purpose**: End-to-end testing of complete user journeys and system integration.

- `test_comprehensive_chat.py` - Complete chat API integration with all tool calls
- `test_unified_sessions.py` - Unified session management across API and MCP servers
- `test_complete_unified_session.py` - Complete session lifecycle testing
- `test_final_unified_verification.py` - Final architecture verification
- `test_select_with_real_providers.py` - Real provider SELECT operation testing

### 🛒 Cart Tests (`tests/cart/`)
**Purpose**: Cart management and isolation testing.

- `test_cart_isolation_direct.py` - Direct cart isolation between sessions
- `test_cart_with_items.py` - Cart operations with actual items

### 🔍 Search Tests (`tests/search/`)
**Purpose**: Search functionality including vector search and semantic matching.

- `test_complete_search_flow.py` - Complete search workflow testing
- `test_jams_fix.py` - Specific jam search fixes and validation
- `test_jams_search.py` - Jam search functionality
- `test_vector_direct.py` - Direct vector search testing

### 🔐 Session Tests (`tests/session/`)
**Purpose**: Session management, persistence, and isolation.

- `test_session_fix.py` - Session management fixes
- `test_session_isolation.py` - Session isolation between users
- `test_session_persistence.py` - Session persistence across restarts

## Running Tests

### All Tests
```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=backend --cov-report=html
```

### Category-Specific Tests
```bash
# Integration tests only
python -m pytest tests/integration/ -v

# Cart functionality
python -m pytest tests/cart/ -v

# Search functionality  
python -m pytest tests/search/ -v

# Session management
python -m pytest tests/session/ -v
```

### Individual Test Files
```bash
# Run specific comprehensive test
python tests/integration/test_comprehensive_chat.py

# Run specific cart test
python tests/cart/test_cart_with_items.py
```

## Tool Call Detection Testing

With the new enhanced GoogleAugmentedLLM implementation, tests now validate:

### ✅ Tool Call Monitoring
- Pre-tool call hooks are triggered
- Post-tool call hooks process results
- Performance metrics are collected
- Parallel extraction is attempted for eligible tools

### ✅ Parallel Structured Data Extraction
- Search tools trigger parallel extraction
- Cart tools provide enhanced context
- Checkout tools include UI hints
- Error handling for failed extractions

### ✅ Performance Metrics
- Tool execution timing
- Parallel extraction success rates
- Optimization suggestions
- Cache performance

## Test Performance Monitoring

### Metrics Endpoints for Testing
```bash
# Get performance metrics during tests
curl http://localhost:8001/api/v1/metrics/performance

# Reset metrics between test runs
curl -X POST http://localhost:8001/api/v1/metrics/reset
```

### Expected Behavior
- Tool call detection: 100% of eligible tools should be detected
- Parallel extraction: 80%+ success rate for eligible tools
- Performance overhead: <50ms additional per tool call
- Context enhancement: All major tools should include _context_type and _ui_hints

## Test Environment

### Prerequisites
```bash
# Ensure services are running
make up

# Initialize test data
make init

# Check health
curl http://localhost:8001/health
```

### Environment Variables for Testing
```bash
export LOG_LEVEL=DEBUG
export MCP_DEBUG_LEVEL=FULL
export MOCK_PAYMENTS_ENABLED=true
```

## Archived Tests

Legacy and redundant tests have been moved to `tests/archive/` with full documentation. These can be restored if needed but should not be required for normal development.

## Adding New Tests

### For Tool Call Detection Features
```python
async def test_tool_call_detection():
    # Test that tool calls are properly detected
    # Test parallel extraction works
    # Test performance metrics are updated
    pass
```

### For New MCP Tools
```python
async def test_new_mcp_tool():
    # Test tool functionality
    # Test integration with chat API
    # Test structured data extraction
    pass
```

### Test Guidelines
1. **Integration tests** for user-facing workflows
2. **Unit tests** for isolated functionality
3. **Performance tests** for tool call detection
4. **Validation tests** for data extraction accuracy

## Troubleshooting

### Common Issues
- **Tool call detection not working**: Check EnhancedGoogleAugmentedLLM is being used
- **Parallel extraction failing**: Check timeout settings and error logs
- **Session isolation issues**: Verify session persistence files
- **Performance degradation**: Check metrics endpoint for optimization suggestions

### Debug Logging
```bash
# Enable debug logging for tool calls
export MCP_DEBUG_LEVEL=FULL
export DEBUG_CURL_LOGGING=true

# Monitor tool call detection
tail -f backend/logs/mcp_operations.log | grep -E "PRE_TOOL|POST_TOOL"
```

This test suite provides comprehensive coverage while maintaining efficiency through the September 2024 cleanup and enhancement process.