# Archived Test Files

This directory contains test files that were archived during the September 2024 codebase cleanup to implement enhanced tool call detection and parallel structured data extraction.

## Archived Categories

### Debug Tests (Archived: Sept 26, 2024)
- `test_cart_debug.py` - Cart debugging tests, superseded by comprehensive integration tests
- `test_cart_debug_live.py` - Live cart debugging, no longer needed
- `test_format_debug.py` - Format debugging tests, functionality integrated into main tests

### Provider Fix Tests (Archived: Sept 26, 2024)
- `test_provider_fix_after_deployment.py` - Post-deployment provider fixes, issue resolved
- `test_provider_fix_final.py` - Final provider fix tests, no longer needed
- `test_provider_data_fix.py` - Provider data fix tests, superseded by current implementation

### Integration Test Duplicates (Archived: Sept 26, 2024)
- `test_credential_isolation.py` - Credential isolation tests, functionality merged into unified tests
- `test_new_architecture.py` - Architecture transition tests, no longer relevant
- `test_update_cart.py` - Cart update tests, covered by comprehensive integration tests

### Root-Level Test Files (Archived: Sept 26, 2024)
- `test_multiprovider_select.py` - Multi-provider SELECT tests, specific debugging case
- `test_select_with_real_data.py` - SELECT testing with real data, debugging purpose

## Retained Essential Tests

The following tests were kept as they provide comprehensive coverage:

### Integration Tests
- `test_comprehensive_chat.py` - Complete chat API integration testing
- `test_unified_sessions.py` - Unified session management testing
- `test_complete_unified_session.py` - Complete unified session flow
- `test_final_unified_verification.py` - Final verification of unified architecture
- `test_select_with_real_providers.py` - Real provider SELECT testing

### Functional Tests
- `tests/search/` - Search functionality tests (all retained)
- `tests/session/` - Session management tests (all retained)
- `tests/cart/test_cart_isolation_direct.py` - Direct cart isolation testing
- `tests/cart/test_cart_with_items.py` - Cart with items testing

## Restoration

If any of these archived tests need to be restored for debugging purposes, they can be moved back to their original locations. However, the current test suite should provide comprehensive coverage of all functionality.

## Tool Call Detection Tests

With the implementation of enhanced tool call detection and parallel structured data extraction, new test files should focus on:

1. Tool call pattern recognition
2. Parallel extraction performance
3. Performance metrics accuracy
4. Integration with existing MCP tools

These new capabilities are tested through the retained comprehensive integration tests.