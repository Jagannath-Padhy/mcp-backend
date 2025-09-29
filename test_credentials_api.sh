#!/bin/bash

echo "=== Testing Credential-Based Cart Isolation via API ==="
echo ""

# Test 1: Initialize without credentials (should get temporary ones)
echo "1. Testing initialization without credentials..."
curl -s -X POST http://localhost:8001/chat/message \
  -H "Content-Type: application/json" \
  -d '{
    "message": "initialize_shopping",
    "session_id": null
  }' | python3 -m json.tool | grep -E "(user_id|device_id|message)" | head -5

echo ""
echo "2. Testing initialization WITH specific credentials..."
curl -s -X POST http://localhost:8001/chat/message \
  -H "Content-Type: application/json" \
  -d '{
    "message": "initialize_shopping userId=test_user_abc deviceId=test_device_xyz",
    "session_id": null
  }' | python3 -m json.tool | grep -E "(user_id|device_id|message)" | head -5

echo ""
echo "3. Testing initialization with different credentials..."
curl -s -X POST http://localhost:8001/chat/message \
  -H "Content-Type: application/json" \
  -d '{
    "message": "initialize_shopping userId=another_user_123 deviceId=another_device_456",
    "session_id": null
  }' | python3 -m json.tool | grep -E "(user_id|device_id|message)" | head -5

echo ""
echo "=== Test Complete ==="
echo "Check above outputs to verify:"
echo "1. First test should show temp_user_* and temp_device_*"
echo "2. Second test should show test_user_abc and test_device_xyz"
echo "3. Third test should show another_user_123 and another_device_456"