#!/bin/bash

# Complete Shopping Journey Test Script
# Tests entire flow and captures all requests/responses

echo "=== COMPLETE SHOPPING JOURNEY TEST ==="
echo "Timestamp: $(date)"
echo "Testing with userId: EUSJ0ypAJJVdo3gXrUJe4uIBwDB2"
echo "deviceId: ed0bda0dd8c167a73721be5bb142dfc9"
echo ""

# Create logs directory
mkdir -p test_logs
LOG_DIR="test_logs/journey_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "Saving logs to: $LOG_DIR"
echo ""

# Function to make request and save logs
make_request() {
    local step_name="$1"
    local request_data="$2"
    local session_id="$3"
    
    echo "=== STEP: $step_name ==="
    echo "Request data: $request_data"
    
    # Prepare request data with session_id if provided
    if [ ! -z "$session_id" ]; then
        # Add session_id to existing request data
        request_data=$(echo "$request_data" | sed "s/}/,\"session_id\":\"$session_id\"}/")
    fi
    
    echo "Final request: $request_data" > "$LOG_DIR/${step_name}_request.json"
    
    # Make the request and capture response
    echo "Making request..."
    curl --location 'http://localhost:8001/api/v1/chat/stream' \
         --header 'Content-Type: application/json' \
         --header 'Accept: text/event-stream' \
         --data "$request_data" \
         --max-time 30 \
         --verbose \
         > "$LOG_DIR/${step_name}_response.log" 2> "$LOG_DIR/${step_name}_verbose.log"
    
    local exit_code=$?
    echo "Request completed with exit code: $exit_code"
    
    # Show response summary
    echo "Response summary:"
    echo "Lines: $(wc -l < "$LOG_DIR/${step_name}_response.log")"
    echo "Size: $(wc -c < "$LOG_DIR/${step_name}_response.log") bytes"
    
    # Extract session_id from response if this is initialization
    if [ "$step_name" = "01_initialize" ]; then
        SESSION_ID=$(grep -o '"session_id":"[^"]*"' "$LOG_DIR/${step_name}_response.log" | head -1 | cut -d'"' -f4)
        echo "Extracted session_id: $SESSION_ID"
        echo "$SESSION_ID" > "$LOG_DIR/session_id.txt"
    fi
    
    echo "Last few lines of response:"
    tail -5 "$LOG_DIR/${step_name}_response.log"
    echo ""
    echo "---"
    
    sleep 2
}

# Step 1: Initialize Shopping Session
make_request "01_initialize" '{
  "message": "initialize shopping session with userId: EUSJ0ypAJJVdo3gXrUJe4uIBwDB2 deviceid: ed0bda0dd8c167a73721be5bb142dfc9"
}'

# Get session ID for subsequent requests
if [ -f "$LOG_DIR/session_id.txt" ]; then
    SESSION_ID=$(cat "$LOG_DIR/session_id.txt")
    echo "Using session_id for remaining requests: $SESSION_ID"
else
    SESSION_ID="journey_test_$(date +%s)"
    echo "No session_id extracted, using fallback: $SESSION_ID"
fi

# Step 2: Search for Haldi (Test auto-add behavior)
make_request "02_search_haldi" '{
  "message": "i need some haldi for haldi ceremony"
}' "$SESSION_ID"

# Step 3: Search for Rice (Test search relevance)
make_request "03_search_rice" '{
  "message": "i also need rice for the ceremony"
}' "$SESSION_ID"

# Step 4: View Cart
make_request "04_view_cart" '{
  "message": "show me my cart"
}' "$SESSION_ID"

# Step 5: Checkout Process
make_request "05_checkout" '{
  "message": "proceed to checkout"
}' "$SESSION_ID"

# Step 6: Initialize Order (if we get that far)
make_request "06_payment" '{
  "message": "confirm payment with razorpay"
}' "$SESSION_ID"

echo ""
echo "=== JOURNEY TEST COMPLETE ==="
echo "All logs saved in: $LOG_DIR"
echo ""

# Generate summary report
echo "=== SUMMARY REPORT ===" > "$LOG_DIR/SUMMARY.md"
echo "Journey Test Results" >> "$LOG_DIR/SUMMARY.md"
echo "====================" >> "$LOG_DIR/SUMMARY.md"
echo "" >> "$LOG_DIR/SUMMARY.md"
echo "Test completed at: $(date)" >> "$LOG_DIR/SUMMARY.md"
echo "Session ID: $SESSION_ID" >> "$LOG_DIR/SUMMARY.md"
echo "" >> "$LOG_DIR/SUMMARY.md"

# Analyze each step
echo "## Step Analysis" >> "$LOG_DIR/SUMMARY.md"
for step_file in "$LOG_DIR"/*_response.log; do
    if [ -f "$step_file" ]; then
        step_name=$(basename "$step_file" _response.log)
        echo "" >> "$LOG_DIR/SUMMARY.md"
        echo "### $step_name" >> "$LOG_DIR/SUMMARY.md"
        echo "Response size: $(wc -c < "$step_file") bytes" >> "$LOG_DIR/SUMMARY.md"
        echo "Response lines: $(wc -l < "$step_file")" >> "$LOG_DIR/SUMMARY.md"
        
        # Check for truncation indicators
        if grep -q "complete.*true" "$step_file"; then
            echo "Status: ✅ Complete response received" >> "$LOG_DIR/SUMMARY.md"
        else
            echo "Status: ⚠️  Possibly truncated response" >> "$LOG_DIR/SUMMARY.md"
        fi
        
        # Extract final response content
        final_content=$(grep '"type":"response"' "$step_file" | tail -1 | grep -o '"content":"[^"]*"' | cut -d'"' -f4)
        if [ ! -z "$final_content" ]; then
            echo "Final response: ${final_content:0:100}..." >> "$LOG_DIR/SUMMARY.md"
        fi
    fi
done

echo ""
echo "Summary report generated: $LOG_DIR/SUMMARY.md"
echo ""

# Show immediate analysis
echo "=== IMMEDIATE ANALYSIS ==="
echo "Checking for common issues..."

# Check for truncation patterns
truncated_responses=$(grep -L "complete.*true" "$LOG_DIR"/*_response.log | wc -l)
echo "Potentially truncated responses: $truncated_responses"

# Check for session continuity
if [ -f "$LOG_DIR/session_id.txt" ]; then
    echo "✅ Session ID extracted successfully"
else
    echo "❌ Failed to extract session ID"
fi

# Check backend logs for errors
echo ""
echo "Recent backend errors:"
docker-compose logs backend | tail -20 | grep -i error || echo "No recent errors found"

echo ""
echo "=== TEST COMPLETE ==="
echo "Review detailed logs in: $LOG_DIR"