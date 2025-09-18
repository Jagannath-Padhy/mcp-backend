#!/bin/bash
# MCP Backend Log Monitor - Real-time monitoring of all backend logs

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

echo "=========================================="
echo "      MCP BACKEND LOG MONITOR            "
echo "=========================================="
echo ""
echo "Log locations:"
echo "  API Errors:  backend/logs/api-server.err.log"
echo "  MCP Ops:     backend/logs/mcp_operations.log"
echo "  Supervisor:  backend/logs/supervisord.log"
echo ""
echo "Commands:"
echo "  1) Monitor all logs (default)"
echo "  2) API errors only"
echo "  3) MCP operations only"
echo "  4) Container logs"
echo "  5) SELECT errors only"
echo "  6) Checkout flow errors"
echo ""
read -p "Select option [1-6]: " option

LOG_DIR="/Users/jagannath/Desktop/mcp-backend/backend/logs"

case ${option:-1} in
  1)
    echo -e "${GREEN}Monitoring all logs...${NC}"
    tail -f "$LOG_DIR/api-server.err.log" "$LOG_DIR/mcp_operations.log" "$LOG_DIR/supervisord.log" 2>/dev/null | \
    while IFS= read -r line; do
      if [[ $line == *"ERROR"* ]]; then
        echo -e "${RED}[ERROR]${NC} $line"
      elif [[ $line == *"WARNING"* ]]; then
        echo -e "${YELLOW}[WARN]${NC} $line"
      elif [[ $line == *"SELECT"* ]] || [[ $line == *"select"* ]]; then
        echo -e "${BLUE}[SELECT]${NC} $line"
      elif [[ $line == *"INIT"* ]] || [[ $line == *"initialize"* ]]; then
        echo -e "${PURPLE}[INIT]${NC} $line"
      elif [[ $line == *"SUCCESS"* ]] || [[ $line == *"✅"* ]]; then
        echo -e "${GREEN}[OK]${NC} $line"
      else
        echo "$line"
      fi
    done
    ;;
    
  2)
    echo -e "${RED}Monitoring API errors only...${NC}"
    tail -f "$LOG_DIR/api-server.err.log" | grep -E "ERROR|FAILED|Exception" --color=always
    ;;
    
  3)
    echo -e "${BLUE}Monitoring MCP operations...${NC}"
    tail -f "$LOG_DIR/mcp_operations.log" --color=always
    ;;
    
  4)
    echo -e "${PURPLE}Monitoring container logs...${NC}"
    docker-compose logs -f backend
    ;;
    
  5)
    echo -e "${YELLOW}Monitoring SELECT errors only...${NC}"
    tail -f "$LOG_DIR/mcp_operations.log" "$LOG_DIR/api-server.err.log" | \
    grep -E "select|SELECT|on_select|locations|provider" --color=always
    ;;
    
  6)
    echo -e "${BLUE}Monitoring checkout flow...${NC}"
    tail -f "$LOG_DIR/mcp_operations.log" | \
    grep -E "CheckoutService|SELECT|INIT|CONFIRM|checkout|initialize_order|confirm_order" --color=always
    ;;
    
  *)
    echo "Invalid option. Using default (all logs)."
    exec "$0"
    ;;
esac