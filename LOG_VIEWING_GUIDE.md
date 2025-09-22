# MCP Backend Log Viewing Guide

## Quick Access Commands

### 🔥 Most Important - MCP Operations Log
```bash
# Real-time MCP tool executions (main debugging log)
docker exec -it mcp-backend tail -f /app/logs/mcp_operations.log

# Last 100 MCP operations
docker exec mcp-backend tail -100 /app/logs/mcp_operations.log

# Search for specific operations
docker exec mcp-backend grep -i "search_products\|add_to_cart" /app/logs/mcp_operations.log | tail -20
```

### 🚨 API Server Errors
```bash
# Real-time API errors
docker exec -it mcp-backend tail -f /app/logs/api-server.err.log

# Recent API errors
docker exec mcp-backend tail -50 /app/logs/api-server.err.log
```

### 📊 Combined Logs
```bash
# All important logs together
docker exec -it mcp-backend tail -f /app/logs/mcp_operations.log /app/logs/api-server.err.log

# Container logs (includes supervisor output)
docker-compose logs -f backend

# Container logs with timestamps
docker-compose logs -f --timestamps backend
```

## Enhanced Monitor Script

Use the existing color-coded monitor:
```bash
./monitor_logs.sh
```

**Options:**
- `1` - All logs with color coding (ERROR=red, WARN=yellow, etc.)
- `2` - API errors only
- `3` - MCP operations only (recommended for debugging)
- `4` - Container logs
- `5` - SELECT/checkout errors only
- `6` - Checkout flow monitoring

## Available Log Files

| File | Purpose | Key Content |
|------|---------|-------------|
| `/app/logs/mcp_operations.log` | **Main MCP debugging** | All tool calls, sessions, search results |
| `/app/logs/api-server.err.log` | API server errors | FastAPI errors, agent failures |
| `/app/logs/api-server.out.log` | API server output | Request logs, successful operations |
| `/app/logs/supervisord.log` | Process management | Service starts/stops |

## Debugging Common Issues

### Search Problems
```bash
# Monitor search operations
docker exec mcp-backend tail -f /app/logs/mcp_operations.log | grep -i "search\|vector\|rerank"
```

### Cart Issues
```bash
# Monitor cart operations
docker exec mcp-backend tail -f /app/logs/mcp_operations.log | grep -i "cart\|add_to_cart\|view_cart"
```

### Checkout Flow
```bash
# Monitor complete checkout process
docker exec mcp-backend tail -f /app/logs/mcp_operations.log | grep -i "select\|init\|confirm\|checkout"
```

### Session Problems
```bash
# Monitor session management
docker exec mcp-backend tail -f /app/logs/mcp_operations.log | grep -i "session\|initialize_shopping"
```

## Supervisor Web Interface ✅

**Access**: http://localhost:9001  
**Username**: `admin` (configurable via `SUPERVISOR_WEB_USERNAME`)  
**Password**: `supervisor2024` (configurable via `SUPERVISOR_WEB_PASSWORD`)

**Features:**
- View process status and control (start/stop/restart)
- View recent log output for each service
- Real-time process monitoring
- Web-based process management

**Configuration (Optional):**
```bash
# In .env file
SUPERVISOR_WEB_ENABLED=true
SUPERVISOR_WEB_PORT=9001
SUPERVISOR_WEB_USERNAME=admin
SUPERVISOR_WEB_PASSWORD=your_secure_password
```

**Security Note**: Change the default password in production environments.

## Tips

1. **Use multiple terminals** - Run different log commands in separate terminals
2. **Filter by time** - Add `--since="5m"` to docker-compose logs for recent entries
3. **Save searches** - Create aliases for common log queries
4. **Real-time monitoring** - Use `tail -f` for live log streaming

## Troubleshooting Log Access

If log files are empty or missing:
```bash
# Check log directory
docker exec mcp-backend ls -la /app/logs/

# Check if services are running
docker exec mcp-backend supervisorctl status

# Restart if needed
docker-compose restart backend
```