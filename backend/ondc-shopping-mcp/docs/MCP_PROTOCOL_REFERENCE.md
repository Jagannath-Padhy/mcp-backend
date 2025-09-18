# Model Context Protocol (MCP) - Complete Reference

## Overview

The Model Context Protocol (MCP) is an open standard created by Anthropic for connecting AI systems to external tools and data sources. Released in November 2024, MCP provides a standardized way to expose context to Large Language Models (LLMs).

**Current Specification Version**: 2025-03-26  
**Previous Versions**: 2024-11-05 (legacy)

## Table of Contents

1. [Protocol Architecture](#protocol-architecture)
2. [Core Components](#core-components)
3. [Protocol Methods](#protocol-methods)
4. [Error Handling](#error-handling)
5. [Capability Negotiation](#capability-negotiation)
6. [Transport Mechanisms](#transport-mechanisms)
7. [Session Management](#session-management)
8. [Advanced Features](#advanced-features)
9. [Implementation Status](#implementation-status)

## Protocol Architecture

### Three Main Components

1. **Host**: The AI-powered application (e.g., Claude Desktop, Langflow, IDE)
2. **Client**: The connector component that lives within the host
3. **Server**: The service that exposes tools, resources, and prompts

### Communication Flow
```
Host Application
    ↓
MCP Client (1:1 relationship with server)
    ↓
MCP Server (provides tools, resources, prompts)
```

### Message Types (JSON-RPC 2.0)

1. **Requests**: Messages that initiate operations (require response)
2. **Responses**: Reply messages to requests
3. **Notifications**: One-way messages (no response required)

## Core Components

### 1. Tools
- **Purpose**: Enable models to perform actions with side effects
- **Examples**: Database queries, API calls, computations
- **Key Feature**: Can modify state

### 2. Resources
- **Purpose**: Provide read-only data access
- **Examples**: File contents, database records, API responses
- **Key Feature**: Idempotent operations only

### 3. Prompts
- **Purpose**: Predefined templates for interactions
- **Examples**: Code review templates, data analysis prompts
- **Key Feature**: Ensure consistency across teams

### 4. Sampling (Advanced)
- **Purpose**: Server-initiated LLM completions
- **Examples**: Context summarization, code explanation
- **Key Feature**: Enables agentic workflows

## Protocol Methods

### Required Methods (MUST implement)

#### `initialize`
- **Direction**: Client → Server
- **Purpose**: Establish connection and negotiate capabilities
- **Request**:
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "initialize",
  "params": {
    "protocolVersion": "2025-03-26",
    "capabilities": {
      "roots": { "listChanged": true },
      "sampling": {}
    },
    "clientInfo": {
      "name": "ExampleClient",
      "version": "1.0.0"
    }
  }
}
```
- **Response**: Server capabilities and version

#### `initialized`
- **Direction**: Client → Server
- **Purpose**: Notification that client is ready
- **Type**: Notification (no response)

#### `ping`
- **Direction**: Either direction
- **Purpose**: Health check / keep-alive
- **Can be used**: Before and after initialization

#### `tools/list`
- **Direction**: Client → Server
- **Purpose**: Discover available tools
- **Supports**: Pagination via `cursor` parameter
- **Response**: Array of Tool objects

#### `tools/call`
- **Direction**: Client → Server
- **Purpose**: Execute a tool
- **Parameters**: Tool name and arguments
- **Response**: Tool execution result

### Optional Methods (MAY implement)

#### `resources/list`
- **Direction**: Client → Server
- **Purpose**: Discover available resources
- **Supports**: Pagination
- **Response**: Array of Resource objects

#### `resources/read`
- **Direction**: Client → Server
- **Purpose**: Read resource content
- **Parameters**: Resource URI
- **Response**: Resource content (text or blob)

#### `resources/subscribe`
- **Direction**: Client → Server
- **Purpose**: Subscribe to resource changes
- **Response**: Subscription confirmation

#### `resources/unsubscribe`
- **Direction**: Client → Server
- **Purpose**: Unsubscribe from resource
- **Response**: Unsubscription confirmation

#### `prompts/list`
- **Direction**: Client → Server
- **Purpose**: Discover available prompts
- **Response**: Array of Prompt objects

#### `prompts/get`
- **Direction**: Client → Server
- **Purpose**: Get prompt with arguments
- **Response**: Formatted prompt messages

#### `sampling/createMessage`
- **Direction**: Server → Client
- **Purpose**: Request LLM completion
- **Requires**: Client sampling capability
- **Response**: LLM-generated message

#### `completion/complete`
- **Direction**: Either direction
- **Purpose**: Request text completion
- **Response**: Completion result

#### `logging/setLevel`
- **Direction**: Client → Server
- **Purpose**: Set server logging level
- **Levels**: debug, info, warning, error

#### `roots/list`
- **Direction**: Server → Client
- **Purpose**: Request filesystem boundaries
- **Response**: Array of root URIs

## Error Handling

### JSON-RPC Standard Error Codes

| Code    | Name                | Description                           |
|---------|---------------------|---------------------------------------|
| -32700  | Parse Error         | Invalid JSON received                 |
| -32600  | Invalid Request     | Not a valid Request object            |
| -32601  | Method Not Found    | Method doesn't exist/unavailable      |
| -32602  | Invalid Params      | Invalid method parameters             |
| -32603  | Internal Error      | Internal JSON-RPC error               |

### MCP-Specific Error Codes

| Code    | Name                | Description                           |
|---------|---------------------|---------------------------------------|
| -32002  | Resource Not Found  | Requested resource doesn't exist      |
| -32800  | Request Cancelled   | Request was cancelled                 |
| -32801  | Content Too Large   | Response content exceeds limits       |

### Error Response Format
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "error": {
    "code": -32601,
    "message": "Method not found",
    "data": {
      "method": "unknown_method"
    }
  }
}
```

## Capability Negotiation

### Client Capabilities
```json
{
  "roots": {
    "listChanged": true  // Supports roots list change notifications
  },
  "sampling": {},        // Supports server-initiated sampling
  "experimental": {      // Optional experimental features
    "feature_x": true
  }
}
```

### Server Capabilities
```json
{
  "tools": {
    "listChanged": true  // Tools can change dynamically
  },
  "resources": {
    "subscribe": true,   // Supports resource subscriptions
    "listChanged": true  // Resources can change dynamically
  },
  "prompts": {
    "listChanged": true  // Prompts can change dynamically
  },
  "logging": {}          // Supports logging
}
```

### Version Negotiation
- Client sends its supported version
- Server responds with compatible version
- If incompatible, server returns error

## Transport Mechanisms

### 1. STDIO (Standard Input/Output)
- **Status**: Current standard
- **Use Case**: Local process communication
- **Implementation**: Read/write to stdin/stdout

### 2. SSE (Server-Sent Events)
- **Status**: Available
- **Use Case**: Web-based clients
- **Implementation**: HTTP with event streams

### 3. HTTP with Streaming
- **Status**: New in 2025-03-26 spec
- **Use Case**: Serverless platforms
- **Improvement**: Better than legacy HTTP+SSE

## Session Management

### Protocol-Level Sessions
- **Mechanism**: JSON-RPC message ID correlation
- **Transport**: Each transport maintains session state
- **Isolation**: Each client-server pair is isolated

### Session Lifecycle
1. **Initialization**: Client connects and negotiates capabilities
2. **Active**: Normal message exchange
3. **Termination**: Client disconnects or timeout

### Session Continuity
- Maintained through transport layer
- Message IDs ensure request-response correlation
- Stateful connections preserve context

## Advanced Features

### Notifications

#### `notifications/cancelled`
- **Purpose**: Inform about cancelled request
- **Data**: Request ID that was cancelled

#### `notifications/progress`
- **Purpose**: Report operation progress
- **Data**: Progress percentage and message

#### `notifications/tools/list_changed`
- **Purpose**: Tools list has changed
- **Action**: Client should re-fetch tools

#### `notifications/resources/list_changed`
- **Purpose**: Resources list has changed
- **Action**: Client should re-fetch resources

#### `notifications/prompts/list_changed`
- **Purpose**: Prompts list has changed
- **Action**: Client should re-fetch prompts

#### `notifications/message`
- **Purpose**: Server log message
- **Data**: Log level and message

### Progress Tracking
```json
{
  "jsonrpc": "2.0",
  "method": "notifications/progress",
  "params": {
    "progress": 0.75,
    "total": 1.0,
    "message": "Processing items..."
  }
}
```

### Cancellation
```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "$/cancel",
  "params": {
    "id": 1  // ID of request to cancel
  }
}
```

### Logging
```json
{
  "jsonrpc": "2.0",
  "method": "notifications/message",
  "params": {
    "level": "info",
    "logger": "server.component",
    "message": "Operation completed successfully",
    "data": { "duration": 1234 }
  }
}
```

## Implementation Status

### ✅ Implemented in Our Server
- [x] Initialize/Initialized
- [x] Tools (list, call)
- [x] Resources (list, read)
- [x] Basic error handling
- [x] STDIO transport

### ⚠️ Partially Implemented
- [ ] Full capability negotiation
- [ ] Comprehensive error codes
- [ ] Protocol version validation

### ❌ Not Yet Implemented
- [ ] Prompts (list, get)
- [ ] Sampling (createMessage)
- [ ] Subscriptions (resources)
- [ ] Notifications (list_changed)
- [ ] Progress tracking
- [ ] Cancellation
- [ ] Logging levels
- [ ] Roots (filesystem boundaries)
- [ ] SSE transport
- [ ] HTTP streaming transport

## Best Practices

### 1. Initialization Sequence
- Always initialize before other operations
- Send initialized notification after receiving response
- Only ping allowed before initialization

### 2. Error Handling
- Use appropriate error codes
- Include helpful error data
- Handle all standard error codes

### 3. Capability Usage
- Check capabilities before using features
- Gracefully degrade if feature unavailable
- Don't assume capabilities exist

### 4. Session Management
- Maintain session through transport
- Don't manually manage session IDs in protocol
- Let SDK handle correlation

### 5. Resource URIs
- Use consistent URI schemes
- Make URIs predictable
- Include versioning if needed

## Security Considerations

### 1. User Consent
- Always require explicit user consent
- Clear authorization UIs
- No automatic actions without permission

### 2. Data Privacy
- Protect user data
- Implement access controls
- Minimal data exposure

### 3. Tool Safety
- Validate all tool inputs
- Sanitize outputs
- Prevent injection attacks

### 4. Sampling Controls
- User approval for LLM calls
- Rate limiting
- Cost controls

## References

- [Official MCP Specification](https://modelcontextprotocol.io/specification)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [MCP TypeScript SDK](https://github.com/modelcontextprotocol/typescript-sdk)
- [JSON-RPC 2.0 Specification](https://www.jsonrpc.org/specification)

## Version History

- **2025-03-26**: Current version with HTTP streaming
- **2024-11-05**: Initial release with STDIO and SSE
- **2024-11**: MCP announced by Anthropic

---

*This document serves as a complete reference for the Model Context Protocol. Even features not currently used in our implementation are documented here for future reference and to ensure nothing is missed.*