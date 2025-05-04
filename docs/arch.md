# Neural Interpreter System - Single-Port Architecture

## Architecture Overview

This document explains the secure single-port architecture implemented for the Neural Interpreter system. All external traffic flows through HTTPS (port 443), enhancing security and simplifying firewall configurations.

```
                                      ┌─────────────────────┐
                                      │                     │
                                      │  Clerk JWT Sidecar  │
                                      │                     │
                                      └─────────┬───────────┘
                                                │
                                                │ Auth Verification
                                                │
Internet         Firewall          ┌────────────▼─────────────┐
   ↓               ↓               │                          │
   │               │               │      Neural Interpreter  │
   │               │               │                          │
───┘     Only     ┌┘               └────────────┬─────────────┘
HTTPS    Port 443 │                             │
(443)   Allowed   │                             │
   │              │                             │
   │              │                             │
   └──────────────▶    ┌─────────────┐          │
                       │             │          │
                       │   Traefik   ◄──────────┘
                       │             │
                       └──────┬──────┘
                              │
                              │ Internal
                              │ Routing
                              │
              ┌───────────────┼───────────────┐
              │               │               │
              │               │               │
    ┌─────────▼─────┐ ┌───────▼───────┐ ┌─────▼───────────┐
    │               │ │               │ │                 │
    │  Admin UI     │ │ Pulsar        │ │ Other Internal  │
    │               │ │ (WebSocket)   │ │ Services        │
    └───────────────┘ └───────────────┘ └─────────────────┘
```

## Security Benefits

1. **Minimal Attack Surface**: Only a single port (443) is exposed to the internet, reducing potential entry points for attackers.

2. **Simplified Security Rules**: Firewall configurations are streamlined with a single inbound port.

3. **TLS Everywhere**: All traffic is encrypted with TLS, protecting data in transit.

4. **Centralized Authentication**: The JWT authentication sidecar verifies all requests before they reach application services.

5. **Defense in Depth**: Multiple security layers (TLS, JWT verification, role-based authorization) provide comprehensive protection.

## Component Roles

### 1. Traefik

Serves as the secure gateway with responsibilities for:
- TLS termination for all incoming HTTPS traffic
- Routing based on hostnames and URL paths
- Integration with the authentication sidecar via middleware
- Rate limiting and security headers

### 2. Clerk JWT Sidecar

Handles all authentication concerns:
- JWT token verification
- Role-based authorization
- Token refreshing and validation
- User attribute extraction

### 3. Pulsar over WebSocket

Instead of exposing the native Pulsar ports (6650/6651), clients connect through:
- WebSocket over HTTPS (wss://pulsar.yourdomain.com/ws)
- JWT authentication for client connections
- End-to-end encryption through TLS

### 4. Neural Interpreter

Configured to work with this architecture:
- Connects to Pulsar internally via direct protocol
- Provides clients with WebSocket connection details
- Processes requests after authentication verification

## Implementation Details

1. **Traefik Configuration**: 
   - Single entrypoint (websecure:443)
   - Host and path-based routing rules
   - Forward Auth middleware for JWT verification

2. **Pulsar WebSocket Configuration**:
   - WebSocket proxy enabled
   - Authentication integrated with JWT system
   - Internal services connect directly

3. **JWT Authentication Flow**:
   - Token verification before request processing
   - Role-based access enforcement
   - Integration with Clerk authentication system

## Routing Configuration

All routing is based on hosts and paths:

| Host                    | Path                | Service             | Authentication       |
|-------------------------|---------------------|---------------------|----------------------|
| api.yourdomain.com      | /v1/interpret       | Neural Interpreter  | JWT + Rate Limiting  |
| admin.yourdomain.com    | /                   | Admin Interface     | JWT + Admin Role     |
| pulsar.yourdomain.com   | /ws                 | Pulsar WebSocket    | JWT                  |
| traefik.yourdomain.com  | /                   | Traefik Dashboard   | JWT + Admin Role     |

## Network Flow

1. Client initiates HTTPS connection to appropriate hostname
2. Traefik terminates TLS and routes based on hostname/path
3. JWT verification occurs via the Clerk sidecar
4. If authentication succeeds, request forwards to appropriate service
5. Response returns through the same path, encrypted with TLS

## Scaling Considerations

This architecture can scale horizontally:
- Multiple Traefik instances behind a load balancer
- Multiple JWT sidecars and service instances
- Internal load balancing for high-demand services

## Monitoring and Observability

All traffic flows through a single entry point, enabling:
- Comprehensive traffic logging
- Centralized monitoring of all requests
- Simplified metrics collection
- End-to-end request tracing