from src.services.shared.logging import get_logger

# Get a logger for your module
logger = get_logger("my_module")

# Log messages at different levels
logger.debug("Debug message")
logger.info("Info message")
logger.warning("Warning message")
logger.error("Error message")
logger.critical("Critical message")

# Custom log levels
logger.success("Operation succeeded")  # Between INFO and WARNING
logger.security("Security event detected")  # Between WARNING and ERROR
logger.performance("Performance metric")  # Between DEBUG and INFO
logger.trace("Detailed trace information")  # Below DEBUG

# Log with structured data
logger.struct(logging.INFO, "User registered", {
    "user_id": "123",
    "email": "user@example.com",
    "signup_source": "web"
})

# Performance logging with metrics
logger.performance("Database query completed", extra={
    "structured_data": {
        "metric": "db_query_time",
        "value": 0.235,  # seconds
        "tags": {
            "table": "users",
            "query_type": "select"
        }
    }
})

from src.services.shared.logging import log_context

# Set context variables for a block of code
with log_context(
    tenant_id="tenant-123",
    request_id="req-456",
    user_id="user-789",
    correlation_id="corr-abc",
    session_id="sess-def",
    operation="login"
):
    logger.info("Processing within context")
    # All logs inside this block will include the context information
from src.services.shared.logging import RequestLogger

# Create a request logger
req_logger = RequestLogger("api")

# Option 1: Manual tracking
req_logger.start_request("get_user_profile", 
                        tenant_id="tenant-123", 
                        user_id="user-456")
try:
    # Process request
    req_logger.log("info", "Fetching user data")
    # ...
    req_logger.end_request("success", user_found=True)
except Exception as e:
    req_logger.end_request("error", error=str(e))
    
# Option 2: Context manager
with req_logger.request_context("update_user", 
                               tenant_id="tenant-123", 
                               user_id="user-456") as req:
    req.log("info", "Updating user data")
    # ...
from src.services.shared.logging import log_execution_time, log_method_calls

# Log function execution time
@log_execution_time()
def process_data(data):
    # This will log when the function starts and ends, with execution time
    # ...
    return result

# Log all method calls in a class
@log_method_calls()
class UserService:
    def get_user(self, user_id):
        # This will log when the method is called and when it returns
        # ...
        return user
        
    def update_user(self, user_id, data):
        # Also logged automatically
        # ...
        return updated_user
from src.services.shared.logging import configure_logging

# Basic configuration
configure_logging({
    "level": "INFO",
    "console": True,
    "file": True
})

# Advanced configuration
configure_logging({
    "level": "DEBUG",
    "directory": "/var/log/myapp",
    "use_json": True,                  # Output logs as JSON
    "use_colors": True,                # Use colored console output
    "multi_tenant": True,              # Enable multi-tenant filtering
    "collect_performance": True,       # Enable performance metrics collection
    "console": True,
    "console_level": "INFO",
    "console_format": "%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    "file": True,
    "file_level": "DEBUG",
    "log_file": "/var/log/myapp/app.log",
    "max_file_size": 10485760,         # 10 MB
    "backup_count": 5                  # Keep 5 backup logs
})

from src.services.shared.logging import get_metrics, reset_metrics

# Log performance metrics
logger.performance("API request completed", extra={
    "structured_data": {
        "metric": "api_request_time",
        "value": 0.456,
        "tags": {
            "endpoint": "/users",
            "method": "GET"
        }
    }
})

# Get collected metrics
metrics = get_metrics()
for key, metric in metrics.items():
    print(f"{metric['metric']} - count: {metric['count']}, avg: {metric['sum']/metric['count']}")

# Reset metrics after reporting
reset_metrics()

# Custom filter for specific tenants
from src.services.shared.logging import MultiTenantFilter

# Only allow logs for specific tenants
tenant_filter = MultiTenantFilter(tenant_whitelist={"tenant-123", "tenant-456"})

# Or block specific tenants
tenant_filter = MultiTenantFilter(tenant_blacklist={"tenant-789"})

# Add to a handler
import logging
handler = logging.StreamHandler()
handler.addFilter(tenant_filter)