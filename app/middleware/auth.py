# Update app/middleware/auth.py
import time
from fastapi import Request, HTTPException, Depends
from fastapi.security import APIKeyHeader
from fastapi.responses import JSONResponse
from starlette.status import HTTP_403_FORBIDDEN

from app.core.config import settings

# Define the API key header
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# Authentication dependency
async def verify_api_key(api_key: str = Depends(api_key_header)):
    if not api_key or api_key != settings.API_KEY:
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN, detail="Invalid or missing API key"
        )
    return api_key

# Rate limiting variables
request_counts = {}  # Keep track of requests per client

# Rate limiting middleware
async def rate_limit_middleware(request: Request, call_next):
    # Get client IP or use a unique identifier
    client_id = request.client.host if request.client else "unknown"
    
    # Check if client has exceeded rate limit
    current_minute = int(time.time() / 60)
    if client_id in request_counts:
        if current_minute == request_counts[client_id]["minute"]:
            if request_counts[client_id]["count"] >= settings.MAX_REQUESTS_PER_MINUTE:
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Rate limit exceeded. Try again later."}
                )
            request_counts[client_id]["count"] += 1
        else:
            # Reset for new minute
            request_counts[client_id] = {"minute": current_minute, "count": 1}
    else:
        request_counts[client_id] = {"minute": current_minute, "count": 1}
    
    # Process the request
    response = await call_next(request)
    return response