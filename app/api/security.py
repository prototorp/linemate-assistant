"""API-key security scheme.

Declaring the key through fastapi.security (instead of reading the header by hand) puts it in the
OpenAPI schema, which is what makes Swagger UI show the "Authorize" button.
"""

import secrets

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from app.core.config import API_KEY

# auto_error=False -> a missing header gives us None, so we can return our own 401 instead of a generic 403
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(key: str | None = Security(_api_key_header)) -> str:
    if key is None or not secrets.compare_digest(key, API_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid X-API-Key header",
        )
    return key