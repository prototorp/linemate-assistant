"""FastAPI entrypoint.  From the project root:

    uvicorn app.api.main:app --reload
    Swagger UI: http://127.0.0.1:8000/docs   (click Authorize, key: linemate-local-key)
"""

import logging
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routers import crew, documents
from app.core.exceptions import InvalidReferenceError

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
access_log = logging.getLogger("linemate.access")

app = FastAPI(
    title="LineMate",
    version="1.0.0",
    description="Hearthline kitchen operations assistant: documents, tickets, analytics, and grounded Q&A.",
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Runs for every request before it reaches a route: tags it with an id, times it, logs
    method/path/status/duration, and echoes both back as response headers."""
    request_id = uuid.uuid4().hex[:8]
    start = time.perf_counter()
    response = await call_next(request)  # hand control to the matching route, get its response back
    duration_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-Ms"] = f"{duration_ms:.1f}"
    access_log.info(
        "%s %s %s -> %s (%.1fms)", request_id, request.method, request.url.path,
        response.status_code, duration_ms,
    )
    return response


@app.exception_handler(InvalidReferenceError)
async def invalid_reference_handler(request: Request, exc: InvalidReferenceError):
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.get("/", tags=["health"])
def health_check() -> dict[str, str]:
    return {"status": "ok", "service": "LineMate"}


app.include_router(documents.router)
app.include_router(crew.router)