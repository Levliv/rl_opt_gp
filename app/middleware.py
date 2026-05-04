import logging
import os
import time
import traceback

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

API_KEY = os.getenv("API_KEY")


async def logging_middleware(request: Request, call_next):
    start_time = time.time()
    client_ip = request.headers.get("X-Real-IP") or (request.client.host if request.client else "unknown")

    body_text = ""
    if request.method == "POST":
        try:
            body_bytes = await request.body()
            body_text = body_bytes.decode("utf-8")[:1000]
        except Exception:
            body_text = "<failed to read body>"

    try:
        response = await call_next(request)
    except Exception as exc:
        duration = time.time() - start_time
        logger.error(
            f"REQUEST FAILED: {request.method} {request.url.path} "
            f"from {client_ip} after {duration:.3f}s: {type(exc).__name__}: {exc}"
        )
        raise

    duration = time.time() - start_time
    log_msg = (
        f"{request.method} {request.url.path} "
        f"from {client_ip} -> {response.status_code} in {duration:.3f}s"
    )
    if body_text:
        log_msg += f" | body: {body_text}"

    if response.status_code >= 400:
        logger.warning(log_msg)
    else:
        logger.info(log_msg)

    return response


async def auth_middleware(request: Request, call_next):
    public_paths = ["/", "/health", "/metrics"]
    if request.url.path in public_paths:
        return await call_next(request)

    if not API_KEY:
        return await call_next(request)

    api_key = request.headers.get("X-API-Key")
    if api_key != API_KEY:
        return JSONResponse(
            status_code=401,
            content={"detail": "Unauthorized: invalid or missing API key"}
        )

    return await call_next(request)


async def validation_exception_handler(request: Request, exc: RequestValidationError):
    client_ip = request.headers.get("X-Real-IP") or (request.client.host if request.client else "unknown")
    body_text = ""
    try:
        body_bytes = await request.body()
        body_text = body_bytes.decode("utf-8")[:1000]
    except Exception:
        body_text = "<failed to read body>"

    logger.error(
        f"VALIDATION ERROR: {request.method} {request.url.path} from {client_ip} | "
        f"errors: {exc.errors()} | body: {body_text}"
    )
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()}
    )


async def global_exception_handler(request: Request, exc: Exception):
    client_ip = request.headers.get("X-Real-IP") or (request.client.host if request.client else "unknown")
    logger.error(
        f"UNHANDLED ERROR: {request.method} {request.url.path} from {client_ip} | "
        f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


def register_middleware(app: FastAPI):
    app.middleware("http")(logging_middleware)
    app.middleware("http")(auth_middleware)


def register_exception_handlers(app: FastAPI):
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, global_exception_handler)
