import logging
import os

from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

from app.tasks import lifespan
from app.middleware import register_middleware, register_exception_handlers
from app.routes import router

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

DISABLE_DOCS = os.getenv("DISABLE_DOCS", "true").lower() == "true"

app = FastAPI(
    title="RL Ad Reward Optimization Service",
    description="Reinforcement Learning service for optimizing ad rewards in mobile game",
    version="1.0.0",
    docs_url=None if DISABLE_DOCS else "/docs",
    redoc_url=None if DISABLE_DOCS else "/redoc",
    openapi_url=None if DISABLE_DOCS else "/openapi.json",
    lifespan=lifespan,
)
Instrumentator().instrument(app).expose(app)
register_middleware(app)
register_exception_handlers(app)
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
