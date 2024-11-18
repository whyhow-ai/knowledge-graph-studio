"""Main entrypoint."""

import logging
import pathlib
from contextlib import asynccontextmanager
from logging import basicConfig
from pathlib import Path
from typing import Annotated, Any

import jwt
import logfire
from asgi_correlation_id import CorrelationIdMiddleware
from asgi_correlation_id.context import correlation_id
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exception_handlers import http_exception_handler
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from motor.motor_asyncio import AsyncIOMotorDatabase

from whyhow_api import __version__
from whyhow_api.config import Settings
from whyhow_api.custom_logging import configure_logging
from whyhow_api.database import close_mongo_connection, connect_to_mongo
from whyhow_api.dependencies import get_db, get_settings
from whyhow_api.middleware import RateLimiter
from whyhow_api.routers import (
    chunks,
    documents,
    graphs,
    nodes,
    queries,
    rules,
    schemas,
    tasks,
    triples,
    users,
    workspaces,
)

logger = logging.getLogger(
    "whyhow_api.main"
)  # set manually because of uvicorn


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore
    """Read environment (settings of the application)."""
    # Load settings
    settings = app.dependency_overrides.get(get_settings, get_settings)()

    # Set up jwks_client
    jwks_url = settings.api.auth0.jwks_url
    app.state.jwks_client = jwt.PyJWKClient(jwks_url)

    # Configure logging
    configure_logging(project_log_level=settings.dev.log_level)
    basicConfig(handlers=[logfire.LogfireLoggingHandler()])

    logger.info("Settings loaded")

    # Instrument PyMongo
    logfire.instrument_pymongo(capture_statement=True)

    # Startup database
    connect_to_mongo(uri=settings.mongodb.uri)
    try:
        yield
    finally:
        # Cleanup: close database connection
        close_mongo_connection()
        logger.info("Database connection closed")


settings_ = get_settings()
if settings_.logfire.token is None:
    logfire_token = None
else:
    logfire_token = settings_.logfire.token.get_secret_value()

logfire.configure(
    token=logfire_token, send_to_logfire="if-token-present", console=False
)

app = FastAPI(
    title="WhyHow API",
    summary="RAG with knowledge graphs",
    version=__version__,
    lifespan=lifespan,
    openapi_url=get_settings().dev.openapi_url,
)
logfire.instrument_fastapi(app)


@app.exception_handler(Exception)
async def unhandled_exception_handler(
    request: Request, exc: Exception
) -> Response:
    """Handle unhandled exceptions.

    The reason for this is to ensure that all exceptions
    have the correlation ID in the response headers.
    """
    # For debugging and in tests
    # import traceback

    # traceback.print_exception(exc)

    return await http_exception_handler(
        request,
        HTTPException(
            500,
            "Internal Server Error",
            headers={"X-Request-ID": correlation_id.get() or ""},
        ),
    )


app.add_middleware(RateLimiter)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"],
)


app.add_middleware(CorrelationIdMiddleware)

app.mount(
    "/static",
    StaticFiles(directory=Path(__file__).resolve().parent / "static"),
    name="static",
)

app.include_router(workspaces.router)
app.include_router(schemas.router)
app.include_router(graphs.router)
app.include_router(triples.router)
app.include_router(nodes.router)
app.include_router(documents.router)
app.include_router(chunks.router)
app.include_router(users.router)
app.include_router(queries.router)
app.include_router(rules.router)
app.include_router(tasks.router)


@app.get("/")
def root() -> str:
    """Check if the API is ready to accept traffic."""
    return f"Welcome to version {__version__} of the WhyHow API."


@app.get("/db")
async def database(db: AsyncIOMotorDatabase = Depends(get_db)) -> str:
    """Check if the database is connected."""
    ping_response = await db.command("ping")
    if int(ping_response["ok"]) != 1:
        return "Problem connecting to database cluster."
    else:
        return "Connected to database cluster."


@app.get("/settings")
def settings(settings: Annotated[Settings, Depends(get_settings)]) -> Any:
    """Get settings.

    The return type is Any not to put too much
    information in the OpenAPI schema.
    """
    return settings


def locate() -> None:
    """Find absolute path to this file and format for uvicorn."""
    file_path = pathlib.Path(__file__).resolve()
    current_path = pathlib.Path.cwd()

    relative_path = file_path.relative_to(current_path)
    dotted_path = str(relative_path).strip(".py").strip("/").replace("/", ".")

    res = f"{dotted_path}:app"
    print(res)
