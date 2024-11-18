"""Logging configuration for the project."""

from logging.config import dictConfig
from typing import Literal

LOG_LEVEL_TYPE = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


def configure_logging(project_log_level: LOG_LEVEL_TYPE = "INFO") -> None:
    """Configure logging for the project."""
    dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "filters": {
                "correlation_id": {
                    "()": "asgi_correlation_id.CorrelationIdFilter",
                    "uuid_length": 8,
                    "default_value": "-",
                },
            },
            "formatters": {
                "console": {
                    "class": "logging.Formatter",
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                    "format": "%(levelname)s:  %(asctime)s %(name)s:%(lineno)d [%(correlation_id)s] %(message)s",
                },
                # Copy uvicorn's formatters
                "access": {
                    "()": "uvicorn.logging.AccessFormatter",
                    "fmt": "%(levelprefix)s %(client_addr)s - %(request_line)s [%(correlation_id)s] %(status_code)s",
                },
                "default": {
                    "()": "uvicorn.logging.DefaultFormatter",
                    "fmt": "%(levelprefix)s %(message)s",
                    "use_colors": None,
                },
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "filters": ["correlation_id"],
                    "formatter": "console",
                    "stream": "ext://sys.stdout",
                },
                # Copy uvicorn's handlers
                "access": {
                    "class": "logging.StreamHandler",
                    "filters": ["correlation_id"],
                    "formatter": "access",
                    "stream": "ext://sys.stdout",
                },
                "default": {
                    "class": "logging.StreamHandler",
                    "formatter": "default",
                    "stream": "ext://sys.stderr",
                },
            },
            "loggers": {
                # root logger
                "": {"handlers": ["console"], "level": "WARNING"},
                # project logger
                "whyhow_api": {
                    "handlers": ["console"],
                    "level": project_log_level,
                    "propagate": False,
                },
                # uvicorn loggers
                "uvicorn": {
                    "handlers": ["default"],
                    "level": "INFO",
                    "propagate": False,
                },
                "uvicorn.error": {
                    "level": "INFO",
                },
                "uvicorn.access": {
                    "handlers": ["access"],
                    "level": "INFO",
                    "propagate": False,
                },
            },
        }
    )
