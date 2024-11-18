import logging
from unittest.mock import Mock

import pytest

from whyhow_api.custom_logging import configure_logging


@pytest.mark.parametrize("log_level", ["DEBUG", "INFO", "ERROR"])
def test_configure_logging(monkeypatch, log_level):
    fake_dictConfig = Mock()

    monkeypatch.setattr(
        "whyhow_api.custom_logging.dictConfig", fake_dictConfig
    )

    configure_logging(log_level)

    assert fake_dictConfig.call_count == 1
    args, _ = fake_dictConfig.call_args

    assert args[0]["loggers"]["whyhow_api"]["level"] == log_level


def test_logger_whyhow_api(capsys):
    # unfortunately, caplog does not work because loggers don't have propagate=True
    configure_logging(project_log_level="DEBUG")
    logger = logging.getLogger("whyhow_api")

    logger.info("This is an info message")
    logger.warning("This is a warning")
    logger.debug("This is a debug message")

    captured = capsys.readouterr().out

    assert "This is an info message" in captured
    assert "This is a warning" in captured
    assert "This is a debug message" in captured


@pytest.mark.parametrize("logger_name", ["foo", "bar"])
def test_loggers_rest(logger_name, capsys):
    configure_logging()

    logger = logging.getLogger(logger_name)

    logger.warning("This is a warning")
    logger.info("This is an info warning")
    logger.debug("This is a debug message")

    captured = capsys.readouterr().out
    assert "This is a warning" in captured
    assert "This is an info warning" not in captured
    assert "This is a debug message" not in captured
