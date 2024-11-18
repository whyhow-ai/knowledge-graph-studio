import logging
from unittest.mock import AsyncMock, patch

import pytest
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorClientSession

from whyhow_api.database import (
    close_mongo_connection,
    connect_to_mongo,
    get_client,
    get_session,
)

logger = logging.getLogger(__name__)


@pytest.fixture
def mock_client():
    return AsyncMock(spec=AsyncIOMotorClient)


@pytest.fixture
def mock_session():
    return AsyncMock(spec=AsyncIOMotorClientSession)


@patch("motor.motor_asyncio.AsyncIOMotorClient", new_callable=AsyncMock)
@patch("motor.core.AgnosticClientSession", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_get_session(mock_session, mock_client):
    # Arrange
    mock_client.start_session.return_value = mock_session

    # Act
    async with get_session(mock_client):
        # Assert
        mock_client.start_session.assert_called_once()
        mock_session.start_transaction.assert_called_once()

    # Assert
    mock_session.commit_transaction.assert_called_once()
    mock_session.end_session.assert_called_once()


@patch("motor.motor_asyncio.AsyncIOMotorClient", new_callable=AsyncMock)
@patch("motor.core.AgnosticClientSession", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_get_session_with_exception(mock_session, mock_client):
    # Arrange
    mock_client.start_session.return_value = mock_session
    mock_session.start_transaction.side_effect = Exception("Test exception")

    # Act & Assert
    with pytest.raises(Exception, match="Test exception"):
        async with get_session(mock_client):
            pass

    # Assert
    mock_session.abort_transaction.assert_called_once()
    mock_session.end_session.assert_called_once()


@pytest.mark.asyncio
async def test_connect_to_mongo(mock_client):
    with patch("whyhow_api.database.get_client", return_value=mock_client):
        connect_to_mongo("mongodb://localhost:27017")
        assert get_client() is not None
        assert isinstance(get_client(), AsyncIOMotorClient)


@pytest.mark.asyncio
async def test_connect_to_mongo_no_client():
    with patch("whyhow_api.database.get_client", return_value=None):
        connect_to_mongo("mongodb://localhost:27017")
        assert get_client() is not None
        assert isinstance(get_client(), AsyncIOMotorClient)


@pytest.mark.asyncio
async def test_close_mongo_connection(mock_client):
    with patch("whyhow_api.database.get_client", return_value=mock_client):
        connect_to_mongo("mongodb://localhost:27017")
        close_mongo_connection()
        assert get_client() is None


@pytest.mark.asyncio
async def test_close_mongo_connection_no_client():
    with patch("whyhow_api.database.get_client", return_value=None):
        close_mongo_connection()
        assert get_client() is None


@pytest.mark.asyncio
async def test_get_session_exception(mock_client):
    mock_client.start_session.side_effect = Exception("Connection error")

    with pytest.raises(Exception):
        async with get_session(mock_client):
            pass


@patch("whyhow_api.database.client", None)
def test_get_client_none():
    """Test the get_client function when client is None."""
    result = get_client()
    assert result is None


@patch("whyhow_api.database.client", new_callable=lambda: None)
def test_get_client(mock_client):
    """Test the get_client function with a mocked client."""

    # Set up the mock to be an AsyncIOMotorClient instance
    mock_client = AsyncIOMotorClient("mongodb://testhost:27017")

    # Set the mock to the global variable 'client'
    with patch("whyhow_api.database.client", mock_client):
        result = get_client()

    # Assert that the result is the expected mocked instance
    assert result == mock_client
    assert isinstance(result, AsyncIOMotorClient)
