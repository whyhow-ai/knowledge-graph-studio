from unittest.mock import AsyncMock, MagicMock

import pytest
from bson import ObjectId

from whyhow_api.schemas.users import UserDocumentModel
from whyhow_api.services.crud.user import delete_user, get_user


class MockDocumentModel(UserDocumentModel):
    """Mock user document model."""

    name: str


@pytest.mark.asyncio
async def test_get_user(db_mock):
    api_key_mock = ObjectId()
    mock_collection = AsyncMock()
    mock_collection.find_one.return_value = {
        "api_key": str(api_key_mock),
        "active": True,
        "email": "test@whyhow.ai",
        "username": "test",
        "firstname": "test",
        "lastname": "test",
        "created_by": None,
    }

    db_mock.user = mock_collection

    result = await get_user(db_mock, username="test")

    assert result is not None
    assert result.api_key == str(api_key_mock)
    assert result.active is True
    assert result.email == "test@whyhow.ai"
    assert result.firstname == "test"
    assert result.lastname == "test"
    assert result.created_by is None


@pytest.mark.asyncio
async def test_get_user_not_found(db_mock):
    mock_collection = AsyncMock()
    mock_collection.find_one.return_value = None

    db_mock.user = mock_collection

    result = await get_user(db_mock, username="nonexistent")

    assert result is None
    mock_collection.find_one.assert_called_once_with(
        {"username": "nonexistent"}
    )


class AsyncContextManagerMock(AsyncMock):
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass


@pytest.mark.asyncio
async def test_delete_existed_user(monkeypatch):
    db = MagicMock()
    user_id = ObjectId()
    auth0 = MagicMock()

    session = MagicMock()
    session.start_transaction.return_value = AsyncMock()
    session.commit_transaction = AsyncMock()

    db_client = AsyncMock()
    db_client.start_session.return_value.__aenter__.return_value = session
    monkeypatch.setattr(db, "client", db_client)

    collections = [
        "chunk",
        "document",
        "graph",
        "node",
        "query",
        "schema",
        "triple",
        "workspace",
        "user",
    ]

    for collection in collections:
        setattr(db, collection, MagicMock())
        getattr(db, collection).delete_many = AsyncMock()
        if collection == "user":
            getattr(db, collection).delete_one = AsyncMock()
            getattr(db, collection).find_one = AsyncMock(
                return_value={"_id": user_id, "sub": "auth0|123456"}
            )

    auth0.users.delete = MagicMock()

    result = await delete_user(db=db, user_id=user_id, auth0=auth0)

    assert result is None
    session.start_transaction.assert_called_once()
    session.commit_transaction.assert_called_once()

    for collection in collections[:-1]:
        getattr(db, collection).delete_many.assert_called_once_with(
            {"created_by": user_id}, session=session
        )

    db.user.find_one.assert_called_once_with(
        {"_id": user_id}, {"sub": 1}, session=session
    )
    auth0.users.delete.assert_called_once_with("auth0|123456")
    db.user.delete_one.assert_called_once_with(
        {"_id": user_id}, session=session
    )
