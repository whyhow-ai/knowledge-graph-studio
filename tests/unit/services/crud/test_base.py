from unittest.mock import AsyncMock, MagicMock

import pytest
from bson import ObjectId

from whyhow_api.schemas.base import BaseDocument, get_utc_now
from whyhow_api.services.crud.base import (
    create_one,
    delete_all,
    delete_one,
    get_all,
    get_all_count,
    get_one,
    update_one,
)


class MockDocumentModel(BaseDocument):
    """Mock document model."""

    name: str


@pytest.mark.asyncio
async def test_get_all_limit_negative_1():
    user_id = ObjectId()
    mock_collection = MagicMock()

    mock_cursor = MagicMock()
    mock_cursor.to_list = AsyncMock(
        return_value=[
            {
                "_id": ObjectId(),
                "name": "test",
                "created_by": user_id,
                "created_at": get_utc_now(),
                "updated_at": get_utc_now(),
            }
        ]
    )

    mock_collection.aggregate.return_value = mock_cursor

    result = await get_all(
        collection=mock_collection,
        document_model=MockDocumentModel,
        user_id=user_id,
        skip=0,
        limit=-1,
        aggregation_query=[{"$match": {"foo": "bar"}}],
    )

    assert len(result) == 1
    assert result[0].name == "test"
    assert result[0].created_by == user_id

    mock_collection.aggregate.assert_called_once_with(
        [
            {"$match": {"created_by": user_id}},
            {"$match": {"foo": "bar"}},
            {"$sort": {"created_at": -1, "_id": -1}},
            {"$skip": 0},
        ]
    )
    mock_cursor.to_list.assert_awaited_once_with(length=None)


@pytest.mark.asyncio
async def test_get_all_limit_positive():
    user_id = ObjectId()
    mock_collection = MagicMock()

    mock_cursor = MagicMock()
    mock_cursor.to_list = AsyncMock(
        return_value=[
            {
                "_id": ObjectId(),
                "name": "test",
                "created_by": user_id,
                "created_at": get_utc_now(),
                "updated_at": get_utc_now(),
            }
        ]
    )

    mock_collection.aggregate.return_value = mock_cursor

    result = await get_all(
        collection=mock_collection,
        document_model=MockDocumentModel,
        user_id=user_id,
        skip=0,
        limit=10,
        aggregation_query=[{"$match": {"foo": "bar"}}],
    )

    assert len(result) == 1
    assert result[0].name == "test"
    assert result[0].created_by == user_id
    mock_collection.aggregate.assert_called_once_with(
        [
            {"$match": {"created_by": user_id}},
            {"$match": {"foo": "bar"}},
            {"$sort": {"created_at": -1, "_id": -1}},
            {"$skip": 0},
            {"$limit": 10},
        ]
    )
    mock_cursor.to_list.assert_awaited_once_with(length=10)


@pytest.mark.asyncio
async def test_get_all_limit_gt_negative_1():
    user_id = ObjectId()
    mock_collection = MagicMock()

    with pytest.raises(
        ValueError,
        match="Limit must be greater than or equal to 0 or -1 for unrestricted.",
    ):
        await get_all(
            collection=mock_collection,
            document_model=MockDocumentModel,
            user_id=user_id,
            skip=0,
            limit=-5,
        )


@pytest.mark.asyncio
async def test_get_one():
    user_id = ObjectId()
    document_id = ObjectId()
    mock_collection = AsyncMock()
    dt_now = get_utc_now()
    mock_collection.find_one.return_value = {
        "_id": document_id,
        "name": "test",
        "created_by": user_id,
        "created_at": dt_now,
        "updated_at": dt_now,
    }

    result = await get_one(
        collection=mock_collection,
        document_model=MockDocumentModel,
        user_id=user_id,
        id=document_id,
    )

    assert result is not None
    assert result.id == document_id
    assert result.name == "test"
    assert result.created_by == user_id
    assert result.created_at == dt_now
    assert result.updated_at == dt_now


@pytest.mark.asyncio
async def test_get_one_with_filters():
    user_id = ObjectId()
    document_id = ObjectId()
    mock_collection = AsyncMock()
    dt_now = get_utc_now()
    mock_collection.find_one.return_value = {
        "_id": document_id,
        "name": "test",
        "created_by": user_id,
        "created_at": dt_now,
        "updated_at": dt_now,
    }

    result = await get_one(
        collection=mock_collection,
        document_model=MockDocumentModel,
        user_id=user_id,
        id=document_id,
        filters={"foo": "bar"},
    )

    assert result is not None
    assert result.id == document_id
    assert result.name == "test"
    assert result.created_by == user_id
    assert result.created_at == dt_now
    assert result.updated_at == dt_now

    expected_query = {"_id": document_id, "created_by": user_id, "foo": "bar"}
    actual_query = mock_collection.find_one.call_args[0][0]
    assert expected_query == actual_query


@pytest.mark.asyncio
async def test_get_one_no_return():
    user_id = ObjectId()
    mock_collection = AsyncMock()
    mock_collection.find_one.return_value = None

    result = await get_one(
        collection=mock_collection,
        document_model=MockDocumentModel,
        user_id=user_id,
    )

    assert result is None
    assert mock_collection.find_one.called


@pytest.mark.asyncio
async def test_get_all_count_with_filters():
    user_id = ObjectId()
    mock_collection = MagicMock()

    mock_cursor = MagicMock()
    mock_cursor.to_list = AsyncMock(return_value=[{"total": 5}])

    mock_collection.aggregate.return_value = mock_cursor

    result = await get_all_count(
        collection=mock_collection,
        user_id=user_id,
        aggregation_query=[{"$match": {"foo": "bar"}}],
    )

    assert result == 5
    mock_collection.aggregate.assert_called_once_with(
        [
            {"$match": {"created_by": user_id}},
            {"$match": {"foo": "bar"}},
            {"$count": "total"},
        ]
    )
    # Ensure that to_list is awaited correctly
    mock_cursor.to_list.assert_awaited_once_with(None)


@pytest.mark.asyncio
async def test_get_all_count():
    user_id = ObjectId()
    mock_collection = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.to_list = AsyncMock(return_value=[{"total": 5}])
    mock_collection.aggregate.return_value = mock_cursor

    result = await get_all_count(collection=mock_collection, user_id=user_id)

    assert result == 5


@pytest.mark.asyncio
async def test_create_one():
    user_id = ObjectId()
    document_data = MockDocumentModel(
        id=ObjectId(),
        name="test",
        created_by=user_id,
        created_at=get_utc_now(),
        updated_at=get_utc_now(),
    )
    mock_collection = AsyncMock()
    mock_collection.insert_one.return_value.inserted_id = document_data.id

    result = await create_one(
        collection=mock_collection,
        document_model=MockDocumentModel,
        document=document_data,
        user_id=user_id,
    )

    assert result is not None
    assert result.id == document_data.id
    assert result.name == "test"
    assert result.created_by == user_id
    assert result.created_at == document_data.created_at
    assert result.updated_at == document_data.updated_at


@pytest.mark.asyncio
async def test_create_one_missing_created_by():
    user_id = ObjectId()

    class TemptDocumentModel(BaseDocument):
        # We want to test that the created_by field is not required
        name: str
        created_by: ObjectId | None = None

    document_data = TemptDocumentModel(
        id=ObjectId(),
        name="test",
        created_at=get_utc_now(),
        updated_at=get_utc_now(),
    )
    mock_collection = AsyncMock()
    mock_collection.insert_one.return_value.inserted_id = document_data.id

    result = await create_one(
        collection=mock_collection,
        document_model=TemptDocumentModel,
        document=document_data,
        user_id=user_id,
    )

    assert result is not None
    assert result.id == document_data.id
    assert result.name == "test"
    assert result.created_by == user_id
    assert result.created_at == document_data.created_at
    assert result.updated_at == document_data.updated_at


@pytest.mark.asyncio
async def test_update_one():
    document_id = ObjectId()
    updated_data = MockDocumentModel(
        id=document_id,
        name="updated_test",
        created_by=ObjectId(),
        created_at="2023-01-01T00:00:00Z",
        updated_at="2023-01-01T00:00:00Z",
    )
    mock_collection = AsyncMock()
    mock_collection.find_one.return_value = updated_data.model_dump(
        by_alias=True
    )

    result = await update_one(
        collection=mock_collection,
        document_model=MockDocumentModel,
        id=document_id,
        document=updated_data,
        user_id=ObjectId(),
    )

    assert result is not None
    assert result.name == "updated_test"


@pytest.mark.asyncio
async def test_update_one_no_return():
    document_id = ObjectId()
    updated_data = MockDocumentModel(
        id=document_id,
        name="updated_test",
        created_by=ObjectId(),
        created_at="2023-01-01T00:00:00Z",
        updated_at="2023-01-01T00:00:00Z",
    )
    mock_collection = AsyncMock()
    mock_collection.find_one.return_value = None

    result = await update_one(
        collection=mock_collection,
        document_model=MockDocumentModel,
        id=document_id,
        document=updated_data,
        user_id=ObjectId(),
    )

    assert result is None


@pytest.mark.asyncio
async def test_delete_one():
    user_id = ObjectId()
    document_id = ObjectId()
    mock_collection = AsyncMock()
    mock_collection.find_one.return_value = {
        "_id": document_id,
        "name": "test",
        "created_by": user_id,
        "created_at": get_utc_now(),
        "updated_at": get_utc_now(),
    }

    result = await delete_one(
        collection=mock_collection,
        document_model=MockDocumentModel,
        id=document_id,
        user_id=user_id,
    )

    assert result is not None
    assert result.id == document_id


@pytest.mark.asyncio
async def test_delete_one_no_return():
    user_id = ObjectId()
    document_id = ObjectId()
    mock_collection = AsyncMock()
    mock_collection.find_one.return_value = None

    result = await delete_one(
        collection=mock_collection,
        document_model=MockDocumentModel,
        id=document_id,
        user_id=user_id,
    )

    assert result is None


@pytest.mark.asyncio
async def test_delete_all():
    with pytest.raises(NotImplementedError):
        await delete_all()
