from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from bson import ObjectId
from fastapi import BackgroundTasks
from motor.motor_asyncio import AsyncIOMotorDatabase

from whyhow_api.schemas.tasks import TaskDocumentModel
from whyhow_api.services.crud.task import create_task


@pytest.fixture
def mock_db():
    mock = AsyncMock(spec=AsyncIOMotorDatabase)
    mock.task = AsyncMock()
    mock.task.insert_one = AsyncMock()
    return mock


@pytest.fixture
def mock_background_tasks():
    return MagicMock(spec=BackgroundTasks)


@pytest.fixture
def mock_func():
    return AsyncMock()


@pytest.mark.asyncio
async def test_create_task_success(mock_db, mock_background_tasks, mock_func):
    user_id = ObjectId()
    task_id = ObjectId()

    mock_db.task.insert_one.return_value = AsyncMock(inserted_id=task_id)

    result = await create_task(
        mock_db, user_id, mock_background_tasks, mock_func, arg1="test"
    )

    assert isinstance(result, TaskDocumentModel)
    assert result.id == task_id
    assert result.created_by == user_id
    assert result.status == "pending"

    mock_db.task.insert_one.assert_called_once()
    mock_background_tasks.add_task.assert_called_once_with(
        mock_func, arg1="test", task_id=task_id
    )


@pytest.mark.asyncio
async def test_create_task_with_existing_task_id(
    mock_db, mock_background_tasks, mock_func
):
    user_id = ObjectId()
    existing_task_id = ObjectId()

    mock_db.task.insert_one.return_value = AsyncMock(
        inserted_id=existing_task_id
    )

    result = await create_task(
        mock_db,
        user_id,
        mock_background_tasks,
        mock_func,
        task_id=str(existing_task_id),
    )

    assert result.id == existing_task_id
    mock_db.task.insert_one.assert_called_once()
    inserted_task = mock_db.task.insert_one.call_args[0][0]
    assert inserted_task["_id"] == existing_task_id


@pytest.mark.asyncio
@patch("whyhow_api.services.crud.task.logger")
async def test_create_task_logs_error(
    mock_logger, mock_db, mock_background_tasks, mock_func
):
    user_id = ObjectId()
    error_message = "Database error"
    mock_db.task.insert_one.side_effect = Exception(error_message)

    with pytest.raises(Exception, match=error_message):
        await create_task(mock_db, user_id, mock_background_tasks, mock_func)

    mock_logger.error.assert_called_once_with(
        f"Error creating background task: {error_message}"
    )
