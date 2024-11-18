"""Task CRUD operations."""

import logging
from typing import Any, Callable, Coroutine

from bson import ObjectId
from fastapi import BackgroundTasks
from motor.motor_asyncio import AsyncIOMotorDatabase

from whyhow_api.schemas.tasks import TaskDocumentModel

logger = logging.getLogger(__name__)


async def create_task(
    _db: AsyncIOMotorDatabase,
    _user_id: ObjectId,
    _background_tasks: BackgroundTasks,
    func: Callable[..., Coroutine[Any, Any, None]],
    *args: Any,
    **kwargs: Any,
) -> TaskDocumentModel:
    """Create a background task."""
    logger.info(f"Creating background task: {func.__name__}")
    logger.debug(f"Args: {args}")
    logger.debug(f"Kwargs: {kwargs}")
    try:
        task_id = kwargs.pop("task_id", None)
        task_id = ObjectId(task_id) if task_id else None
        task = TaskDocumentModel(
            id=task_id,
            created_by=_user_id,
            status="pending",
        )
        result = await _db.task.insert_one(
            task.model_dump(by_alias=True, exclude_none=True)
        )
        task.id = ObjectId(result.inserted_id)
        kwargs["task_id"] = ObjectId(result.inserted_id)
        _background_tasks.add_task(func, *args, **kwargs)
        return task
    except Exception as e:
        logger.error(f"Error creating background task: {e}")
        raise e
