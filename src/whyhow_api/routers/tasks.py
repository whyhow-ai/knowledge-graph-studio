"""Task CRUD router."""

import bson
from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException
from motor.motor_asyncio import AsyncIOMotorDatabase

from whyhow_api.dependencies import get_db, get_user
from whyhow_api.schemas.tasks import TaskOut, TaskResponse

router = APIRouter(tags=["Tasks"], prefix="/tasks")


@router.get("/{task_id}", response_model=TaskResponse)
async def get_task(
    task_id: str,
    db: AsyncIOMotorDatabase = Depends(get_db),
    user_id: ObjectId = Depends(get_user),
) -> TaskResponse:
    """Get a task by ID."""
    try:
        task_obj = await db.task.find_one(
            {"_id": ObjectId(task_id), "created_by": user_id}
        )
        if task_obj is None:
            raise HTTPException(status_code=404, detail="Task not found.")
        task = TaskOut.model_validate(task_obj)
        task.id = str(task.id)
        task.created_by = str(task.created_by)
        return TaskResponse(
            message="Task retrieved successfully.",
            status="success",
            task=task,
            count=1,
        )
    except bson.errors.InvalidId:
        raise HTTPException(status_code=400, detail="Invalid task ID.")
