"""Task schemas."""

from datetime import datetime

from pydantic import Field

from whyhow_api.schemas.base import (
    BaseDocument,
    BaseResponse,
    TaskStatus,
    get_utc_now,
)


class TaskDocumentModel(BaseDocument):
    """Task schema."""

    start_time: datetime = Field(default_factory=get_utc_now)
    end_time: datetime | None = None
    status: TaskStatus = Field(..., description="Status of task")
    result: str | None = None


class TaskOut(TaskDocumentModel):
    """Task schema for response."""

    pass


class TaskResponse(BaseResponse):
    """Task response schema."""

    task: TaskOut
