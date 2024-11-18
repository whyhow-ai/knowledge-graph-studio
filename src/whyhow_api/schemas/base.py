"""Base classes for request, response, and return schemas."""

from abc import ABC
from datetime import datetime, timezone
from typing import Annotated, Any, Literal

from bson import ObjectId
from bson import errors as BsonErrors
from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    field_validator,
)

# Custom types
AnnotatedObjectId = Annotated[str, BeforeValidator(lambda x: str(x))]
AllowedUserMetadataTypes = str | int | bool | float
AllowedChunkContentTypes = str | int | bool | float | None
AllowedPropertyTypes = str | int | bool | float | None

Status = Literal["success", "pending", "failed"]
Graph_Status = Literal["creating", "updating", "ready", "failed"]
Document_Status = Literal["uploaded", "processing", "processed", "failed"]
Chunk_Data_Type = Literal["string", "object"]
Default_Entity_Type = "entity"
Default_Relation_Type = "related_to"
File_Extensions = Literal["csv", "json", "pdf", "txt"]
Rule_Type = Literal["merge_nodes"]
TaskStatus = Literal["pending", "success", "failed"]


def validate_object_id(value: str) -> ObjectId:
    """Validate the ObjectId."""
    try:
        return ObjectId(value)
    except BsonErrors.InvalidId as e:
        raise ValueError(f"{value} is not a valid ObjectId") from e


AfterAnnotatedObjectId = Annotated[
    str | ObjectId,
    BeforeValidator(lambda x: str(x)),
    AfterValidator(validate_object_id),
]


def get_utc_now() -> datetime:
    """Get the current time in UTC."""
    return datetime.now(timezone.utc)


Error_Level = Literal["error", "critical"]


class ErrorDetails(BaseModel):
    """Model for holding details about an error or other message types."""

    message: str = Field(
        ...,
        description="The error message detailing what went wrong.",
        min_length=10,
    )
    created_at: datetime = Field(
        default_factory=get_utc_now,
        description="The UTC timestamp when the error was logged.",
    )
    level: Error_Level = Field(
        ..., description="The severity level of the error."
    )


class FilterBody(BaseModel):
    """Filter body for query operations."""

    filters: dict[str, Any] | None = None


class DeleteResponseModel(BaseModel):
    """Response model for delete operations."""

    message: str
    status: Status


class BaseDocument(BaseModel):
    """Base class for all mongodb documents."""

    id: AfterAnnotatedObjectId | None = Field(default=None, alias="_id")
    created_at: datetime = Field(default_factory=get_utc_now)
    updated_at: datetime = Field(default_factory=get_utc_now)
    created_by: AfterAnnotatedObjectId

    model_config = ConfigDict(
        populate_by_name=True,
        use_enum_values=True,
        from_attributes=True,
        arbitrary_types_allowed=True,
    )

    @field_validator("id", "created_by", mode="before")
    def validate_object_id(cls, v) -> ObjectId | str | None:  # type: ignore[no-untyped-def]
        """Validate the ObjectId and convert it to a string if necessary."""
        if v is not None:
            try:
                return ObjectId(v)
            except BsonErrors.InvalidId:
                raise ValueError("Invalid ObjectId")
        return v


class BaseRequest(BaseModel, ABC):
    """Base class for all request schemas."""

    model_config = ConfigDict(extra="forbid")


class BaseResponse(BaseModel, ABC):
    """Base class for all response schemas.

    Since the API can change, we want to ignore any extra fields that are not
    defined in the schema.
    """

    message: str
    status: Status
    count: int = 0

    model_config = ConfigDict(extra="ignore")


class BaseUnassignmentModel(BaseModel):
    """Base unassignments model."""

    unassigned: list[str]
    not_found: list[str]
    not_found_in_workspace: list[str]


class BaseAssignmentModel(BaseModel):
    """Base assignments model."""

    assigned: list[str]
    not_found: list[str]
    already_assigned: list[str]
