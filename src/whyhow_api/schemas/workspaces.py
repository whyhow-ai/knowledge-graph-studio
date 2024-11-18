"""Workspace and chunk models and schemas."""

from pydantic import BaseModel, ConfigDict, Field

from whyhow_api.schemas.base import (
    AnnotatedObjectId,
    BaseDocument,
    BaseResponse,
)


class WorkspaceDetails(BaseModel):
    """Schema for workspace details."""

    id: AnnotatedObjectId = Field(..., alias="_id", description="Workspace ID")
    name: str = Field(..., description="Name of the workspace", min_length=1)


class WorkspaceDocumentModel(BaseDocument):
    """Workspace document model."""

    name: str = Field(..., description="Name of the workspace", min_length=1)


class WorkspaceCreate(BaseModel):
    """API POST body model."""

    name: str = Field(..., description="Name of the workspace", min_length=1)


class WorkspaceUpdate(BaseModel):
    """API PUT body model."""

    name: str | None = Field(
        default=None, description="Name of the workspace", min_length=1
    )


class WorkspaceOut(WorkspaceDocumentModel):
    """API Response model."""

    id: AnnotatedObjectId = Field(..., alias="_id", description="Workspace ID")
    created_by: AnnotatedObjectId = Field(
        ..., description="Id of the user who created the workspace"
    )

    model_config = ConfigDict(use_enum_values=True, from_attributes=True)


class WorkspacesResponse(BaseResponse):
    """Schema for the response body of the workspaces endpoints."""

    workspaces: list[WorkspaceOut]


class WorkspaceTagsResponse(BaseResponse):
    """Schema for the response body of the workspace tags endpoints."""

    workspace_id: AnnotatedObjectId
    tags: list[str]


class WorkspaceTagsOut(BaseModel):
    """Workspace tags API response model."""

    tags: list[str]
