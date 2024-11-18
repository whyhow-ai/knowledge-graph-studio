"""Document models and schemas."""

from pydantic import BaseModel, ConfigDict, Field

from whyhow_api.schemas.base import (
    AfterAnnotatedObjectId,
    AllowedUserMetadataTypes,
    AnnotatedObjectId,
    BaseAssignmentModel,
    BaseDocument,
    BaseResponse,
    BaseUnassignmentModel,
    Document_Status,
    ErrorDetails,
    File_Extensions,
)
from whyhow_api.schemas.workspaces import WorkspaceDetails


class DocumentInfo(BaseModel):
    """Document info model.

    This model is used to store information that is parsed by `UploadFile`  and
    includes user fields.
    """

    content: bytes = Field(..., description="Content of the document in bytes")
    filename: str = Field(..., description="Filename of the document")
    content_type: str
    size: int = Field(..., description="Size of the document", examples=[1234])
    tags: list[str] | None = None
    user_metadata: (
        dict[
            str,
            dict[
                str, AllowedUserMetadataTypes | list[AllowedUserMetadataTypes]
            ],
        ]
        | None
    ) = None


class DocumentMetadata(BaseModel):
    """Document metadata model."""

    size: int = Field(..., description="Size of the document", examples=[1234])
    format: File_Extensions = Field(
        ...,
        description="Format of the document",
        examples=["pdf", "csv", "json", "txt"],
        min_length=1,
    )
    filename: str = Field(
        ..., description="Filename of the document", min_length=1
    )


class DocumentDocumentModel(BaseDocument):
    """Document document model."""

    workspaces: list[AfterAnnotatedObjectId] = Field(
        ..., description="Workspace IDs document is assigned to."
    )
    status: Document_Status
    errors: list[ErrorDetails] = Field(
        default=[],
        description="Details about the error that occurred during document processing.",
    )
    metadata: DocumentMetadata
    tags: dict[str, list[str]] = Field(default={}, description="List of tags")
    user_metadata: dict[
        str,
        dict[str, AllowedUserMetadataTypes | list[AllowedUserMetadataTypes]],
    ] = Field(default={}, description="User defined metadata")


class DocumentOut(DocumentDocumentModel):
    """Document response model."""

    id: AnnotatedObjectId = Field(..., alias="_id", description="Document ID")
    created_by: AnnotatedObjectId
    workspaces: list[AnnotatedObjectId]  # type: ignore[assignment]

    model_config = ConfigDict(
        use_enum_values=True, from_attributes=True, populate_by_name=True
    )


class DocumentOutWithWorkspaceDetails(DocumentOut):
    """Document response model with workspace details."""

    workspaces: list[WorkspaceDetails]  # type: ignore[assignment]


class DocumentUpdate(BaseModel):
    """Document model for PUT body."""

    user_metadata: (
        dict[str, AllowedUserMetadataTypes | list[AllowedUserMetadataTypes]]
        | None
    ) = Field(
        default=None,
        description="User supplied metadata to update",
    )
    tags: list[str] | None = Field(
        default=None, description="List of tags to update"
    )


class DocumentDetail(BaseModel):
    """Schema for document details."""

    id: AnnotatedObjectId = Field(..., alias="_id")
    filename: str = Field(
        ..., description="Filename of the document", min_length=1
    )


class DocumentsResponse(BaseResponse):
    """Schema for the response body of the documents endpoints."""

    documents: list[DocumentOut] = Field(
        default=[], description="list of documents"
    )


class DocumentsResponseWithWorkspaceDetails(BaseResponse):
    """Schema for the response body of the documents endpoints with workspace details."""

    documents: list[DocumentOutWithWorkspaceDetails] = Field(
        default=[], description="list of documents"
    )


class DocumentUnassignments(BaseUnassignmentModel):
    """Document unassignments model."""

    pass


class DocumentAssignments(BaseAssignmentModel):
    """Document assignments model."""

    pass


class DocumentAssignmentResponse(BaseResponse):
    """Schema for the response body of the assign documents endpoint."""

    documents: DocumentAssignments


class DocumentUnassignmentResponse(BaseResponse):
    """Schema for the response body of the unassign documents endpoint."""

    documents: DocumentUnassignments


class GeneratePresignedRequest(BaseModel):
    """Request model for generating a presigned post."""

    filename: str = Field(
        ..., description="Filename of the document", min_length=1
    )
    workspace_id: AnnotatedObjectId


class GeneratePresignedResponse(BaseModel):
    """Response model for generating a presigned post."""

    url: str
    fields: dict[str, str]


class GeneratePresignedDownloadRequest(BaseModel):
    """Request model for generating a presigned download url."""

    filename: str = Field(
        ..., description="Filename of the document", min_length=1
    )


class GeneratePresignedDownloadResponse(BaseModel):
    """Response model for generating a presigned download url."""

    url: str = Field(
        ..., description="Download URL of the document", min_length=1
    )


class DocumentStateErrorsUpdate(BaseModel):
    """Model for updating the state and errors of a document."""

    status: Document_Status = Field(..., description="Status of the document")
    errors: list[ErrorDetails] = Field(
        default=[],
        description="Details about the error that occurred during document processing.",
    )
