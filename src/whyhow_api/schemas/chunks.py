"""Data models and schemas."""

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

from whyhow_api.config import Settings
from whyhow_api.schemas.base import (
    AfterAnnotatedObjectId,
    AllowedChunkContentTypes,
    AllowedUserMetadataTypes,
    AnnotatedObjectId,
    BaseAssignmentModel,
    BaseDocument,
    BaseResponse,
    BaseUnassignmentModel,
    Chunk_Data_Type,
)
from whyhow_api.schemas.documents import DocumentDetail
from whyhow_api.schemas.workspaces import WorkspaceDetails

settings = Settings()


class ChunkMetadata(BaseModel):
    """Chunk metadata model."""

    language: str = Field(default="en", description="Language of the chunk")
    length: int | None = Field(
        default=None,
        description="Length of the chunk in characters if string or keys if object",
    )
    size: int | None = Field(
        default=None, description="Size of the chunk in bytes"
    )
    data_source_type: Literal["manual", "automatic", "external"] | None = (
        Field(default=None, description="Source of how the chunk was created")
    )

    index: int | None = Field(
        default=None,
        description="Index of the chunk in csv and json documents",
    )
    page: int | None = Field(
        default=None,
        description="Page number of the chunk in pdf and txt documents",
    )
    start: int | None = Field(
        default=None,
        description="Start position of the chunk in pdf and txt documents",
    )
    end: int | None = Field(
        default=None,
        description="End position of the chunk in pdf and txt documents",
    )


class ChunkDocumentModel(BaseDocument):
    """Chunk document model."""

    workspaces: list[AfterAnnotatedObjectId] = Field(
        ..., description="list of workspaces chunk is assigned to."
    )
    document: AfterAnnotatedObjectId | None = Field(
        default=None, description="Document id associated with the chunk"
    )
    data_type: Chunk_Data_Type = Field(
        ..., description="Type of the content in the chunk"
    )
    content: str | dict[str, AllowedChunkContentTypes] = Field(
        ..., description="Content of the chunk", min_length=1
    )
    embedding: list[float] | None = Field(
        default=None, description="Embedding of the chunk"
    )
    metadata: ChunkMetadata
    tags: dict[str, list[str]] = {}
    user_metadata: (
        dict[
            str,
            dict[
                str, AllowedUserMetadataTypes | list[AllowedUserMetadataTypes]
            ],
        ]
        | None
    ) = Field(default={}, description="User defined metadata")


class ChunkOut(ChunkDocumentModel):
    """API Response model."""

    id: AnnotatedObjectId = Field(..., alias="_id")
    created_by: AnnotatedObjectId
    workspaces: list[AnnotatedObjectId]  # type: ignore[assignment]
    document: AnnotatedObjectId | None = Field(
        default=None, description="Document id associated with the chunk"
    )
    tags: dict[str, list[str]] | list[str]  # type: ignore[assignment]
    user_metadata: (  # type: ignore[assignment]
        dict[
            str,
            dict[
                str, AllowedUserMetadataTypes | list[AllowedUserMetadataTypes]
            ],
        ]
        | dict[str, AllowedUserMetadataTypes | list[AllowedUserMetadataTypes]]
    )

    model_config = ConfigDict(use_enum_values=True, from_attributes=True)


class ChunksOutWithWorkspaceDetails(ChunkOut):
    """API Response model with workspace details."""

    workspaces: list[WorkspaceDetails]  # type: ignore[assignment]
    document: DocumentDetail | None = None  # type: ignore[assignment]


class PublicChunksOutWithWorkspaceDetails(ChunksOutWithWorkspaceDetails):
    """Public API Response model."""

    @model_validator(mode="after")
    def obfuscate_names(self) -> Self:
        """Obfuscate the chunk's workspaces and document names."""
        for w in self.workspaces:
            w.name = "hidden"
        if self.document:
            self.document.filename = "hidden"
        return self


class AddChunkModel(BaseModel):
    """API POST body model."""

    content: str | dict[str, AllowedChunkContentTypes] = Field(
        ..., description="Content of the chunk", min_length=1
    )
    user_metadata: (
        dict[str, AllowedUserMetadataTypes | list[AllowedUserMetadataTypes]]
        | None
    ) = Field(default=None, description="User defined metadata")
    tags: list[str] | None = None

    @model_validator(mode="after")
    def check_content_length(self) -> Self:
        """Check if the content is longer than allowed length."""
        max_length = settings.api.max_chars_per_chunk
        content_to_check = (
            self.content
            if isinstance(self.content, str)
            else json.dumps(self.content)
        )

        if len(content_to_check) > max_length:
            raise ValueError(
                f"Content length of {len(content_to_check)} exceeds the maximum allowed length of {max_length} characters. "
                "If content is a JSON object, the entire structure (including JSON characters like "
                "braces, commas, and quotes) will be counted towards the length."
            )
        return self


class AddChunksModel(BaseModel):
    """API POST body model for multiple chunks."""

    chunks: list[AddChunkModel]


class UpdateChunkModel(BaseModel):
    """API PUT body model."""

    user_metadata: (
        dict[str, AllowedUserMetadataTypes | list[AllowedUserMetadataTypes]]
        | None
    ) = Field(default=None, description="User defined metadata to update")
    tags: list[str] | None = Field(
        default=None, description="List of tags to update"
    )


class ChunksResponse(BaseResponse):
    """Schema for the response body of the chunks endpoints."""

    chunks: list[ChunkOut]


class ChunksResponseWithWorkspaceDetails(BaseResponse):
    """Schema for the response body of the chunks endpoints with workspace details."""

    chunks: list[ChunksOutWithWorkspaceDetails] = Field(
        default=[], description="list of chunks"
    )


class PublicChunksResponseWithWorkspaceDetails(
    ChunksResponseWithWorkspaceDetails
):
    """Schema for the response body of the public chunks endpoints with workspace details."""

    @model_validator(mode="after")
    def obfuscate_names(self) -> Self:
        """Obfuscate the chunk's workspaces and document names."""
        for chunk in self.chunks:
            for w in chunk.workspaces:
                w.name = "hidden"
            if chunk.document:
                chunk.document.filename = "hidden"
        return self


class ChunkUnassignments(BaseUnassignmentModel):
    """Chunk unassignments model."""

    pass


class ChunkAssignments(BaseAssignmentModel):
    """Chunk assignments model."""

    pass


class ChunkAssignmentResponse(BaseResponse):
    """Schema for the response body of the assign chunks endpoint."""

    chunks: ChunkAssignments


class ChunkUnassignmentResponse(BaseResponse):
    """Schema for the response body of the unassign chunks endpoint."""

    chunks: ChunkUnassignments


class AddChunksResponse(BaseResponse):
    """Schema for the response body of the add chunks endpoint."""

    chunks: list[ChunkOut]
