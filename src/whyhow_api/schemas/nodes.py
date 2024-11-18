"""Node part of semantic triple created from chunk by user to form part of graph."""

from typing import Any

from pydantic import ConfigDict, Field

from whyhow_api.schemas.base import (
    AfterAnnotatedObjectId,
    AllowedPropertyTypes,
    AnnotatedObjectId,
    BaseDocument,
    BaseModel,
    BaseResponse,
    Default_Entity_Type,
)
from whyhow_api.schemas.chunks import ChunksOutWithWorkspaceDetails


class NodeWithId(BaseModel):
    """Schema for a node which includes the id in the output."""

    id: AnnotatedObjectId = Field(..., alias="_id")
    name: str = Field(..., description="Name of the node", min_length=1)
    label: str | None = Field(
        ..., description="Label of the node", min_length=1
    )
    properties: dict[
        str, AllowedPropertyTypes | list[AllowedPropertyTypes]
    ] = Field(default={}, description="Properties of the node")
    chunks: list[AnnotatedObjectId] = Field(
        default=[], description="Chunk ids to which the node was found in"
    )


class NodeWithIdAndSimilarity(NodeWithId):
    """Schema for a node which includes the id and similarity in the output."""

    similarity: float = Field(..., description="Similarity of the node")


class NodeDocumentModel(BaseDocument):
    """Node part of semantic triple created from chunk by user to form part of graph."""

    name: str = Field(..., description="Name of the node", min_length=1)
    type: str = Field(
        default=Default_Entity_Type,
        description="Type of the node",
        min_length=1,
    )
    properties: dict[
        str, AllowedPropertyTypes | list[AllowedPropertyTypes]
    ] = Field(default={}, description="Properties of the node")
    graph: AfterAnnotatedObjectId | None = Field(
        ..., description="Graph id to which the node belongs"
    )
    chunks: list[AfterAnnotatedObjectId] = Field(
        default=[], description="Chunk ids to which the node was found in"
    )

    def __str__(self) -> str:
        """Return string representation."""
        return f"{self.name} ({self.type})"


class NodeCreate(BaseModel):
    """Node model for POST body."""

    name: str = Field(..., description="Name of the node", min_length=1)
    type: str = Field(
        ...,
        description="Type of the node",
        min_length=1,
    )
    properties: dict[
        str, AllowedPropertyTypes | list[AllowedPropertyTypes]
    ] = Field(default={}, description="Properties of the node")
    graph: AfterAnnotatedObjectId = Field(
        ..., description="Graph id to which the node belongs"
    )
    chunks: list[AnnotatedObjectId] = Field(
        default=[],
        description="Chunk ids to which the node is associated with",
    )
    strict_mode: bool = Field(
        default=False,
        description="Strict mode for node creation. If True, node validation will be performed. If False, invalid node will be used to extend the graph's schema.",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)


class NodeUpdate(BaseModel):
    """Node model for PUT body."""

    name: str | None = Field(
        default=None, description="Name of the node", min_length=1
    )
    type: str | None = Field(
        default=None, description="Type of the node", min_length=1
    )
    properties: dict[str, Any] | None = Field(
        default=None, description="Properties of the node"
    )
    graph: AfterAnnotatedObjectId | None = Field(
        default=None, description="Graph id to which the node belongs"
    )
    chunks: list[AfterAnnotatedObjectId] | None = Field(
        default=[],
        description="Chunk ids to which the node is associated with",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __str__(self) -> str:
        """Return string representation."""
        return f"{self.name} ({self.type})"


class NodeOut(NodeDocumentModel):
    """Node response model."""

    id: AnnotatedObjectId = Field(..., alias="_id")
    name: str = Field(..., description="Name of the node", min_length=1)
    type: str = Field(..., description="Type of the node", min_length=1)
    properties: dict[str, Any] = Field(
        ..., description="Properties of the node"
    )
    graph: AnnotatedObjectId = Field(
        ..., description="Graph id to which the node belongs"
    )
    created_by: AnnotatedObjectId = Field(
        ..., description="User who created the node"
    )
    chunks: list[AnnotatedObjectId] = Field(  # type: ignore[assignment]
        ..., description="Chunk ids to which the node was found in"
    )
    model_config = ConfigDict(
        use_enum_values=True, from_attributes=True, populate_by_name=True
    )

    def __str__(self) -> str:
        """Return string representation."""
        return f"{self.name} ({self.type})"


class NodesResponse(BaseResponse):
    """Schema for the response body of the node endpoint."""

    nodes: list[NodeOut]


class NodeChunksResponse(BaseResponse):
    """Schema for the response body of the node chunks endpoint."""

    chunks: list[ChunksOutWithWorkspaceDetails]
