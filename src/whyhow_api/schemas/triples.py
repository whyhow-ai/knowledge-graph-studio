"""Triple schema module."""

from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    model_validator,
)

from whyhow_api.schemas.base import (
    AfterAnnotatedObjectId,
    AnnotatedObjectId,
    BaseDocument,
    BaseResponse,
    Default_Entity_Type,
    Default_Relation_Type,
)
from whyhow_api.schemas.chunks import (
    ChunksOutWithWorkspaceDetails,
    PublicChunksOutWithWorkspaceDetails,
)
from whyhow_api.schemas.nodes import NodeWithId


class TripleDocumentModel(BaseDocument):
    """Triple model for DB object.

    Semantic triples are extracted from chunk by users to form part of graph.
    """

    head_node: AfterAnnotatedObjectId
    tail_node: AfterAnnotatedObjectId
    type: str = Field(
        default=Default_Relation_Type,
        description="Relation type of the triple",
        min_length=1,
    )
    properties: dict[str, Any] = {}
    chunks: list[AfterAnnotatedObjectId] = []
    graph: AfterAnnotatedObjectId | None
    embedding: list[float] | None = None


class TripleCreateNode(BaseModel):
    """Triple node model for POST body."""

    name: str = Field(..., description="Name of the node", min_length=1)
    type: str = Field(
        default=Default_Entity_Type,
        description="Type of the node",
        min_length=1,
    )
    properties: dict[str, Any] = Field(
        default={}, description="Properties of the node"
    )


class TripleCreate(BaseModel):
    """Triple model for POST body."""

    head_node: TripleCreateNode | AnnotatedObjectId
    tail_node: TripleCreateNode | AnnotatedObjectId
    type: str = Field(
        default=Default_Relation_Type,
        description="Relation type of the triple",
        min_length=1,
    )
    properties: dict[str, Any] = Field(
        default={}, description="Properties of the triple"
    )
    chunks: list[AnnotatedObjectId] = Field(
        default=[],
        description="Chunk ids to which the triple is associated with",
    )

    @model_validator(mode="before")
    def check_nodes(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Check if the nodes are in the correct format."""
        head_node = values.get("head_node")
        if isinstance(head_node, dict):
            try:
                values["head_node"] = TripleCreateNode(**head_node)
            except ValidationError as e:
                raise ValueError(f"Invalid head node format: {e}")

        tail_node = values.get("tail_node")
        if isinstance(tail_node, dict):
            try:
                values["tail_node"] = TripleCreateNode(**tail_node)
            except ValidationError as e:
                raise ValueError(f"Invalid tail node format: {e}")
        return values


class TriplesCreate(BaseModel):
    """Schema for the request body of the triples endpoint."""

    graph: AnnotatedObjectId
    strict_mode: bool = Field(
        default=False,
        description="Strict mode for triple creation. If True, triple validation will be performed. If False, invalid triples will be used to extend the graph's schema.",
    )
    triples: list[TripleCreate]


class TripleUpdate(BaseModel):
    """Triple model for PUT body."""

    type: str | None = Field(
        default=None, description="Relation type of the triple", min_length=1
    )
    properties: dict[str, Any] | None = Field(
        default=None, description="Properties of the triple"
    )


class RelationOut(BaseModel):
    """Relation response model for Triples."""

    name: str = Field(..., description="Name of the relation", min_length=1)
    properties: dict[str, Any] = Field(
        default={}, description="Properties of the relation"
    )


class TripleWithId(BaseModel):
    """Triple response model containing ids for triple, head, and tail."""

    id: AnnotatedObjectId = Field(..., alias="_id")
    head_node: NodeWithId
    relation: RelationOut
    tail_node: NodeWithId
    chunks: list[AnnotatedObjectId] = Field(
        default=[], description="Chunk ids to which the triple was found in"
    )


class PublicTripleWithId(BaseModel):
    """Public triple response model containing ids for triple, head, and tail."""

    head_node: NodeWithId
    relation: RelationOut
    tail_node: NodeWithId


class TripleOut(TripleDocumentModel):
    """Triple response model."""

    id: AnnotatedObjectId = Field(..., alias="_id")
    head_node: AnnotatedObjectId
    tail_node: AnnotatedObjectId
    type: str = Field(
        ..., description="Relation type of the triple", min_length=1
    )
    properties: dict[str, Any] = Field(
        ..., description="Properties of the triple"
    )
    graph: AnnotatedObjectId
    created_by: AnnotatedObjectId
    chunks: list[AnnotatedObjectId] = Field(  # type: ignore[assignment]
        default=[], description="Chunk ids to which the triple was found in"
    )

    model_config = ConfigDict(
        use_enum_values=True, from_attributes=True, populate_by_name=True
    )


class TriplesResponse(BaseResponse):
    """Schema for the response body of the triple endpoint."""

    triples: list[TripleOut]


class TripleChunksResponse(BaseResponse):
    """Schema for the response body of the triple chunks endpoint."""

    chunks: list[ChunksOutWithWorkspaceDetails]


class PublicTripleChunksResponse(BaseResponse):
    """Schema for the response body of the public graph triple chunks endpoint."""

    chunks: list[PublicChunksOutWithWorkspaceDetails] = []
