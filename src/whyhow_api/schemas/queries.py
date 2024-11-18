"""Query model and schemas."""

from pydantic import BaseModel, ConfigDict, Field

from whyhow_api.schemas.base import (
    AfterAnnotatedObjectId,
    AnnotatedObjectId,
    BaseDocument,
    BaseResponse,
    Status,
)
from whyhow_api.schemas.nodes import NodeWithId
from whyhow_api.schemas.triples import TripleWithId


class QueryParameters(BaseModel):
    """Query model."""

    content: str | None = Field(
        default=None, description="Query content", min_length=1
    )
    values: list[str] = Field(
        default=[],
        description="A list of entity values (e.g. their content, names) to use for the graph.",
        examples=["Apple", "Tesla", "Mark Zuckerberg"],
    )
    entities: list[str] = Field(
        default=[],
        description="A list of entity types to use for the graph.",
        examples=["Organization", "Person"],
    )
    relations: list[str] = Field(
        default=[],
        description="A list of relations to use for the graph.",
        examples=["founder", "CEO"],
    )
    return_answer: bool = Field(
        default=False,
        description="A boolean specifying whether to return natural language answer or not.",
    )
    include_chunks: bool = Field(
        default=False,
        description="A boolean specifying if to include the chunks in the query or not.",
    )


class QueryDocumentModel(BaseDocument):
    """Query document model."""

    query: QueryParameters = Field(..., description="Content of the query")
    response: str | None = Field(
        default=None,
        description="Response associated with the query",
        min_length=1,
    )
    graph: AfterAnnotatedObjectId = Field(  # type: ignore[assignment]
        default=None, description="Graph id to which the query belongs"
    )
    triples: list[TripleWithId] = Field(
        default=[], description="Triple ids associated with the query"
    )
    nodes: list[NodeWithId] = Field(
        default=[], description="Node ids associated with the query"
    )
    status: Status


class QueryOut(QueryDocumentModel):
    """Query output model."""

    id: AnnotatedObjectId = Field(..., alias="_id")
    created_by: AnnotatedObjectId
    graph: AnnotatedObjectId

    model_config = ConfigDict(
        use_enum_values=True,
        from_attributes=True,
        arbitrary_types_allowed=True,
        populate_by_name=True,
    )


class QueryResponse(BaseResponse):
    """Queries output response model."""

    queries: list[QueryOut] = []

    model_config = ConfigDict(
        use_enum_values=True,
        from_attributes=True,
        arbitrary_types_allowed=True,
    )
