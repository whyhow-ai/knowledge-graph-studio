"""Graphs models and schemas."""

from typing import Annotated, Any

from annotated_types import Len
from bson import ObjectId
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

from whyhow_api.models.common import Node, Triple
from whyhow_api.schemas.base import (
    AfterAnnotatedObjectId,
    AnnotatedObjectId,
    BaseDocument,
    BaseRequest,
    BaseResponse,
    Chunk_Data_Type,
    ErrorDetails,
    Graph_Status,
)
from whyhow_api.schemas.nodes import NodeWithId, NodeWithIdAndSimilarity
from whyhow_api.schemas.queries import QueryOut
from whyhow_api.schemas.schemas import SchemaDetails
from whyhow_api.schemas.triples import TripleWithId
from whyhow_api.schemas.workspaces import WorkspaceDetails


class GraphDocumentModel(BaseDocument):
    """Graph document model."""

    name: str = Field(..., description="Name of the graph", min_length=1)
    workspace: AfterAnnotatedObjectId
    schema_: AfterAnnotatedObjectId | None = Field(None, alias="schema_id")
    status: Graph_Status = Field(..., description="Status of the graph")
    errors: list[ErrorDetails] = Field(
        default=[],
        description="Details about the error that occurred during graph creation.",
    )
    public: bool = Field(
        False,
        description="Whether the graph is public or not",
    )

    def __str__(self) -> str:
        """Return a string representation of the graph."""
        return f"""{self.name}
        (id: {self.id},
        workspace: {self.workspace},
        schema: {self.schema_})"""


class DetailedGraphDocumentModel(BaseDocument):
    """Graph document model."""

    name: str = Field(..., description="Name of the graph", min_length=1)
    workspace: WorkspaceDetails
    schema_: SchemaDetails = Field(..., alias="schema")
    status: Graph_Status = Field(..., description="Status of the graph")
    public: bool = Field(..., description="Whether the graph is public or not")
    errors: list[ErrorDetails] = Field(
        default=[],
        description="Details about the error that occurred during graph creation.",
    )

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )


class GraphStateErrorsUpdate(BaseModel):
    """Model for updating the state and errors of a graph."""

    status: Graph_Status = Field(..., description="Status of the graph")
    errors: list[ErrorDetails] = Field(
        default=[],
        description="Details about the error that occurred during graph creation.",
    )


class GraphUpdate(BaseModel):
    """Graph model for PUT body."""

    name: str | None = Field(
        default=None, description="Name of the graph", min_length=1
    )
    public: bool | None = Field(
        default=None, description="Whether the graph is public or not"
    )


class GraphOut(GraphDocumentModel):
    """Graph response model."""

    id: AnnotatedObjectId = Field(..., alias="_id")
    created_by: AnnotatedObjectId
    workspace: AnnotatedObjectId = Field(..., alias="workspace_id")
    schema_: AnnotatedObjectId | None = Field(default=None, alias="schema_id")

    model_config = ConfigDict(
        use_enum_values=True, from_attributes=True, populate_by_name=True
    )


class DetailedGraphOut(DetailedGraphDocumentModel):
    """Graph response model."""

    id: AnnotatedObjectId = Field(..., alias="_id")
    created_by: AnnotatedObjectId
    workspace: WorkspaceDetails
    schema_: SchemaDetails = Field(..., alias="schema")
    public: bool = Field(..., description="Whether the graph is public or not")

    model_config = ConfigDict(
        use_enum_values=True,
        from_attributes=True,
        populate_by_name=True,
        arbitrary_types_allowed=True,
    )


class PublicDetailedGraphOut(DetailedGraphOut):
    """Public graph response model."""

    @model_validator(mode="before")
    @classmethod
    def obfuscate_names(cls, v: Any) -> Any:
        """Obfuscate the graphs's workspace and schema names."""
        v["schema"]["name"] = "hidden"
        v["workspace"]["name"] = "hidden"
        return v


class GraphsResponse(BaseResponse):
    """Schema for the response body of the graphs endpoints."""

    query_id: str | None = None
    query: str | None = None
    graphs: list[GraphOut]
    nodes: list[Node] | None = None
    relations: list[str] | None = None
    triples: list[Triple] | None = None
    count: int
    answer: str | None = None


class DetailedGraphsResponse(BaseResponse):
    """Schema for the response body of the graphs endpoints."""

    graphs: list[DetailedGraphOut]
    queries: list[QueryOut] | None = None
    relations: list[str] | None = None
    nodes: list[NodeWithId] | None = None


class PublicGraphsResponse(DetailedGraphsResponse):
    """Schema for the response body of the public graphs endpoints."""

    @model_validator(mode="after")
    @classmethod
    def obfuscate_names(cls, v: Any) -> Any:
        """Obfuscate the names of the graph's schema and workspace."""
        for g in v.graphs:
            g.schema_.name = "hidden"
            g.workspace.name = "hidden"
        return v


class GraphsDetailedNodeResponse(BaseResponse):
    """Schema for the response body of the graphs nodes endpoints."""

    graphs: list[DetailedGraphOut]
    nodes: list[NodeWithId] | None = None


class PublicGraphsDetailedNodeResponse(GraphsDetailedNodeResponse):
    """Schema for the response body of the public graphs nodes endpoints."""

    @model_validator(mode="after")
    @classmethod
    def obfuscate_names(cls, v: Any) -> Any:
        """Obfuscate the names of the graph's schema and workspace."""
        for g in v.graphs:
            g.schema_.name = "hidden"
            g.workspace.name = "hidden"
        return v


class GraphsSimilarNodesResponse(BaseResponse):
    """Schema for the response body of the graphs endpoints."""

    graphs: list[DetailedGraphOut]
    similar_nodes: list[list[NodeWithIdAndSimilarity]]


# Request and response schemas
class GraphsDetailedTripleResponse(BaseResponse):
    """Schema for the response body of the graphs endpoints."""

    graphs: list[DetailedGraphOut]
    triples: list[TripleWithId] | None = None


class PublicGraphsTripleResponse(GraphsDetailedTripleResponse):
    """Schema for the response body of the graphs endpoints."""

    @model_validator(mode="after")
    @classmethod
    def obfuscate_names(cls, v: Any) -> Any:
        """Obfuscate the names of the graph's schema and workspace."""
        for g in v.graphs:
            g.schema_.name = "hidden"
            g.workspace.name = "hidden"
        return v


class ChunkFilters(BaseModel):
    """Filters for chunk retrieval."""

    document_ids: list[AnnotatedObjectId] | None = Field(
        default=None, description="Document IDs that contain chunks."
    )
    data_types: list[Chunk_Data_Type] | None = Field(
        default=None, description="Data types of chunks."
    )
    tags: list[str] | None = Field(
        default=None, description="Tags for chunks."
    )
    user_metadata: dict[str, Any] | None = None
    ids: list[AnnotatedObjectId] | None = Field(
        default=None, description="IDs of the chunks."
    )
    # seed_concept: str | None = Field(
    #     default=None,
    #     description=(
    #         "A seed concept for similarity search during chunk retrieval."
    #     ),
    #     min_length=1,
    # )

    @property
    def mql_filter(self) -> dict[str, Any]:
        """Return the MQL filter for chunk retrieval."""
        mql_filter = {}
        if self.data_types:
            mql_filter["data_type"] = {"$in": self.data_types}
        if self.tags:
            mql_filter["tags"] = {"$in": self.tags}  # type: ignore[dict-item]
        if self.user_metadata:
            mql_filter["user_metadata"] = self.user_metadata
        if self.ids:
            mql_filter["_id"] = {"$in": [ObjectId(i) for i in self.ids]}  # type: ignore[misc]
        if self.document_ids:
            mql_filter["document"] = {
                "$in": [ObjectId(i) for i in self.document_ids]  # type: ignore[misc]
            }
        # if self.seed_concept:
        #     mql_filter["seed_concept"] = self.seed_concept  # type: ignore[assignment]
        return mql_filter


class CreateGraphBody(BaseRequest):
    """Schema for creating a graph."""

    name: str = Field(..., description="The name of the graph.", min_length=1)
    workspace: AfterAnnotatedObjectId = Field(
        ...,
        description=(
            "The ID of the workspace for the graph. This is used for chunk"
            " retrieval."
        ),
    )
    schema_: AfterAnnotatedObjectId = Field(
        description="The ID of the schema to guide the graph creation.",
        alias="schema",
    )
    filters: ChunkFilters | None = Field(
        default=None,
        description="Filters to apply to the chunk retrieval. If not provided, all chunks will be used.",
    )

    model_config = ConfigDict(
        use_enum_values=True,
        from_attributes=True,
        populate_by_name=True,
        arbitrary_types_allowed=True,
    )


class AddChunksToGraphBody(BaseRequest):
    """Schema for creating a graph."""

    graph: AnnotatedObjectId
    filters: ChunkFilters | None = Field(
        default=None,
        description="Filters to apply to the chunk retrieval. If not provided, all chunks will be used.",
    )

    model_config = ConfigDict(
        use_enum_values=True,
        from_attributes=True,
        populate_by_name=True,
        arbitrary_types_allowed=True,
    )


class CreateGraphFromTriplesBody(BaseRequest):
    """Schema for creating a graph from triples."""

    name: str = Field(..., description="The name of the graph.", min_length=1)
    workspace: AfterAnnotatedObjectId
    schema_: AfterAnnotatedObjectId | None = Field(
        default=None,
        alias="schema",
    )
    triples: Annotated[list[Triple], Len(min_length=1)] = Field(
        ...,
        description="The triples to create the graph from.",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)


class MergeNodesRequest(BaseRequest):
    """Schema for the request body of the merge nodes endpoint."""

    from_nodes: list[AnnotatedObjectId]
    to_node: AnnotatedObjectId
    save_as_rule: bool = Field(
        default=False,
        description="A boolean specifying whether to save the merge as a rule or not.",
    )


class QueryGraphRequest(BaseRequest):
    """Schema for the request body of the query graph endpoint."""

    query: str | None = Field(
        None,
        description="The query to use for the graph.",
        examples=["Who is the CEO of Apple?"],
        min_length=1,
    )
    values: list[str] | None = Field(
        default=None,
        description="A list of entity values (e.g. their content, names) to use for the graph.",
        examples=["Apple", "Tesla", "Mark Zuckerberg"],
    )
    entities: list[str] | None = Field(
        default=None,
        description="A list of entity types to use for the graph.",
        examples=["Organization", "Person"],
    )
    relations: list[str] | None = Field(
        default=None,
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

    @model_validator(mode="after")
    def check_return_answer_valid(self) -> Self:
        """Check if the return_answer field is valid."""
        if not self.query and self.return_answer:
            raise ValueError("Cannot return answer without a query.")
        return self

    @model_validator(mode="after")
    def check_include_chunks_valid(self) -> Self:
        """Check if the include chunks field is valid."""
        if not self.query and self.include_chunks:
            raise ValueError("Cannot include chunks without a query.")
        return self

    @model_validator(mode="after")
    def check_query_input_valid(self) -> Self:
        """Ensure valid input for querying, either structured or unstructured."""
        if all(
            [
                self.query is None,
                self.values is None,
                self.entities is None,
                self.relations is None,
            ]
        ):
            raise ValueError(
                "You must provide either a query or at least one of the following structured data: values, entities, or relations."
            )
        return self

    @property
    def is_unstructured_query(self) -> bool:
        """Check if the query is unstructured."""
        return self.query is not None


class CypherResponse(BaseResponse):
    """Schema for the cypher text generation response."""

    cypher_text: str


class CreateGraphDetailsResponse(BaseResponse):
    """Schema for the response body of the create graph details endpoint."""

    cost: float = Field(
        ..., description="Estimated cost of the graph creation."
    )
    time: float = Field(
        ..., description="Estimated time taken for the graph creation."
    )
    chunks_selected: int = Field(..., description="Number of chunks selected.")
    chunks_allowed: int = Field(..., description="Number of chunks allowed.")
