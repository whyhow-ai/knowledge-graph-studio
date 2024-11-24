"""Shared models."""

from typing import Any, Dict, List, Union

from openai import AsyncAzureOpenAI, AsyncOpenAI
from pydantic import BaseModel, Field

from whyhow_api.config import Settings
from whyhow_api.schemas.users import (
    BYOAzureOpenAIMetadata,
    BYOOpenAIMetadata,
    WhyHowOpenAIMetadata,
)

settings = Settings()


class LLMClient:
    """LLM client."""

    def __init__(
        self,
        client: AsyncOpenAI | AsyncAzureOpenAI,
        metadata: Union[
            BYOOpenAIMetadata, BYOAzureOpenAIMetadata, WhyHowOpenAIMetadata
        ],
    ) -> None:
        """Initialize the LLM client."""
        self.client = client
        self.metadata = metadata


class Node(BaseModel):
    """Schema for a single node."""

    name: str = Field(
        ...,
        description="The name of the node.",
        examples=["Python"],
    )
    label: str | None = Field(
        None,
        description=(  # noqa: E501
            "The label assigned to the node (e.g., person, organization,"
            " location)."
        ),
        examples=["Programming Language"],
    )
    properties: dict[str, Any] = Field(
        default_factory=dict, description="Properties of the node."
    )


class Relation(BaseModel):
    """Schema for a single relationship."""

    label: str
    start_node: Node
    end_node: Node
    properties: dict[str, Any] = Field(default_factory=dict)


class Entity(BaseModel):
    """Schema for a single entity.

    Note that this is not identical to Node because
    it has an optional label and the text is a required field.
    """

    text: str = Field(
        ...,
        description=(  # noqa: E501
            "The exact text that was identified as an entity within the larger"
            " text body."
        ),
        examples=["Python"],
    )
    label: str | None = Field(
        None,
        description=(  # noqa: E501
            "The classification label assigned to the identified entity. This"
            " label describes the type of entity (e.g., person, organization,"
            " location)."
        ),
        examples=["Programming Language"],
    )
    properties: dict[str, Any] = Field(
        default_factory=dict, description="Properties of the entity."
    )


class Triple(BaseModel):
    """Schema for a single triple."""

    head: str = Field(
        ...,
        description=(  # noqa: E501
            "The subject/head of the triple, representing an entity or a"
            " concept."
        ),
        examples=["Python"],
        min_length=1,
    )
    head_type: str = Field(
        default="Entity",
        description="The semantic type of the triples subject/head.",
        examples=["Chunk", "Person", "Organisation"],
        min_length=1,
    )
    relation: str = Field(
        ...,
        description=(  # noqa: E501
            "The predicate of the triple, describing the relationship between"
            " the subject and object."
        ),
        examples=["is a"],
        min_length=1,
    )
    tail: str = Field(
        ...,
        description=(  # noqa: E501
            "The object/tail of the triple, representing an entity or concept"
            " that is related to the subject."
        ),
        examples=["Programming Language"],
        min_length=1,
    )
    tail_type: str = Field(
        default="Entity",
        description="The semantic type of the triples object/tail.",
        examples=["Chunk", "Person", "Organisation"],
        min_length=1,
    )
    head_properties: Dict[Any, Any] = Field(
        default={},
        description="Properties of the head entity.",
    )
    relation_properties: Dict[Any, Any] = Field(
        default={},
        description="Properties of the relation.",
    )
    tail_properties: Dict[Any, Any] = Field(
        default={},
        description="Properties of the tail entity.",
    )

    def __str__(self) -> str:
        """Get string representation of the triple."""
        return self.model_dump_json(indent=2)


class EntityField(BaseModel):
    """Schema for a single entity field."""

    name: str
    properties: List[str] = []


class SchemaEntity(BaseModel):
    """Schema Entity model."""

    name: str
    description: str
    fields: List[EntityField] = Field(
        default=[],
        description="Fields corresponding to keys in structured content objects.",
    )


class SchemaRelation(BaseModel):
    """Schema Relation model."""

    name: str
    description: str


class TriplePattern(BaseModel):
    """Schema Triple Pattern model."""

    head: str
    relation: str
    tail: str
    description: str


class SchemaTriplePattern(BaseModel):
    """Schema Triple Pattern model."""

    head: SchemaEntity
    relation: SchemaRelation
    tail: SchemaEntity
    description: str

    # def __str__(self) -> str:
    #     """Return a string representation of the triple pattern."""
    #     # Check if the head and tail are the same
    #     if self.head == self.tail:
    #         return (
    #             f"Head/Tail: {self.head.name} ({self.head.description})\n"
    #             f"Relation: {self.relation.name} ({self.relation.description})\n"  # noqa: E501
    #             f"Pattern: (head/tail:{self.head.name})-[rel:{self.relation.name}]->(head/tail:{self.tail.name}) ({self.description})\n"  # noqa: E501
    #         )
    #     else:
    #         return (
    #             f"Head: {self.head.name} ({self.head.description})\n"
    #             f"Relation: {self.relation.name} ({self.relation.description})\n"  # noqa: E501
    #             f"Tail: {self.tail.name} ({self.tail.description})\n"
    #             f"Pattern: (head:{self.head.name})-[rel:{self.relation.name}]->(tail:{self.tail.name}) ({self.description})\n"  # noqa: E501
    #         )

    # def show_pattern(self) -> str:
    #     """Return a string representation of the triple pattern."""
    #     return f"Pattern: (head:{self.head.name})-[rel:{self.relation.name}]->(tail:{self.tail.name})"  # noqa: E501


class StructuredSchemaEntity(BaseModel):
    """Structured Schema Entity model."""

    name: str
    field: EntityField
    properties: List[str] = Field(default_factory=list)


class StructuredSchemaTriplePattern(BaseModel):
    """Structured Schema Triple Pattern model."""

    head: StructuredSchemaEntity
    relation: str
    tail: StructuredSchemaEntity


class Schema(BaseModel):
    """Schema model."""

    entities: List[SchemaEntity] = Field(default_factory=list)
    relations: List[SchemaRelation] = Field(default_factory=list)
    patterns: List[Union[SchemaTriplePattern, TriplePattern]] = Field(
        default_factory=list
    )

    def get_entity(self, name: str) -> SchemaEntity | None:
        """Return an entity by name if it exists in the schema."""
        for entity in self.entities:
            if entity.name == name:
                return entity
        return None  # Return None if no entity with that name is found

    def get_relation(self, name: str) -> SchemaRelation | None:
        """Return a relation by name if it exists in the schema."""
        for relation in self.relations:
            if relation.name == name:
                return relation
        return None  # Return None if no relation with that name is found


class OpenAICompletionsConfig(BaseModel):
    """OpenAI completions configuration."""

    model: str = Field(default="gpt-4o")  # "gpt-3.5-turbo-0125"
    temperature: float = Field(default=0.1)
    max_tokens: int = Field(default=2000)


class MasterOpenAICompletionsConfig(BaseModel):
    """Master OpenAI completions configuration."""

    default: OpenAICompletionsConfig
    entity: OpenAICompletionsConfig
    triple: OpenAICompletionsConfig
    entity_questions: OpenAICompletionsConfig
    triple_questions: OpenAICompletionsConfig
    entity_concepts: OpenAICompletionsConfig
    triple_concepts: OpenAICompletionsConfig
    merge_graph: OpenAICompletionsConfig


class OpenAIDirectivesConfig(BaseModel):
    """OpenAI directives configuration."""

    entity_questions: str = ""
    triple_questions: str = ""
    entity_concepts: str = ""
    triple_concepts: str = ""
    merge_graph: str = ""
    specific_query: str = ""
    improve_matched_relations: str = ""
    improve_matched_entities: str = ""


class DatasetModel(BaseModel):
    """Dataset model."""

    dataset: Union[Dict[str, List[str]], List[str]]


class PDFProcessorConfig(BaseModel):
    """PDF processor configuration."""

    file_path: str = Field(..., description="File path to PDF document")
    chunk_size: int = Field(default=settings.api.max_chars_per_chunk)
    chunk_overlap: int = Field(0)


class TextWithEntities(BaseModel):
    """Text with extracted entities."""

    text: str = Field(
        ...,
        description=(
            "The original text that was analyzed to identify entities. This"
            " text contains one or more entities that have been classified and"
            " extracted."
        ),
    )
    entities: List[Entity] = Field(
        ...,
        description=(
            "A list of entities extracted from the original text. Each entity"
            " is represented as a combination of the entity's surface form and"
            " its classification label."
        ),
    )
