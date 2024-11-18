"""Schema models and schemas."""

import logging

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

from whyhow_api.models.common import (
    SchemaEntity,
    SchemaRelation,
    SchemaTriplePattern,
    TriplePattern,
)
from whyhow_api.schemas.base import (
    AfterAnnotatedObjectId,
    AnnotatedObjectId,
    BaseDocument,
    BaseResponse,
    ErrorDetails,
)
from whyhow_api.schemas.workspaces import WorkspaceDetails

logger = logging.getLogger(__name__)


class SchemaDocumentModel(BaseDocument):
    """Schema document model."""

    name: str = Field(..., description="Name of the schema", min_length=1)
    workspace: AfterAnnotatedObjectId = Field(
        ..., description="Workspace id associated with the schema"
    )
    entities: list[SchemaEntity]
    relations: list[SchemaRelation]
    patterns: list[SchemaTriplePattern]


class SchemaCreate(BaseModel):
    """API POST body model."""

    name: str = Field(..., description="Name of the schema", min_length=1)
    workspace: AfterAnnotatedObjectId = Field(
        ..., description="Workspace id associated with the schema"
    )
    entities: list[SchemaEntity]
    relations: list[SchemaRelation]
    patterns: list[TriplePattern]

    @model_validator(mode="after")
    def check_sizes(self) -> Self:
        """Validate the schema size.

        Check that the schema has the required number of entities, relations, and patterns.
        """
        if len(self.entities) < 1:
            raise ValueError("At least one entity must be supplied.")
        if len(self.relations) < 1:
            raise ValueError("At least one relation must be supplied.")
        if len(self.patterns) < 1:
            raise ValueError("At least one pattern must be supplied.")

        return self

    @model_validator(mode="after")
    def update_patterns(self) -> Self:
        """Update the patterns.

        Update the patterns with the entities and relations from the schema.
        """
        patterns = []
        for p in self.patterns:
            head_entity = self.get_entity(p.head)
            tail_entity = self.get_entity(p.tail)
            relation_obj = self.get_relation(p.relation)

            # Only append if all necessary parts are found
            if head_entity and tail_entity and relation_obj:
                patterns.append(
                    SchemaTriplePattern(
                        head=head_entity,
                        relation=relation_obj,
                        tail=tail_entity,
                        description=p.description,
                    ).model_dump()
                )
            else:
                logger.error(
                    f"Warning: Missing entity or relation for pattern {p}"
                )
                raise ValueError(
                    f"Warning: Missing entity or relation for pattern {p}"
                )

        self.patterns = patterns  # type: ignore[assignment]
        return self

    def get_entity(self, name: str) -> SchemaEntity | None:
        """Return an entity by name if it exists in the schema."""
        for entity in self.entities:
            if entity.name == name:
                return entity
        return None

    def get_relation(self, name: str) -> SchemaRelation | None:
        """Return a relation by name if it exists in the schema."""
        for relation in self.relations:
            if relation.name == name:
                return relation
        return None

    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
    )


class SchemaUpdate(BaseModel):
    """API PUT body model."""

    name: str | None = Field(
        default=None, description="Name of the schema", min_length=1
    )


class SchemaOut(SchemaDocumentModel):
    """API response model."""

    id: AnnotatedObjectId = Field(..., alias="_id")
    workspace: AnnotatedObjectId = Field(..., alias="workspace_id")
    created_by: AnnotatedObjectId

    model_config = ConfigDict(
        use_enum_values=True, from_attributes=True, populate_by_name=True
    )


class SchemaOutWithWorkspaceDetails(SchemaOut):
    """Schema response model with workspace details."""

    workspace: WorkspaceDetails  # type: ignore[assignment]


class SchemasResponse(BaseResponse):
    """Schema for the response body of the schemas endpoints."""

    schemas: list[SchemaOut]


class SchemasResponseWithWorkspaceDetails(BaseResponse):
    """Schema for the response body of the schemas endpoints with workspace details."""

    schemas: list[SchemaOutWithWorkspaceDetails] = Field(
        default=[], description="list of schemas"
    )


class GenerateSchemaBody(BaseModel):
    """Generate schema body."""

    workspace: AfterAnnotatedObjectId = Field(
        ..., description="Workspace to generate schema for."
    )
    questions: list[str] = Field(
        ..., description="list of questions to generate schema for."
    )

    @model_validator(mode="after")
    def validate_question_sizes(self) -> Self:
        """Validate question sizes."""
        self.questions = [
            q
            for q in filter(lambda x: x is not None, self.questions)
            if q.strip()
        ]
        if len(self.questions) == 0:
            raise ValueError("At least one question must be provided.")
        return self

    model_config = ConfigDict(arbitrary_types_allowed=True)


class GeneratedSchema(BaseModel):
    """Schema document model."""

    entities: list[SchemaEntity]
    relations: list[SchemaRelation]
    patterns: list[TriplePattern]

    @model_validator(mode="after")
    def validate_patterns(self) -> Self:
        """Validate the patterns.

        Check that the entities and relations in the patterns are in the entities and relations lists.
        """
        for pattern in self.patterns:
            if pattern.head not in [entity.name for entity in self.entities]:
                raise ValueError(
                    f"Pattern head '{pattern.head}' not found in entities."
                )
            if pattern.tail not in [entity.name for entity in self.entities]:
                raise ValueError(
                    f"Pattern tail '{pattern.tail}' not found in entities."
                )
            if pattern.relation not in [
                relation.name for relation in self.relations
            ]:
                raise ValueError(
                    f"Pattern relation '{pattern.relation}' not found in relations."
                )
        return self


class GeneratedSchemaResponse(BaseResponse):
    """Response of the generated schema."""

    questions: list[str]
    generated_schema: GeneratedSchema
    errors: list[ErrorDetails] = Field(default=[])


class SchemaDetails(BaseModel):
    """Schema for schema details."""

    id: AnnotatedObjectId = Field(..., alias="_id")
    name: str = Field(..., description="Name of the schema", min_length=1)
