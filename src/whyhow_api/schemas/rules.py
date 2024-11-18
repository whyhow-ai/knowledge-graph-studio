"""Rule models and schemas."""

from pydantic import BaseModel, ConfigDict, Field

from whyhow_api.schemas.base import (
    AfterAnnotatedObjectId,
    AnnotatedObjectId,
    BaseDocument,
    BaseResponse,
    Rule_Type,
)


class RuleBody(BaseModel):
    """Base schema for rule body."""

    rule_type: Rule_Type = Field(..., description="Type of rule")


class MergeNodesRule(RuleBody):
    """Merge nodes rule schema."""

    from_node_names: list[str]
    to_node_name: str
    node_type: str


class RuleBase(BaseModel):
    """Base schema for rule."""

    workspace: AfterAnnotatedObjectId = Field(
        ..., description="Workspace id associated with the schema"
    )
    rule: MergeNodesRule = Field(..., description="Rule body")

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )


class RuleDocumentModel(BaseDocument, RuleBase):
    """Rule document model."""

    pass


class RuleCreate(RuleBase):
    """API POST body model for creating a rule."""

    pass


class RuleOut(RuleDocumentModel):
    """API Response model for a rule."""

    id: AnnotatedObjectId = Field(..., alias="_id")
    workspace: AnnotatedObjectId = Field(..., alias="workspace_id")
    created_by: AnnotatedObjectId


class RulesResponse(BaseResponse):
    """Schema for the response body of the rules endpoints."""

    rules: list[RuleOut]
