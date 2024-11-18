"""User models and schemas."""

from datetime import datetime
from typing import Any, List, Literal, Optional, Union

from pydantic import Field, field_validator

from whyhow_api.schemas.base import (
    BaseDocument,
    BaseModel,
    BaseResponse,
    get_utc_now,
)


class WhyHowOpenAIMetadata(BaseModel):
    """Metadata model for WhyHow OpenAI provider."""

    language_model_name: Optional[str] = None
    embedding_name: Optional[str] = None


class BYOAzureOpenAIMetadata(BaseModel):
    """Metadata model for BYO Azure OpenAI provider."""

    api_version: Optional[str] = None
    azure_endpoint: Optional[str] = None
    language_model_name: Optional[str] = None
    embedding_name: Optional[str] = None


class BYOOpenAIMetadata(BaseModel):
    """Metadata model for BYO OpenAI provider."""

    language_model_name: Optional[str] = None
    embedding_name: Optional[str] = None


class Provider(BaseModel):
    """Provider model."""

    type: Literal["llm"]
    value: Literal["byo-openai", "byo-azure-openai"]
    api_key: str
    metadata: dict[
        Literal["byo-openai", "byo-azure-openai"],
        Union[BYOOpenAIMetadata, BYOAzureOpenAIMetadata],
    ]

    @field_validator("metadata", mode="before")
    @classmethod
    def validate_metadata(
        cls,
        v: dict[str, Any],
    ) -> dict[
        Literal["byo-openai", "byo-azure-openai"],
        Union[BYOOpenAIMetadata, BYOAzureOpenAIMetadata],
    ]:
        """Validate metadata based on provider."""
        validated_metadata: dict[
            Literal["byo-openai", "byo-azure-openai"],
            Union[BYOOpenAIMetadata, BYOAzureOpenAIMetadata],
        ] = {}
        for provider, metadata in v.items():
            if provider == "byo-openai":
                try:
                    validated_metadata["byo-openai"] = BYOOpenAIMetadata(
                        **metadata
                    )
                except Exception:
                    raise ValueError(
                        "Metadata must be of type BYOOpenAIMetadata for byo-openai provider"
                    )
            elif provider == "byo-azure-openai":
                try:
                    validated_metadata["byo-azure-openai"] = (
                        BYOAzureOpenAIMetadata(**metadata)
                    )
                except Exception:
                    raise ValueError(
                        "Metadata must be of type BYOAzureOpenAIMetadata for byo-azure-openai provider"
                    )
            else:
                raise ValueError("Invalid provider")
        return validated_metadata


class ProviderConfig(BaseModel):
    """Provider config model."""

    providers: List[Provider]


class UserDocumentModel(BaseDocument):
    """User document model."""

    api_key: Optional[str] = None
    active: bool = True
    email: str = Field(..., min_length=1)
    username: str = Field(..., min_length=1)
    firstname: str = Field(..., min_length=1)
    lastname: str = Field(..., min_length=1)
    created_by: Optional[str] = None  # type: ignore[assignment]


class SetProvidersDetailsResponse(BaseResponse, ProviderConfig):
    """Schema for the response body of the set providers details endpoint."""

    pass


class GetProvidersDetailsResponse(BaseResponse, ProviderConfig):
    """Schema for the response body of the get providers details endpoint."""

    pass


class APIKeyOutModel(BaseModel):
    """Schema for retrieving API key."""

    api_key: str = Field(..., description="User's active api key")
    created_at: datetime = Field(default_factory=get_utc_now)
    updated_at: datetime = Field(default_factory=get_utc_now)


class GetAPIKeyResponse(BaseResponse):
    """Schema for get API key endpoint."""

    whyhow_api_key: List[APIKeyOutModel]


class UserAPIKeyUpdate(BaseModel):
    """Model for updating whyhow api key."""

    api_key: str = Field(description="New user API Key")
    updated_at: datetime = Field(default_factory=get_utc_now)


class DeleteUserResponse(BaseResponse):
    """Schema for the response body of the delete user endpoint."""

    pass


class GetUserStatusResponse(BaseResponse):
    """Schema for get user status."""

    active: bool
