"""Configuration."""

import logging
from typing import Literal

from pydantic import BaseModel, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

OPENAI_TOKEN_COSTS = {
    "gpt-4o": {"input": 5 / 1000000, "output": 15 / 1000000},
    "gpt-3.5-turbo": {
        "input": 0.5 / 1000000,
        "output": 1.5 / 1000000,
    },
    "text-embedding-3-large": {"input": 0.002 / 1000000},
    "text-embedding-3-small": {"input": 0.13 / 1000000},
}

OPENAI_RATE_LIMITS = {
    1: {
        "gpt-4o": {"rpm": 500, "tpm": 30000},
        "gpt-3.5-turbo": {"rpm": 3500, "tpm": 200000},
        "text-embedding-3-large": {"rpm": 300, "tpm": 1000000},
        "text-embedding-3-small": {"rpm": 300, "tpm": 1000000},
    },
    2: {
        "gpt-4o": {"rpm": 5000, "tpm": 450000},
        "gpt-3.5-turbo": {"rpm": 3500, "tpm": 2000000},
        "text-embedding-3-large": {"rpm": 5000, "tpm": 1000000},
        "text-embedding-3-small": {"rpm": 5000, "tpm": 1000000},
    },
    3: {
        "gpt-4o": {"rpm": 5000, "tpm": 800000},
        "gpt-3.5-turbo": {"rpm": 3500, "tpm": 4000000},
        "text-embedding-3-large": {"rpm": 5000, "tpm": 5000000},
        "text-embedding-3-small": {"rpm": 5000, "tpm": 5000000},
    },
    4: {
        "gpt-4o": {"rpm": 10000, "tpm": 2000000},
        "gpt-3.5-turbo": {"rpm": 10000, "tpm": 10000000},
        "text-embedding-3-large": {"rpm": 10000, "tpm": 5000000},
        "text-embedding-3-small": {"rpm": 10000, "tpm": 5000000},
    },
    5: {
        "gpt-4o": {"rpm": 10000, "tpm": 3000000},
        "gpt-3.5-turbo": {"rpm": 10000, "tpm": 20000000},
        "text-embedding-3-large": {"rpm": 10000, "tpm": 10000000},
        "text-embedding-3-small": {"rpm": 10000, "tpm": 10000000},
    },
}
OPENAI_TIERS = Literal[1, 2, 3, 4, 5]


class SettingsDev(BaseModel):
    """Developer settings."""

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = (
        "INFO"
    )
    openapi_url: str | None = "/openapi.json"

    model_config = SettingsConfigDict(frozen=True)

    @field_validator("openapi_url")
    @classmethod
    def check_openapi_url(cls, value: str | None) -> str | None:
        """Convert empty string to None."""
        if value == "":
            return None
        return value


class SettingsAuth0(BaseModel):
    """Auth0 settings."""

    domain: SecretStr | None = None
    audience: SecretStr | None = None  # api_identifier
    algorithm: str | None = None
    client_domain: SecretStr | None = None
    client_id: SecretStr | None = None
    client_secret: SecretStr | None = None

    @property
    def userinfo_url(self) -> str:
        """Get the userinfo URL."""
        if self.domain is None:
            raise ValueError("domain must be supplied.")
        return f"https://{self.domain.get_secret_value()}/userinfo"

    @property
    def token_url(self) -> str:
        """Get the token URL."""
        if self.domain is None:
            raise ValueError("domain must be supplied.")
        return f"https://{self.domain.get_secret_value()}/oauth/token"

    @property
    def authorize_url(self) -> str:
        """Get the authorize URL."""
        if self.domain is None:
            raise ValueError("domain must be supplied.")
        return f"https://{self.domain.get_secret_value()}/authorize"

    @property
    def jwks_url(self) -> str:
        """Get the JWKS URL."""
        if self.domain is None:
            raise ValueError("domain must be supplied.")
        return (
            f"https://{self.domain.get_secret_value()}/.well-known/jwks.json"
        )


class SettingsAPI(BaseModel):
    """API settings."""

    auth0: SettingsAuth0 = SettingsAuth0()
    limit_frequency_value: int = 30  # tokens added per second
    bucket_capacity: int = 45  # max tokens in bucket
    excluded_paths: list[str] = [
        "/",
        "/openapi.json",
        "/docs",
    ]  # These paths are excluded from rate limiting
    public_paths: list[str] = [
        "/graphs/public",
        "/graphs/public/triples",
        "/graphs/public/nodes",
        "/graphs/public/chunks",
        "/graphs/public/rules",
    ]  # These paths are rate limited but not authenticated

    max_chars_per_chunk: int = 1024
    max_patterns: int = 64
    max_chunk_pattern_product: int = 512
    max_chunk_per_batch: int = 1

    query_sim_triple_limit: int = (
        64  # max number of triples in a similarity search query
    )
    query_sim_triple_candidates: int = (
        64  # max number of candidates to consider (default mongodb)
    )
    restrict_structured_chunk_retrieval: bool = False

    model_config = SettingsConfigDict(frozen=True)


class SettingsGenerativeOpenAI(BaseModel):
    """OpenAI settings."""

    api_key: SecretStr | None = None
    model: str = "gpt-4o"
    temperature: float = 0.0
    max_tokens: int = 3000
    tier: OPENAI_TIERS = 2

    @property
    def rpm_limit(self) -> int:
        """Get the RPM limit."""
        return OPENAI_RATE_LIMITS[self.tier][self.model]["rpm"]

    @property
    def tpm_limit(self) -> int:
        """Get the TPM limit."""
        return OPENAI_RATE_LIMITS[self.tier][self.model]["tpm"]

    @property
    def input_token_cost(self) -> float:
        """Get the input token cost."""
        return OPENAI_TOKEN_COSTS[self.model]["input"]

    @property
    def output_token_cost(self) -> float:
        """Get the output token cost."""
        return OPENAI_TOKEN_COSTS[self.model]["output"]

    model_config = SettingsConfigDict(frozen=True)


class SettingsGenerative(BaseModel):
    """Generative settings."""

    provider: Literal["openai"] = "openai"
    openai: SettingsGenerativeOpenAI = SettingsGenerativeOpenAI()

    model_config = SettingsConfigDict(frozen=True)


class SettingsEmbeddingOpenAI(BaseModel):
    """OpenAI settings."""

    api_key: SecretStr | None = None
    model: str = "text-embedding-3-large"

    model_config = SettingsConfigDict(frozen=True)


class SettingsEmbedding(BaseModel):
    """Embedding settings."""

    provider: Literal["openai"] = "openai"
    openai: SettingsEmbeddingOpenAI = SettingsEmbeddingOpenAI()

    model_config = SettingsConfigDict(frozen=True)


class SettingsS3(BaseModel):
    """S3 settings."""

    bucket: str = ""
    presigned_post_expiration: int = 360  # in seconds
    presigned_download_expiration: int = 360  # in seconds
    presigned_post_max_bytes: int = 50 * int(1e6)

    model_config = SettingsConfigDict(frozen=True)


class SettingsAWS(BaseModel):
    """AWS settings."""

    s3: SettingsS3 = SettingsS3()

    model_config = SettingsConfigDict(frozen=True)


class SettingsMongoDB(BaseModel):
    """MongoDB settings."""

    username: str | None = None
    password: SecretStr | None = None
    database_name: str | None = None
    host: str | None = None
    public_key: SecretStr | None = None
    private_key: SecretStr | None = None
    group_id: str | None = None
    cluster_name: str | None = None
    chunk_collection_name: str = "chunk"
    vector_search_embedding_size: int = 1536

    model_config = SettingsConfigDict(frozen=True)

    @property
    def uri(self) -> str:
        """Generate the MongoDB URI."""
        if self.username is None or self.password is None or self.host is None:
            raise ValueError(
                "'username', 'password', and 'host' must be supplied."
            )

        # At this point, self.password is definitely not None as
        # per the above check.
        password_value = (
            self.password.get_secret_value()
        )  # Type checkers should be satisfied with the above check.
        return f"mongodb+srv://{self.username}:{password_value}@{self.host}"


class SettingsLogfire(BaseModel):
    """Logfire settings."""

    token: SecretStr | None = None


class Settings(BaseSettings):
    """All settings."""

    api: SettingsAPI = SettingsAPI()
    aws: SettingsAWS = SettingsAWS()
    dev: SettingsDev = SettingsDev()
    embedding: SettingsEmbedding = SettingsEmbedding()
    generative: SettingsGenerative = SettingsGenerative()
    mongodb: SettingsMongoDB = SettingsMongoDB()
    logfire: SettingsLogfire = SettingsLogfire()

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="WHYHOW__",
        env_nested_delimiter="__",
        frozen=True,
    )

    @model_validator(mode="after")
    def check(self) -> "Settings":
        """Check everything correct."""
        return self
