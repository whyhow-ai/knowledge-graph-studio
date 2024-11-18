"""Dependencies for FastAPI."""

import logging
from functools import cache
from typing import Any, AsyncGenerator, Dict, List

import jwt
import requests
from auth0.authentication import GetToken
from auth0.management import Auth0
from bson import ObjectId
from bson.errors import InvalidId
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import APIKeyHeader, OAuth2AuthorizationCodeBearer
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from openai import AsyncAzureOpenAI, AsyncOpenAI
from pydantic import ValidationError

from whyhow_api.config import Settings
from whyhow_api.database import get_client
from whyhow_api.models.common import LLMClient
from whyhow_api.schemas.chunks import ChunkDocumentModel
from whyhow_api.schemas.documents import DocumentOutWithWorkspaceDetails
from whyhow_api.schemas.graphs import (
    CreateGraphBody,
    DetailedGraphDocumentModel,
    GraphDocumentModel,
)
from whyhow_api.schemas.nodes import NodeDocumentModel
from whyhow_api.schemas.queries import QueryDocumentModel
from whyhow_api.schemas.schemas import (
    SchemaDocumentModel,
    SchemaOutWithWorkspaceDetails,
)
from whyhow_api.schemas.triples import TripleDocumentModel
from whyhow_api.schemas.users import (
    BYOAzureOpenAIMetadata,
    BYOOpenAIMetadata,
    ProviderConfig,
)
from whyhow_api.schemas.workspaces import WorkspaceDocumentModel
from whyhow_api.services.crud.base import get_all, get_one
from whyhow_api.services.crud.document import get_document
from whyhow_api.services.crud.graph import get_graph
from whyhow_api.utilities.validation import safe_object_id

logger = logging.getLogger(__name__)


@cache
def get_settings() -> Settings:
    """Get settings."""
    return Settings()


async def get_db(
    settings: Settings = Depends(get_settings),
) -> AsyncGenerator[AsyncIOMotorDatabase, None]:
    """Yield a MongoDB database instance."""
    client = get_client()
    if client is None:
        logger.error("Failed to connect to MongoDB client.")
        raise ConnectionError("Failed to retrieve MongoDB client.")

    db = client.get_default_database(settings.mongodb.database_name)
    # logger.info(f"Connected to database: {db.name}")
    try:
        yield db
    finally:
        if db is not None:
            # logger.info(f"Releasing connection to database: {db.name}")
            pass


async def get_db_client() -> AsyncGenerator[AsyncIOMotorClient, None]:
    """Yield a MongoDB client instance."""
    client = get_client()
    if client is None:
        logger.error("Failed to retrieve MongoDB client.")
        raise ConnectionError("Failed to retrieve MongoDB client.")

    logger.info("Providing MongoDB client.")
    try:
        yield client
    finally:
        if client is not None:
            logger.info("Releasing MongoDB client.")


api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)


def get_oauth2_scheme(
    settings: Settings = Depends(get_settings),
) -> OAuth2AuthorizationCodeBearer:
    """Get the OAuth2 scheme."""
    return OAuth2AuthorizationCodeBearer(
        authorizationUrl=settings.api.auth0.authorize_url,
        tokenUrl=settings.api.auth0.token_url,
        auto_error=False,
    )


async def get_token_header(
    request: Request,
    oauth2_scheme: OAuth2AuthorizationCodeBearer = Depends(get_oauth2_scheme),
) -> str | None:
    """Get the OAuth2 token from the header."""
    token = await oauth2_scheme(request)
    return token


async def get_user(
    request: Request,
    api_key: str | None = Depends(api_key_header),
    token: str | None = Depends(get_token_header),
    db: AsyncIOMotorDatabase = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> ObjectId | None:
    """Get the user ID from the API key or OAuth2 token."""
    if api_key is not None:
        logger.info("Authenticating with API key")
        user_document = await db.user.find_one({"api_key": api_key})
        if user_document:
            return safe_object_id(user_document["_id"])
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key",
            )
    elif token is not None:
        logger.info("Authenticating with OAuth2 token")
        if (
            settings.api.auth0.domain is None
            or settings.api.auth0.audience is None
            or settings.api.auth0.algorithm is None
        ):
            raise ValueError("Auth0 domain, audience, and algorithm required")
        domain = settings.api.auth0.domain.get_secret_value()
        audience = settings.api.auth0.audience.get_secret_value()
        algorithm = settings.api.auth0.algorithm

        try:
            signing_key = (
                request.app.state.jwks_client.get_signing_key_from_jwt(
                    token
                ).key
            )

            payload = jwt.decode(
                token,
                signing_key,
                algorithms=[algorithm],
                audience=audience,
                issuer=f"https://{domain}/",
            )

            user_id = payload["sub"]

            if user_id is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token",
                )

            user_document = await db.user.find_one({"sub": user_id})
            if user_document:
                return ObjectId(user_document["_id"])
            else:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="User not found",
                )
        except jwt.exceptions.PyJWKClientError as error:
            logger.error(f"Failed to fetch JWKS: {error}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=str(error),
            )
        except jwt.exceptions.DecodeError as error:
            logger.error(f"Failed to decode token: {error}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=str(error),
            )
        except requests.exceptions.Timeout as te:
            logger.error(f"Request to Auth0 timed out: {te}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to authenticate",
            )
        except requests.RequestException as e:
            logger.error(f"Failed to fetch user info: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token",
            )
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key or token is required",
        )


async def get_llm_client(
    user_id: ObjectId = Depends(get_user),
    db: AsyncIOMotorDatabase = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> LLMClient:
    """Get an OpenAI client."""
    # Get a user's llm config.
    logger.info(f"Getting LLM client for user {user_id}")
    user_document = await db.user.find_one(
        {
            "_id": user_id,
        }
    )

    if user_document is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )

    if "providers" not in user_document:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing providers",
        )

    try:
        provider_config = ProviderConfig.model_validate(
            {"providers": user_document["providers"]}
        )
    except ValidationError as e:
        logger.error(f"Failed to validate provider config: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid provider config",
        )
    llm_providers = [
        provider
        for provider in provider_config.providers
        if provider.type == "llm"
    ]
    if not llm_providers:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="LLM provider not found",
        )
    llm_provider = llm_providers[0]
    print(llm_provider)

    if llm_provider.value == "byo-azure-openai":
        byo_aoai_metadata = BYOAzureOpenAIMetadata.model_validate(
            llm_provider.metadata["byo-azure-openai"]
        )

        if llm_provider.api_key is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API key is missing",
            )

        if byo_aoai_metadata.api_version is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="BYO Azure OpenAI API version is missing",
            )

        if byo_aoai_metadata.azure_endpoint is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="BYO Azure OpenAI Azure endpoint is missing",
            )

        if byo_aoai_metadata.language_model_name is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="BYO Azure OpenAI language model name is missing",
            )

        if byo_aoai_metadata.embedding_name is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="BYO Azure OpenAI embedding name is missing",
            )

        azure_client = AsyncAzureOpenAI(
            api_key=llm_provider.api_key,
            api_version=byo_aoai_metadata.api_version,
            azure_endpoint=byo_aoai_metadata.azure_endpoint,
        )
        return LLMClient(azure_client, byo_aoai_metadata)
    elif llm_provider.value == "byo-openai":
        byo_oai_metadata = BYOOpenAIMetadata.model_validate(
            llm_provider.metadata["byo-openai"]
        )

        if llm_provider.api_key is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API key is missing",
            )

        client = AsyncOpenAI(api_key=llm_provider.api_key)
        return LLMClient(client, byo_oai_metadata)
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid LLM provider",
        )


async def get_auth0(
    settings: Settings = Depends(get_settings),
) -> Auth0:
    """
    Get an Auth0 management client.

    Parameters
    ----------
    settings : Settings
        The settings.

    Returns
    -------
    Auth0
        The Auth0 management client.
    """
    if settings.api.auth0.client_domain is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Auth0 domain is missing",
        )
    if settings.api.auth0.client_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Auth0 client ID is missing",
        )
    if settings.api.auth0.client_secret is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Auth0 client secret is missing",
        )
    domain = settings.api.auth0.client_domain.get_secret_value()
    client_id = settings.api.auth0.client_id.get_secret_value()
    client_secret = settings.api.auth0.client_secret.get_secret_value()
    get_token = GetToken(domain, client_id, client_secret=client_secret)
    token = get_token.client_credentials("https://{}/api/v2/".format(domain))
    mgmt_api_token = token["access_token"]
    return Auth0(domain, mgmt_api_token)


async def valid_workspace_id(
    workspace_id: str,
    user_id: ObjectId = Depends(get_user),
    db: AsyncIOMotorDatabase = Depends(get_db),
) -> WorkspaceDocumentModel:
    """Validate whether the workspace exists for the given user."""
    try:
        workspace = await get_one(
            collection=db["workspace"],
            document_model=WorkspaceDocumentModel,
            id=ObjectId(workspace_id),
            user_id=user_id,
        )
        if workspace is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Workspace not found",
            )
        return WorkspaceDocumentModel.model_validate(workspace)
    except InvalidId:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Invalid workspace id",
        )
    except Exception as e:
        logger.error(f"Failed to validate workspace: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workspace not found",
        )


async def valid_schema_id(
    schema_id: str,
    user_id: ObjectId = Depends(get_user),
    db: AsyncIOMotorDatabase = Depends(get_db),
) -> SchemaOutWithWorkspaceDetails:
    """Validate whether the schema exists for the given user."""
    try:

        collection = db["schema"]

        pipeline: List[Dict[str, Any]] = [
            {"$match": {"_id": ObjectId(schema_id)}},
            {
                "$lookup": {
                    "from": "workspace",
                    "localField": "workspace",
                    "foreignField": "_id",
                    "as": "workspace",
                }
            },
            {
                "$unwind": {
                    "path": "$workspace",
                    "preserveNullAndEmptyArrays": False,
                }
            },
        ]

        schemas = await get_all(
            collection=collection,
            document_model=SchemaOutWithWorkspaceDetails,
            user_id=user_id,
            aggregation_query=pipeline,
        )

        schema = schemas[0]

        if schema is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Schema not found",
            )
        return SchemaOutWithWorkspaceDetails.model_validate(schema)
    except InvalidId:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Invalid schema id",
        )
    except Exception as e:
        logger.error(f"Failed to validate schema: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Schema not found",
        )


async def valid_chunk_id(
    chunk_id: str,
    user_id: ObjectId = Depends(get_user),
    db: AsyncIOMotorDatabase = Depends(get_db),
) -> ChunkDocumentModel:
    """Validate whether the chunk exists for the given user."""
    try:
        chunk = await get_one(
            collection=db["chunk"],
            document_model=ChunkDocumentModel,
            user_id=user_id,
            id=ObjectId(chunk_id),
            remove_fields=["embedding"],
        )
        if chunk is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Chunk not found",
            )
        return ChunkDocumentModel.model_validate(chunk)
    except InvalidId:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Invalid chunk id",
        )
    except Exception as e:
        logger.error(f"Failed to validate chunk: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chunk not found",
        )


async def valid_node_id(
    node_id: str,
    user_id: ObjectId = Depends(get_user),
    db: AsyncIOMotorDatabase = Depends(get_db),
) -> NodeDocumentModel:
    """Validate whether the node exists for the given user."""
    try:
        node = await get_one(
            collection=db["node"],
            document_model=NodeDocumentModel,
            user_id=user_id,
            id=ObjectId(node_id),
        )
        if node is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Node not found"
            )
        return NodeDocumentModel.model_validate(node)
    except InvalidId:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Invalid node id",
        )
    except Exception as e:
        logger.error(f"Failed to validate node: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Node not found"
        )


async def valid_triple_id(
    triple_id: str,
    user_id: ObjectId = Depends(get_user),
    db: AsyncIOMotorDatabase = Depends(get_db),
) -> TripleDocumentModel:
    """Validate whether the triple exists for the given user."""
    try:
        triple = await get_one(
            collection=db["triple"],
            document_model=TripleDocumentModel,
            user_id=user_id,
            id=ObjectId(triple_id),
        )
        if triple is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Triple not found",
            )
        return TripleDocumentModel.model_validate(triple)
    except InvalidId:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Invalid triple id",
        )
    except Exception as e:
        logger.error(f"Failed to validate triple: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Triple not found"
        )


async def valid_graph_id(
    graph_id: str,
    user_id: ObjectId = Depends(get_user),
    db: AsyncIOMotorDatabase = Depends(get_db),
) -> DetailedGraphDocumentModel:
    """Validate whether the graph exists for the given user."""
    try:

        graph = await get_graph(
            collection=db["graph"],
            graph_id=ObjectId(graph_id),
            user_id=user_id,
        )
        if graph is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Graph not found"
            )
        return DetailedGraphDocumentModel.model_validate(graph)
    except InvalidId:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Invalid graph id",
        )
    except Exception as e:
        logger.error(f"Failed to validate graph: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Graph not found"
        )


async def valid_public_graph_id(
    graph_id: str,
    db: AsyncIOMotorDatabase = Depends(get_db),
) -> DetailedGraphDocumentModel:
    """Validate whether the graph is public."""
    try:
        graph = await get_graph(
            collection=db["graph"],
            graph_id=ObjectId(graph_id),
            user_id=None,
            public=True,
        )
        if graph is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Graph not found"
            )
        return DetailedGraphDocumentModel.model_validate(graph)
    except InvalidId:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Invalid graph id",
        )
    except Exception as e:
        logger.error(f"Failed to validate graph: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Graph not found"
        )


async def valid_query_id(
    query_id: str,
    user_id: ObjectId = Depends(get_user),
    db: AsyncIOMotorDatabase = Depends(get_db),
) -> QueryDocumentModel:
    """Validate whether the query exists for the given user."""
    try:
        query = await get_one(
            collection=db["query"],
            document_model=QueryDocumentModel,
            id=ObjectId(query_id),
            user_id=user_id,
        )
        if query is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Query not found",
            )
        return QueryDocumentModel.model_validate(query)
    except InvalidId:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Invalid query id",
        )
    except Exception as e:
        logger.error(f"Failed to validate query: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Query not found",
        )


async def valid_document_id(
    document_id: str,
    user_id: ObjectId = Depends(get_user),
    db: AsyncIOMotorDatabase = Depends(get_db),
) -> DocumentOutWithWorkspaceDetails:
    """Validate whether the document exists for the given user."""
    try:
        document = await get_document(
            collection=db["document"],
            user_id=user_id,
            id=ObjectId(document_id),
        )
        if document is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found",
            )
        return DocumentOutWithWorkspaceDetails.model_validate(document)
    except InvalidId:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Invalid document id",
        )
    except Exception as e:
        logger.error(f"Failed to validate document: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )


async def valid_create_graph(
    body: CreateGraphBody,
    user_id: ObjectId = Depends(get_user),
    db: AsyncIOMotorDatabase = Depends(get_db),
) -> bool | None:
    """Validate whether the resources used for graph creation are valid.

    Validate whether the resources (workspace and schema) used for graph creation are valid,
    and if so, whether the graph already exists.
    """
    # Validate the workspace
    workspace = await get_one(
        collection=db["workspace"],
        document_model=WorkspaceDocumentModel,
        id=ObjectId(body.workspace),
        user_id=user_id,
    )
    if workspace is None:
        logger.info(f"Workspace with id '{body.workspace}' does not exist.")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workspace not found.",
        )

    # Validate the schema
    if body.schema_ is not None:
        schema = await get_one(
            collection=db["schema"],
            document_model=SchemaDocumentModel,
            id=ObjectId(body.schema_),
            user_id=user_id,
        )
        if schema is None:
            logger.info(f"Schema with id '{body.schema_}' does not exist.")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Schema not found.",
            )

    # Check if the graph already exists
    filters = {
        "name": body.name,
        "workspace": body.workspace,
    }
    if body.schema_:
        filters["schema"] = body.schema_

    graph = await get_one(
        collection=db["graph"],
        document_model=GraphDocumentModel,
        user_id=user_id,
        filters=filters,
    )

    if graph:
        logger.info(
            f"Graph with name '{body.name}' already exists in workspace '{body.workspace}'."  # noqa: E501
        )
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Graph already exists or is being created.",
        )
    return True
