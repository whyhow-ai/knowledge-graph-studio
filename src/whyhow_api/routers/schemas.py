"""Schema CRUD router."""

import logging
from typing import Annotated, Any, Dict, List

from bson import ObjectId
from bson.errors import InvalidId
from fastapi import APIRouter, Depends, HTTPException, Query, status
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo.errors import DuplicateKeyError

from whyhow_api.dependencies import (
    LLMClient,
    get_db,
    get_db_client,
    get_llm_client,
    get_user,
    valid_schema_id,
)
from whyhow_api.schemas.schemas import (
    GeneratedSchemaResponse,
    GenerateSchemaBody,
    SchemaCreate,
    SchemaDocumentModel,
    SchemaOut,
    SchemaOutWithWorkspaceDetails,
    SchemasResponse,
    SchemasResponseWithWorkspaceDetails,
    SchemaUpdate,
)
from whyhow_api.services.crud.base import (
    create_one,
    get_all,
    get_all_count,
    update_one,
)
from whyhow_api.services.crud.schema import delete_schema
from whyhow_api.utilities.builders import OpenAIBuilder
from whyhow_api.utilities.routers import order_query

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Schemas"], prefix="/schemas")


def add_schema_response(schema: SchemaOut) -> SchemasResponse:
    """Add schema response."""
    return SchemasResponse(
        message="Schema created successfully.",
        status="success",
        schemas=[schema],
    )


def get_all_schemas_response(
    schemas: list[SchemaOut], total_count: int
) -> SchemasResponse:
    """Get all schemas response."""
    return SchemasResponse(
        message="Schemas retrieved successfully.",
        status="success",
        schemas=schemas,
        count=total_count,
    )


def get_schema_response(schema: SchemaOut) -> SchemasResponse:
    """Get schema response."""
    return SchemasResponse(
        message="Schema retrieved successfully.",
        status="success",
        schemas=[schema],
    )


def update_schema_response(schema: SchemaOut) -> SchemasResponse:
    """Update schema response."""
    return SchemasResponse(
        message="Schema updated successfully.",
        status="success",
        schemas=[schema],
    )


def delete_schema_response(schema: SchemaOut) -> SchemasResponse:
    """Delete schema response."""
    return SchemasResponse(
        message="Schema deleted successfully.",
        status="success",
        schemas=[schema],
    )


@router.get("", response_model=SchemasResponseWithWorkspaceDetails)
async def read_schemas_endpoint(
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=-1, le=50),
    order: int = Depends(order_query),
    name: Annotated[
        str | None, Query(description="The name of the schema")
    ] = None,
    workspace_id: Annotated[
        str | None, Query(description="The id of the workspace")
    ] = None,
    workspace_name: Annotated[
        str | None, Query(description="The name of the workspace")
    ] = None,
    db: AsyncIOMotorDatabase = Depends(get_db),
    user_id: ObjectId = Depends(get_user),
) -> SchemasResponseWithWorkspaceDetails:
    """Read schemas."""
    if workspace_id and workspace_name:
        raise ValueError(
            "Cannot filter by both workspace_id and workspace_name"
        )

    collection = db["schema"]
    pre_filters: Dict[str, Any] = {}
    post_filters: Dict[str, Any] = {}

    if name:
        pre_filters["name"] = {
            "$regex": name,
            "$options": "i",
        }  # Case-insensitive search
    if workspace_id:
        pre_filters["workspace"] = ObjectId(workspace_id)
    if workspace_name:
        post_filters["workspace.name"] = workspace_name

    pipeline = [
        {"$match": pre_filters},
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
        {"$match": post_filters},
    ]

    schemas = await get_all(
        collection=collection,
        document_model=SchemaOutWithWorkspaceDetails,
        user_id=user_id,
        aggregation_query=pipeline,
        skip=skip,
        limit=limit,
        order=order,
    )

    # Get total count of items in db
    total_count = await get_all_count(
        collection=collection, user_id=user_id, aggregation_query=pipeline
    )

    return SchemasResponseWithWorkspaceDetails(
        message="Schemas retrieved successfully.",
        status="success",
        count=total_count,
        schemas=[
            SchemaOutWithWorkspaceDetails.model_validate(s) for s in schemas
        ],
    )


@router.get("/{schema_id}", response_model=SchemasResponseWithWorkspaceDetails)
async def read_schema_endpoint(
    schema: SchemaDocumentModel = Depends(valid_schema_id),
    db: AsyncIOMotorDatabase = Depends(get_db),
    user_id: ObjectId = Depends(get_user),
) -> SchemasResponseWithWorkspaceDetails:
    """Get schema."""
    collection = db["schema"]

    pipeline: List[Dict[str, Any]] = [
        {"$match": {"_id": ObjectId(schema.id)}},
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

    return SchemasResponseWithWorkspaceDetails(
        message="Schemas retrieved successfully.",
        status="success",
        count=1,
        schemas=[
            SchemaOutWithWorkspaceDetails.model_validate(s) for s in schemas
        ],
    )


@router.post("")
async def create_schema_endpoint(
    schema: SchemaCreate,
    db: AsyncIOMotorDatabase = Depends(get_db),
    user_id: ObjectId = Depends(get_user),
) -> SchemasResponse:
    """Create schema."""
    try:
        schema = await create_one(
            collection=db["schema"],
            document_model=SchemaDocumentModel,  # type: ignore[assignment]
            user_id=user_id,
            document=schema,
        )
        return SchemasResponse(
            message="Schema created successfully.",
            status="success",
            count=1,
            schemas=[SchemaOut.model_validate(schema)],
        )
    except DuplicateKeyError as e:
        logger.error(e)
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Schema with the same name already exists for this workspace.",
        )


@router.put("/{id}", response_model=SchemasResponse)
async def update_schema_endpoint(
    id: str,
    schema: SchemaUpdate,
    db: AsyncIOMotorDatabase = Depends(get_db),
    user_id: ObjectId = Depends(get_user),
) -> SchemasResponse:
    """Update schema."""
    try:
        object_id = ObjectId(id)
    except InvalidId:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid schema id",
        )
    try:
        schema = await update_one(
            collection=db["schema"],
            document_model=SchemaDocumentModel,  # type: ignore[assignment]
            id=object_id,
            document=schema,
            user_id=user_id,
        )
        if schema is None:
            raise HTTPException(status_code=404, detail="Schema not found")
        return SchemasResponse(
            message="Schema updated successfully.",
            status="success",
            count=1,
            schemas=[SchemaOut.model_validate(schema)],
        )
    except DuplicateKeyError as e:
        logger.error(e)
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Schema with the same name already exists.",
        )


@router.delete("/{schema_id}", response_model=SchemasResponse)
async def delete_schema_endpoint(
    schema: SchemaDocumentModel = Depends(valid_schema_id),
    db: AsyncIOMotorDatabase = Depends(get_db),
    db_client: AsyncIOMotorClient = Depends(get_db_client),
    user_id: ObjectId = Depends(get_user),
) -> SchemasResponse:
    """Delete schema."""
    is_deleted = await delete_schema(
        db=db,
        db_client=db_client,
        user_id=user_id,
        schema_id=ObjectId(schema.id),
    )
    if not is_deleted:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Cannot delete schema with associated graphs.",
        )
    return SchemasResponse(
        message="Schema deleted successfully.",
        status="success",
        count=1,
        schemas=[SchemaOut.model_validate(schema)],
    )


@router.post(
    "/generate",
    response_model=GeneratedSchemaResponse,
)
async def generate_schema_endpoint(
    body: GenerateSchemaBody,
    user_id: ObjectId = Depends(get_user),
    llm_client: LLMClient = Depends(get_llm_client),
) -> GeneratedSchemaResponse:
    """Generate schema endpoint."""
    generated_schema, errors = await OpenAIBuilder.generate_schema(
        llm_client=llm_client,
        questions=body.questions,
    )

    return GeneratedSchemaResponse(
        message="Schema generated successfully.",
        status="success",
        questions=body.questions,
        generated_schema=generated_schema,
        errors=errors,
    )
