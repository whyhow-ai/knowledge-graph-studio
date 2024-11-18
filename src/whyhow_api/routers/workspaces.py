"""Workspace CRUD router."""

import logging
from typing import Annotated, Any, Dict, List

from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException, Query, status
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo.errors import DuplicateKeyError

from whyhow_api.data.demo import DemoDataLoader
from whyhow_api.dependencies import (
    get_db,
    get_db_client,
    get_user,
    valid_workspace_id,
)
from whyhow_api.schemas.base import BaseResponse
from whyhow_api.schemas.workspaces import (
    WorkspaceCreate,
    WorkspaceDocumentModel,
    WorkspaceOut,
    WorkspacesResponse,
    WorkspaceTagsOut,
    WorkspaceTagsResponse,
    WorkspaceUpdate,
)
from whyhow_api.services.crud.base import (
    create_one,
    get_all,
    get_all_count,
    update_one,
)
from whyhow_api.services.crud.workspace import delete_workspace
from whyhow_api.utilities.routers import order_query

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Workspaces"], prefix="/workspaces")


def add_workspace_response(workspace: WorkspaceOut) -> WorkspacesResponse:
    """Add documents response."""
    return WorkspacesResponse(
        message="Workspace created successfully.",
        status="success",
        workspaces=[workspace],
    )


def get_all_workspaces_response(
    workspaces: List[WorkspaceOut],
    total_count: int,
) -> WorkspacesResponse:
    """Get all documents response."""
    return WorkspacesResponse(
        message="Workspaces retrieved successfully.",
        status="success",
        workspaces=workspaces,
        count=total_count,
    )


def get_workspace_response(workspace: WorkspaceOut) -> WorkspacesResponse:
    """Get document response."""
    return WorkspacesResponse(
        message="Workspace retrieved successfully.",
        status="success",
        workspaces=[workspace],
    )


@router.get(
    "", response_model=WorkspacesResponse, response_model_exclude_none=True
)
async def read_workspaces_endpoint(
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=-1, le=50),
    order: int = Depends(order_query),
    name: Annotated[
        str | None, Query(description="The name of the workspace")
    ] = None,
    db: AsyncIOMotorDatabase = Depends(get_db),
    user_id: ObjectId = Depends(get_user),
) -> WorkspacesResponse:
    """Read workspaces."""
    filters = {"name": name} if name else {}
    collection = db["workspace"]
    pipeline = [{"$match": filters}]

    workspaces = await get_all(
        collection=collection,
        document_model=WorkspaceDocumentModel,
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

    return WorkspacesResponse(
        message="Workspaces retrieved successfully.",
        status="success",
        count=total_count,
        workspaces=[WorkspaceOut.model_validate(p) for p in workspaces],
    )


@router.get(
    "/{workspace_id}",
    response_model=WorkspacesResponse,
)
async def read_workspace_endpoint(
    workspace: ObjectId = Depends(valid_workspace_id),
) -> WorkspacesResponse:
    """Get workspace."""
    return WorkspacesResponse(
        message="Workspace retrieved successfully.",
        status="success",
        count=1,
        workspaces=[WorkspaceOut.model_validate(workspace)],
    )


@router.get(
    "/{workspace_id}/tags",
    response_model=WorkspaceTagsResponse,
)
async def collect_workspace_tags_endpoint(
    workspace_id: str,
    db: AsyncIOMotorDatabase = Depends(get_db),
    user_id: ObjectId = Depends(get_user),
) -> WorkspaceTagsResponse:
    """Get workspace tags."""
    collection = db["chunk"]

    tags_field_path = f"tags.{workspace_id}"
    document_tags_field_path = f"document_tags.tags.{workspace_id}"

    pipeline: List[Dict[str, Any]] = [
        {
            "$match": {
                "workspaces": ObjectId(workspace_id),
                "tags": {"$ne": {}},
                tags_field_path: {"$exists": True, "$type": "array"},
            }
        },
        {
            "$lookup": {
                "from": "documents",
                "localField": "document",
                "foreignField": "_id",
                "as": "document_tags",
            }
        },
        {
            "$addFields": {
                "document_tags": {"$arrayElemAt": ["$document_tags", 0]}
            }
        },
        {
            "$addFields": {
                "combined_tags": {
                    "$concatArrays": [
                        {"$ifNull": [f"${tags_field_path}", []]},
                        {"$ifNull": [f"${document_tags_field_path}", []]},
                    ]
                }
            }
        },
        {"$group": {"_id": None, "tags": {"$push": "$combined_tags"}}},
        {
            "$project": {
                "_id": 0,
                "tags": {
                    "$reduce": {
                        "input": "$tags",
                        "initialValue": [],
                        "in": {"$setUnion": ["$$value", "$$this"]},
                    }
                },
            }
        },
    ]

    tags_out = await get_all(
        collection=collection,
        document_model=WorkspaceTagsOut,
        user_id=user_id,
        limit=-1,
        aggregation_query=pipeline,
    )

    validated_tags = [WorkspaceTagsOut.model_validate(n) for n in tags_out]

    if len(validated_tags) == 0:
        return WorkspaceTagsResponse(
            message="No tags found for this workspace.",
            status="success",
            count=0,
            workspace_id=workspace_id,
            tags=[],
        )

    tag_count = len(validated_tags[0].tags)

    return WorkspaceTagsResponse(
        message="Workspace tags retrieved successfully.",
        status="success",
        count=tag_count,
        workspace_id=workspace_id,
        tags=validated_tags[0].tags,
    )


@router.post(
    "", response_model=WorkspacesResponse, response_model_exclude_none=True
)
async def create_workspace_endpoint(
    workspace: WorkspaceCreate,
    db: AsyncIOMotorDatabase = Depends(get_db),
    user_id: ObjectId = Depends(get_user),
) -> WorkspacesResponse:
    """Create workspace."""
    try:
        workspace = await create_one(
            collection=db["workspace"],
            document_model=WorkspaceDocumentModel,
            document=workspace,  # type: ignore[assignment]
            user_id=user_id,
        )
        return WorkspacesResponse(
            message="Workspace created successfully.",
            status="success",
            count=1,
            workspaces=[WorkspaceOut.model_validate(workspace)],
        )
    except DuplicateKeyError as e:
        logger.info(e)
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Workspace with this name already exists.",
        )


@router.put(
    "/{workspace_id}",
    response_model=WorkspacesResponse,
)
async def update_workspace_endpoint(
    workspace_id: str,
    workspace: WorkspaceUpdate,
    db: AsyncIOMotorDatabase = Depends(get_db),
    user_id: ObjectId = Depends(get_user),
) -> WorkspacesResponse:
    """Update workspace."""
    try:
        workspace_obj_id = ObjectId(workspace_id)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workspace not found",
        )
    try:
        updated_workspace = await update_one(
            collection=db["workspace"],
            document_model=WorkspaceDocumentModel,
            id=workspace_obj_id,
            document=workspace,
            user_id=user_id,
        )
        if updated_workspace is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Workspace not found",
            )
        return WorkspacesResponse(
            message="Workspace updated successfully.",
            status="success",
            count=1,
            workspaces=[WorkspaceOut.model_validate(updated_workspace)],
        )
    except DuplicateKeyError as e:
        logger.info(e)
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Workspace with this name already exists.",
        )


@router.delete(
    "/{workspace_id}",
    response_model=WorkspacesResponse,
    description="Delete workspace and associated entities",
)
async def delete_workspace_endpoint(
    db: AsyncIOMotorDatabase = Depends(get_db),
    db_client: AsyncIOMotorClient = Depends(get_db_client),
    workspace: WorkspaceDocumentModel = Depends(valid_workspace_id),
    user_id: ObjectId = Depends(get_user),
) -> WorkspacesResponse:
    """Delete workspace and associated entities."""
    await delete_workspace(
        db=db,
        db_client=db_client,
        user_id=user_id,
        workspace_id=ObjectId(workspace.id),
    )
    return WorkspacesResponse(
        message="Workspace deleted successfully.",
        status="success",
        workspaces=[
            WorkspaceOut(
                id=str(workspace.id),
                created_at=workspace.created_at,
                updated_at=workspace.updated_at,
                created_by=str(workspace.created_by),
                name=workspace.name,
            )
        ],
    )


@router.post("/demo", response_model=BaseResponse)
async def demo_endpoint(
    db: AsyncIOMotorDatabase = Depends(get_db),
    user_id: ObjectId = Depends(get_user),
) -> BaseResponse:
    """Create a prepopulated demo workspace.

    This endpoint initialises a demo workspace with pre-defined sets of chunks,
    schema, graph structures (including nodes and triples), based on the
    authenticated user's ID. The demo data is intended for testing and demonstration purposes.

    Returns
    -------
    - BaseResponse: A response object indicating success or failure of the operation.
    """
    try:
        demo = DemoDataLoader(user_id=user_id)
        await db.workspace.insert_one(demo.data["workspace"])
        await db.chunk.insert_many(demo.data["chunks"])
        await db.schema.insert_one(demo.data["schema"])
        await db.graph.insert_one(demo.data["graph"])
        await db.node.insert_many(demo.data["nodes"])
        await db.triple.insert_many(demo.data["triples"])

        # Wait for all database operations to complete concurrently
        # await gather(
        #     workspace_task,
        #     chunks_task,
        #     schema_task,
        #     graph_task,
        #     nodes_task,
        #     triples_task,
        # )

        logger.info(f"Demo workspace successfully created for user {user_id}.")
        return BaseResponse(
            message="Demo workspace successfully created.",
            status="success",
            count=1,
        )
    except Exception as e:
        logger.error(f"Error creating demo workspace: {e}")
        return BaseResponse(
            message="Error creating demo workspace.",
            status="failed",
            count=0,
        )
