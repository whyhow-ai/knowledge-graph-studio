"""Chunks router."""

import logging
from typing import Annotated, List

from bson import ObjectId
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Body,
    Depends,
    HTTPException,
    Query,
    status,
)
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from whyhow_api.dependencies import (
    LLMClient,
    get_db,
    get_db_client,
    get_llm_client,
    get_user,
    valid_chunk_id,
    valid_workspace_id,
)
from whyhow_api.schemas.base import Chunk_Data_Type
from whyhow_api.schemas.chunks import (
    AddChunksModel,
    AddChunksResponse,
    ChunkAssignmentResponse,
    ChunkDocumentModel,
    ChunksResponse,
    ChunksResponseWithWorkspaceDetails,
    ChunkUnassignmentResponse,
    UpdateChunkModel,
)
from whyhow_api.schemas.workspaces import WorkspaceDocumentModel
from whyhow_api.services.crud.chunks import (
    add_chunks,
    assign_chunks_to_workspace,
    delete_chunk,
    get_chunks,
    get_chunks_with_ws_and_doc_details,
    prepare_chunks,
    unassign_chunks_from_workspace,
    update_chunk,
)
from whyhow_api.utilities.routers import order_query

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Chunks"], prefix="/chunks")


@router.get("", response_model=ChunksResponseWithWorkspaceDetails)
async def read_chunks_endpoint(
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=-1, le=50),
    order: int = Depends(order_query),
    data_type: Annotated[
        Chunk_Data_Type | None,
        Query(description="The data type of the chunk(s)"),
    ] = None,
    workspace_id: Annotated[
        str | None,
        Query(description="The workspace id associated with the chunk(s)"),
    ] = None,
    workspace_name: Annotated[
        str | None,
        Query(description="The workspace name associated with the chunk(s)"),
    ] = None,
    document_id: Annotated[
        str | None,
        Query(description="The document id associated with the chunk(s)"),
    ] = None,
    document_filename: Annotated[
        str | None,
        Query(
            description="The document filename associated with the chunk(s)"
        ),
    ] = None,
    include_embeddings: bool = Query(
        False, description="Whether to include embeddings in the response"
    ),
    db: AsyncIOMotorDatabase = Depends(get_db),
    user_id: ObjectId = Depends(get_user),
) -> ChunksResponseWithWorkspaceDetails:
    """Read chunks."""
    chunks, total_count = await get_chunks_with_ws_and_doc_details(
        db=db,
        user_id=user_id,
        data_type=data_type,
        workspace_id=ObjectId(workspace_id) if workspace_id else None,
        workspace_name=workspace_name,
        document_id=ObjectId(document_id) if document_id else None,
        document_filename=document_filename,
        skip=skip,
        limit=limit,
        order=order,
        include_embeddings=include_embeddings,
    )

    if total_count == 0:
        return ChunksResponseWithWorkspaceDetails(
            message="No chunks found.",
            status="success",
            chunks=[],
            count=0,
        )

    return ChunksResponseWithWorkspaceDetails(
        message="Successfully retrieved chunks.",
        status="success",
        chunks=chunks,
        count=total_count,
    )


@router.get("/{chunk_id}", response_model=ChunksResponseWithWorkspaceDetails)
async def read_chunk_endpoint(
    chunk: ChunkDocumentModel = Depends(valid_chunk_id),
    include_embeddings: bool = Query(
        False, description="Whether to include embeddings in the response"
    ),
    db: AsyncIOMotorDatabase = Depends(get_db),
    user_id: ObjectId = Depends(get_user),
) -> ChunksResponseWithWorkspaceDetails:
    """Read chunk."""
    chunks = await get_chunks(
        collection=db["chunk"],
        user_id=user_id,
        include_embeddings=include_embeddings,
        filters={"_id": chunk.id},
    )

    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Chunk not found."
        )

    return ChunksResponseWithWorkspaceDetails(
        message="Successfully retrieved chunk.",
        status="success",
        chunks=chunks,  # type: ignore[arg-type]
        count=1,
    )


@router.put("/assign/{workspace_id}", response_model=ChunkAssignmentResponse)
async def assign_chunks_to_workspace_endpoint(
    workspace: WorkspaceDocumentModel = Depends(valid_workspace_id),
    chunk_ids: List[str] = Body(...),
    user_id: ObjectId = Depends(get_user),
    db: AsyncIOMotorDatabase = Depends(get_db),
) -> ChunkAssignmentResponse:
    """Assign multiple chunks to a workspace."""
    results = await assign_chunks_to_workspace(
        db=db,
        chunk_ids=[ObjectId(i) for i in chunk_ids],
        workspace_id=ObjectId(workspace.id),
        user_id=user_id,
    )
    return ChunkAssignmentResponse(
        message=(
            "No chunks assigned."
            if len(results.assigned) == 0
            else "Chunks assigned successfully."
        ),
        status="success",
        chunks=results,
        count=len(results.assigned),
    )


@router.put(
    "/unassign/{workspace_id}", response_model=ChunkUnassignmentResponse
)
async def unassign_chunks_from_workspace_endpoint(
    workspace: WorkspaceDocumentModel = Depends(valid_workspace_id),
    chunk_ids: List[str] = Body(...),
    db: AsyncIOMotorDatabase = Depends(get_db),
    user_id: ObjectId = Depends(get_user),
    db_client: AsyncIOMotorClient = Depends(get_db_client),
) -> ChunkUnassignmentResponse:
    """Unassign chunks from a workspace."""
    results = await unassign_chunks_from_workspace(
        chunk_ids=[ObjectId(i) for i in chunk_ids],
        workspace_id=ObjectId(workspace.id),
        db=db,
        db_client=db_client,
        user_id=user_id,
    )
    return ChunkUnassignmentResponse(
        message=(
            "No chunks unassigned."
            if len(results.unassigned) == 0
            else "Chunks unassigned successfully."
        ),
        status="success",
        chunks=results,
        count=len(results.unassigned),
    )


@router.put("/{chunk_id}/{workspace_id}", response_model=ChunksResponse)
async def update_chunk_endpoint(
    chunk: ChunkDocumentModel = Depends(valid_chunk_id),
    workspace: WorkspaceDocumentModel = Depends(valid_workspace_id),
    body: UpdateChunkModel = Body(...),
    db: AsyncIOMotorDatabase = Depends(get_db),
    user_id: ObjectId = Depends(get_user),
) -> ChunksResponseWithWorkspaceDetails:
    """Update a chunk within a workspace.

    This endpoint will replace any content provided in the body.
    """
    message, chunks = await update_chunk(
        chunk_id=ObjectId(chunk.id),
        workspace_id=ObjectId(workspace.id),
        body=body,
        user_id=user_id,
        db=db,
    )

    if chunks is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Chunk not found."
        )

    return ChunksResponseWithWorkspaceDetails(
        message=message,
        status="success",
        chunks=chunks,  # type: ignore[arg-type]
        count=1,
    )


@router.post("/{workspace_id}", response_model=AddChunksResponse)
async def add_chunks_endpoint(
    background_tasks: BackgroundTasks,
    workspace: WorkspaceDocumentModel = Depends(valid_workspace_id),
    body: AddChunksModel = Body(...),
    db: AsyncIOMotorDatabase = Depends(get_db),
    user_id: ObjectId = Depends(get_user),
    llm_client: LLMClient = Depends(get_llm_client),
) -> AddChunksResponse:
    """Add chunks to a workspace."""
    if workspace.id is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Workspace ID is required.",
        )

    prepared_chunks = prepare_chunks(
        chunks=body.chunks,
        workspace_id=ObjectId(workspace.id),
        user_id=user_id,
    )

    added_chunks = await add_chunks(
        db=db,
        llm_client=llm_client,
        chunks=prepared_chunks,
    )

    return AddChunksResponse(
        message="Chunks added successfully.",
        status="success",
        chunks=added_chunks,
        count=len(added_chunks),
    )


@router.delete(
    "/{chunk_id}",
    response_model=ChunksResponse,
)
async def delete_chunk_endpoint(
    chunk: ChunkDocumentModel = Depends(valid_chunk_id),
    db: AsyncIOMotorDatabase = Depends(get_db),
    db_client: AsyncIOMotorClient = Depends(get_db_client),
    user_id: ObjectId = Depends(get_user),
) -> ChunksResponse:
    """Delete a chunk."""
    deleted_chunk = await delete_chunk(
        chunk_id=ObjectId(chunk.id),
        db=db,
        db_client=db_client,
        user_id=user_id,
    )

    if deleted_chunk is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Chunk not found."
        )

    return ChunksResponse(
        message="Chunk deleted.",
        status="success",
        chunks=[deleted_chunk],
        count=1,
    )
