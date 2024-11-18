"""Documents router."""

import logging
import re
from typing import Annotated, Any, Dict, List

import boto3
from botocore.exceptions import ClientError
from bson import ObjectId
from fastapi import APIRouter, BackgroundTasks, Depends, Query, status
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from starlette.exceptions import HTTPException

from whyhow_api.config import Settings
from whyhow_api.dependencies import (
    LLMClient,
    get_db,
    get_db_client,
    get_llm_client,
    get_settings,
    get_user,
    valid_document_id,
    valid_workspace_id,
)
from whyhow_api.schemas.base import Document_Status
from whyhow_api.schemas.documents import (
    DocumentAssignmentResponse,
    DocumentOut,
    DocumentOutWithWorkspaceDetails,
    DocumentsResponse,
    DocumentsResponseWithWorkspaceDetails,
    DocumentUnassignmentResponse,
    DocumentUpdate,
    GeneratePresignedDownloadResponse,
    GeneratePresignedRequest,
    GeneratePresignedResponse,
)
from whyhow_api.schemas.workspaces import WorkspaceDocumentModel
from whyhow_api.services.crud.base import get_all, get_all_count
from whyhow_api.services.crud.document import (
    assign_documents_to_workspace,
    delete_document,
    process_document,
    unassign_documents_from_workspace,
    update_document,
)
from whyhow_api.utilities.routers import order_query

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Documents"], prefix="/documents")


def add_documents_response(
    documents: List[DocumentOut], total_count: int
) -> DocumentsResponse:
    """Add documents response."""
    return DocumentsResponse(
        message="Documents uploaded successfully.",
        status="success",
        documents=documents,
        count=total_count,
    )


def update_document_response(document: DocumentOut) -> DocumentsResponse:
    """Update document response."""
    return DocumentsResponse(
        message="Document updated successfully.",
        status="success",
        documents=[document],
        count=1,
    )


def delete_document_response(document: DocumentOut) -> DocumentsResponse:
    """Delete document response."""
    return DocumentsResponse(
        message="Document deleted successfully.",
        status="success",
        documents=[document],
        count=1,
    )


@router.get(
    "",
    response_model=DocumentsResponseWithWorkspaceDetails,
)
async def read_documents_endpoint(
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=-1, le=50),
    order: int = Depends(order_query),
    filename: Annotated[
        str | None, Query(description="The filename of the document(s)")
    ] = None,
    status: Annotated[
        Document_Status | None,
        Query(description="The status of the document(s)"),
    ] = None,
    workspace_id: Annotated[
        str | None,
        Query(description="The workspace id associated with the document(s)"),
    ] = None,
    workspace_name: Annotated[
        str | None,
        Query(
            description="The workspace name associated with the document(s)"
        ),
    ] = None,
    db: AsyncIOMotorDatabase = Depends(get_db),
    user_id: ObjectId = Depends(get_user),
) -> DocumentsResponseWithWorkspaceDetails:
    """Read documents."""
    if workspace_id and workspace_name:
        logger.info("Both workspace_id and workspace_name cannot be provided.")
        raise HTTPException(
            status_code=400,
            detail="Both workspace_id and workspace_name cannot be provided.",
        )

    collection = db["document"]
    pre_filters: Dict[str, Any] = {}
    post_filters: Dict[str, Any] = {}

    if filename:
        pre_filters["metadata.filename"] = filename
    if status:
        pre_filters["status"] = status
    if workspace_id:
        pre_filters["workspaces"] = ObjectId(workspace_id)
    if workspace_name:
        post_filters["workspaces.name"] = workspace_name

    pipeline = [
        {"$match": pre_filters},
        {
            "$lookup": {
                "from": "workspace",
                "localField": "workspaces",
                "foreignField": "_id",
                "as": "workspaces",
            }
        },
        {"$match": {**post_filters}},
    ]

    # Get total count of items in db
    total_count = await get_all_count(
        collection=collection, user_id=user_id, aggregation_query=pipeline
    )

    if total_count == 0:
        logger.info("No documents found, total_count=0")
        return DocumentsResponseWithWorkspaceDetails(
            message="No documents found.",
            status="success",
            documents=[],
            count=0,
        )
    else:
        documents = await get_all(
            collection=collection,
            document_model=DocumentOutWithWorkspaceDetails,
            user_id=user_id,
            aggregation_query=pipeline,
            skip=skip,
            limit=limit,
            order=order,
        )

        if documents is None:
            logger.info(
                "No documents found. Documents get_all() returned None."
            )
            raise HTTPException(status_code=404, detail="No documents found.")

        return DocumentsResponseWithWorkspaceDetails(
            message="Successfully retrieved documents.",
            status="success",
            documents=[
                DocumentOutWithWorkspaceDetails.model_validate(d)
                for d in documents
            ],
            count=total_count,
        )


@router.get(
    "/{document_id}",
    response_model=DocumentsResponseWithWorkspaceDetails,
)
async def get_document_endpoint(
    document: DocumentOutWithWorkspaceDetails = Depends(valid_document_id),
) -> DocumentsResponseWithWorkspaceDetails:
    """Get document."""
    return DocumentsResponseWithWorkspaceDetails(
        message="Document retrieved successfully.",
        status="success",
        documents=[document],
        count=1,
    )


@router.delete(
    "/{document_id}",
    response_model=DocumentsResponse,
)
async def delete_document_endpoint(
    document: DocumentOutWithWorkspaceDetails = Depends(valid_document_id),
    db_client: AsyncIOMotorClient = Depends(get_db_client),
    settings: Settings = Depends(get_settings),
    user_id: ObjectId = Depends(get_user),
    db: AsyncIOMotorDatabase = Depends(get_db),
) -> DocumentsResponse:
    """Delete document in S3 and associated files."""
    deleted_document = await delete_document(
        db=db,
        db_client=db_client,
        user_id=user_id,
        document_id=ObjectId(document.id),
        settings=settings,
    )
    if deleted_document is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    return delete_document_response(
        DocumentOut.model_validate(deleted_document)
    )


@router.post(
    "/{document_id}/process",
    response_model=DocumentsResponseWithWorkspaceDetails,
)
async def process_document_endpoint(
    background_tasks: BackgroundTasks,
    document: DocumentOutWithWorkspaceDetails = Depends(valid_document_id),
    db: AsyncIOMotorDatabase = Depends(get_db),
    llm_client: LLMClient = Depends(get_llm_client),
    user_id: ObjectId = Depends(get_user),
    settings: Settings = Depends(get_settings),
) -> DocumentsResponseWithWorkspaceDetails:
    """Process a document.

    Triggers a document processing job. This endpoint is usually triggered after files are put into S3.
    """
    logger.info("Starting to process document")

    # Check if document status is:
    # `processing` - this means it is currently being processed.
    # `uploaded` - this means it has not been processed yet.
    # `failed` - this means it has failed processing and can be retried.
    logger.info(f"Document has status: {document.status}")
    if document.status == "processing":
        logger.info("Document is currently being processed.")
        raise HTTPException(
            status_code=400,
            detail="Document is currently being processed.",
        )
    if document.status not in ["uploaded", "failed"]:
        logger.info("Document has already been processed.")
        raise HTTPException(
            status_code=400, detail="Document has already been processed."
        )

    background_tasks.add_task(
        process_document,
        document_id=ObjectId(document.id),
        user_id=user_id,
        db=db,
        llm_client=llm_client,
        bucket=settings.aws.s3.bucket,
    )

    return DocumentsResponseWithWorkspaceDetails(
        message="Document processing started.",
        status="success",
        documents=[document],
        count=1,
    )


@router.put(
    "/assign/{workspace_id}",
    response_model=DocumentAssignmentResponse,
)
async def assign_documents_to_workspace_endpoint(
    document_ids: List[str],
    workspace: WorkspaceDocumentModel = Depends(valid_workspace_id),
    db: AsyncIOMotorDatabase = Depends(get_db),
    user_id: ObjectId = Depends(get_user),
) -> DocumentAssignmentResponse:
    """Assign multiple documents to a workspace."""
    results = await assign_documents_to_workspace(
        document_ids=[ObjectId(i) for i in document_ids],
        workspace_id=ObjectId(workspace.id),
        db=db,
        user_id=user_id,
    )
    return DocumentAssignmentResponse(
        message=(
            "No documents assigned."
            if len(results.assigned) == 0
            else "Documents assigned successfully."
        ),
        status="success",
        documents=results,
        count=len(results.assigned),
    )


@router.put(
    "/unassign/{workspace_id}",
    response_model=DocumentUnassignmentResponse,
)
async def unassign_documents_from_workspace_endpoint(
    document_ids: List[str],
    workspace: WorkspaceDocumentModel = Depends(valid_workspace_id),
    db: AsyncIOMotorDatabase = Depends(get_db),
    db_client: AsyncIOMotorClient = Depends(get_db_client),
    user_id: ObjectId = Depends(get_user),
) -> DocumentUnassignmentResponse:
    """Unassign multiple documents from a workspace."""
    results = await unassign_documents_from_workspace(
        document_ids=[ObjectId(i) for i in document_ids],
        workspace_id=ObjectId(workspace.id),
        db=db,
        db_client=db_client,
        user_id=user_id,
    )
    return DocumentUnassignmentResponse(
        message=(
            "No documents unassigned."
            if len(results.unassigned) == 0
            else "Documents unassigned successfully."
        ),
        status="success",
        documents=results,
        count=len(results.unassigned),
    )


@router.put(
    "/{document_id}/{workspace_id}",
    response_model=DocumentsResponse,
)
async def update_document_in_workspace_endpoint(
    body: DocumentUpdate,
    document: DocumentOutWithWorkspaceDetails = Depends(valid_document_id),
    workspace: WorkspaceDocumentModel = Depends(valid_workspace_id),
    db: AsyncIOMotorDatabase = Depends(get_db),
    user_id: ObjectId = Depends(get_user),
) -> DocumentsResponse:
    """Update document in a workspace."""
    updated_document = await update_document(
        collection=db["document"],
        user_id=user_id,
        document=body,
        document_id=ObjectId(document.id),
        workspace_id=ObjectId(workspace.id),
    )
    return update_document_response(
        DocumentOut.model_validate(updated_document)
    )


@router.post(
    "/generate_presigned",
    response_model=GeneratePresignedResponse,
    description="Generate a presigned POST for uploading a document.",
)
async def generate_presigned_post_endpoint(
    request: GeneratePresignedRequest,
    user_id: ObjectId = Depends(get_user),
    settings: Settings = Depends(get_settings),
    llm_client: LLMClient = Depends(get_llm_client),
) -> GeneratePresignedResponse:
    """Generate a presigned POST for uploading a document."""
    filename = re.sub(r"[^a-zA-Z0-9_.-]", "_", request.filename)
    key = f"{str(user_id)}/{filename}"

    s3_client = boto3.client("s3")

    # check if S3 object already exists
    try:
        s3_client.head_object(Bucket=settings.aws.s3.bucket, Key=key)
    except ClientError:
        full_key = f"{settings.aws.s3.bucket}/{key}"
        logger.info(f"Document does not exist: {full_key}")
    else:
        logger.info(f"Document already exists: {key}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document already exists.",
        )

    # generate a random mongo object id to use as the document id
    object_id_str = str(ObjectId())

    # Set origin workspace id
    workspace_id = str(request.workspace_id)

    response = s3_client.generate_presigned_post(
        Bucket=settings.aws.s3.bucket,
        Key=key,
        Fields={
            "x-amz-meta-document-id": object_id_str,
            "x-amz-meta-origin-workspace-id": workspace_id,
        },
        Conditions=[
            {
                "x-amz-meta-document-id": object_id_str,
            },
            {
                "x-amz-meta-origin-workspace-id": workspace_id,
            },
            [
                "content-length-range",
                0,
                settings.aws.s3.presigned_post_max_bytes,
            ],
        ],
        ExpiresIn=settings.aws.s3.presigned_post_expiration,
    )

    return GeneratePresignedResponse.model_validate(response)


@router.post(
    "/{document_id}/download",
    response_model=GeneratePresignedDownloadResponse,
    description="Generate a presigned url for downloading a document.",
)
async def generate_presigned_download_endpoint(
    document: DocumentOutWithWorkspaceDetails = Depends(valid_document_id),
    user_id: ObjectId = Depends(get_user),
    settings: Settings = Depends(get_settings),
) -> GeneratePresignedDownloadResponse:
    """Generate a presigned url for downloading a document."""
    key = f"{str(user_id)}/{document.metadata.filename}"

    s3_client = boto3.client("s3")

    # make sure the S3 object exists
    try:
        s3_client.head_object(Bucket=settings.aws.s3.bucket, Key=key)
    except ClientError:
        logger.info(f"Document does not exist: {key}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document does not exist.",
        )
    else:
        logger.info(f"Document exists, proceed with url generation: {key}")

    response = s3_client.generate_presigned_url(
        "get_object",
        Params={"Bucket": settings.aws.s3.bucket, "Key": key},
        ExpiresIn=settings.aws.s3.presigned_download_expiration,
    )

    return GeneratePresignedDownloadResponse(url=response)
