"""Document CRUD operations."""

import logging
from typing import Any, Dict, List, Tuple

import boto3
from bson import ObjectId
from motor.motor_asyncio import (
    AsyncIOMotorClient,
    AsyncIOMotorCollection,
    AsyncIOMotorDatabase,
)
from pymongo import UpdateMany, UpdateOne

from whyhow_api.config import Settings
from whyhow_api.models.common import LLMClient
from whyhow_api.schemas.base import ErrorDetails
from whyhow_api.schemas.documents import (
    DocumentAssignments,
    DocumentDocumentModel,
    DocumentOutWithWorkspaceDetails,
    DocumentStateErrorsUpdate,
    DocumentUnassignments,
    DocumentUpdate,
)
from whyhow_api.services.crud.base import update_one
from whyhow_api.services.crud.chunks import (
    perform_node_chunk_unassignment,
    perform_triple_chunk_unassignment,
    process_chunks,
)

logger = logging.getLogger(__name__)


async def get_documents(
    collection: AsyncIOMotorCollection,
    user_id: ObjectId,
    filters: Dict[str, Any] = {},
    skip: int = 0,
    limit: int = 10,
    order: int = 1,
) -> List[DocumentOutWithWorkspaceDetails] | None:
    """Get documents for a user."""
    pipeline: list[dict[str, Any]] = [
        {"$match": {"created_by": user_id, **filters}},
        {
            "$lookup": {
                "from": "workspace",
                "localField": "workspaces",
                "foreignField": "_id",
                "as": "workspaces",
            }
        },
        {"$sort": {"created_at": order, "_id": order}},
        {"$skip": skip},
    ]

    if limit >= 0:
        pipeline.append({"$limit": limit})

    document_dicts = await collection.aggregate(pipeline).to_list(None)

    if document_dicts is None:
        logger.warning("No documents found.")
        return None

    documents = []
    for doc in document_dicts:
        doc["workspaces"] = [
            {"_id": ObjectId(ws["_id"]), "name": ws["name"]}
            for ws in doc["workspaces"]
        ]
        documents.append(DocumentOutWithWorkspaceDetails(**doc))
    return documents


async def get_document(
    collection: AsyncIOMotorCollection, user_id: ObjectId, id: ObjectId
) -> DocumentOutWithWorkspaceDetails | None:
    """Get a document for a user."""
    pipeline: list[dict[str, Any]] = [
        {"$match": {"created_by": user_id, "_id": id}},
        {
            "$lookup": {
                "from": "workspace",
                "localField": "workspaces",
                "foreignField": "_id",
                "as": "workspaces",
            }
        },
    ]

    document_dicts = await collection.aggregate(pipeline).to_list(None)

    if document_dicts is None:
        logger.warning(f"Document {id} not found.")
        return None

    document = document_dicts[0]
    document["workspaces"] = [
        {"_id": ObjectId(ws["_id"]), "name": ws["name"]}
        for ws in document["workspaces"]
    ]

    return DocumentOutWithWorkspaceDetails(**document)


async def update_document(
    collection: AsyncIOMotorCollection,
    user_id: ObjectId,
    document: DocumentUpdate,
    document_id: ObjectId,
    workspace_id: ObjectId,
) -> DocumentDocumentModel:
    """Update a document in a workspace."""
    logger.info(
        f"Updating document: {document_id} within workspace: {workspace_id}"
    )

    update_body = {}
    for k, v in document.model_dump(exclude_none=True).items():
        update_body[f"{k}.{workspace_id}"] = v

    update_doc = {"$set": update_body}

    updated_obj = await collection.find_one_and_update(
        filter={"_id": ObjectId(document_id), "created_by": user_id},
        update=update_doc,
        return_document=True,
    )

    return DocumentDocumentModel(**updated_obj)


async def delete_document_from_s3(
    user_id: ObjectId, filename: str, settings: Settings
) -> None:
    """Delete document from S3."""
    try:
        s3_client = boto3.client("s3")
        respose = s3_client.delete_object(
            Bucket=settings.aws.s3.bucket,
            Key=f"{user_id}/{filename}",
        )
        print(respose)
    except Exception as e:
        logger.error(
            f"Error deleting document {filename} for user {user_id} from S3: {e}"
        )


async def delete_document(
    db: AsyncIOMotorDatabase,
    db_client: AsyncIOMotorClient,
    user_id: ObjectId,
    document_id: ObjectId,
    settings: Settings,
) -> DocumentDocumentModel | None:
    """Delete a document.

    Delete document including its PDF file from S3.
    """
    async with await db_client.start_session() as session:
        async with session.start_transaction():
            try:
                document = await db.document.find_one(
                    {
                        "_id": ObjectId(document_id),
                        "created_by": ObjectId(user_id),
                    },
                    session=session,
                )

                if document is None:
                    return None

                # Find chunks associated with the document
                chunks = await db.chunk.find(
                    {"document": document["_id"], "created_by": user_id},
                    session=session,
                ).to_list(None)
                logger.info(f"Deleting {len(chunks)} chunks")
                chunk_ids_to_delete = [ObjectId(c["_id"]) for c in chunks]

                await db.chunk.delete_many(
                    {
                        "_id": {"$in": chunk_ids_to_delete},
                        "document": document["_id"],
                        "created_by": user_id,
                    },
                    session=session,
                )

                # Unset chunks from nodes
                await perform_node_chunk_unassignment(
                    db=db,
                    session=session,
                    chunk_ids_to_delete=chunk_ids_to_delete,
                    user_id=user_id,
                )

                # Unset chunks from triples
                await perform_triple_chunk_unassignment(
                    db=db,
                    session=session,
                    chunk_ids_to_delete=chunk_ids_to_delete,
                    user_id=user_id,
                )

                # Delete the document itself
                await db.document.delete_one(
                    {
                        "_id": ObjectId(document_id),
                        "created_by": ObjectId(user_id),
                    },
                    session=session,
                )

                # Finally, delete the document from s3
                await delete_document_from_s3(
                    user_id=document["created_by"],
                    filename=document["metadata"]["filename"],
                    settings=settings,
                )

                # Commit the transaction
                await session.commit_transaction()

                logger.info(
                    f"Document {document_id} was successfully deleted."
                )

                return DocumentDocumentModel(**document)
            except:
                logger.error(
                    "An error occurred during the document deletion process",
                    exc_info=True,
                )
                raise


async def get_document_content(
    document_id: ObjectId,
    user_id: ObjectId,
    db: AsyncIOMotorDatabase,
    bucket: str,
) -> Tuple[bytes | None, DocumentDocumentModel | None]:
    """Get document content."""
    document = await db.document.find_one(
        {"_id": document_id, "created_by": user_id}
    )

    if document is None:
        return None, None

    retrieved_document = DocumentDocumentModel(**document)

    s3_client = boto3.client("s3")
    response = s3_client.get_object(
        Bucket=bucket,
        Key=f"{user_id}/{retrieved_document.metadata.filename}",
    )

    content = response["Body"].read()

    return content, retrieved_document


async def process_document(
    document_id: ObjectId,
    user_id: ObjectId,
    db: AsyncIOMotorDatabase,
    llm_client: LLMClient,
    bucket: str,
) -> None:
    """Process document."""
    # Fetch document and its contents
    content, document = await get_document_content(
        document_id=ObjectId(document_id),
        user_id=user_id,
        db=db,
        bucket=bucket,
    )

    if content is None or document is None:
        raise ValueError("Document not found.")

    # Check if document status is:
    # `processing` - this means it is currently being processed.
    # `uploaded` - this means it has not been processed yet.
    # `failed` - this means it has failed processing and can be retried.
    logger.info(f"Document has status: {document.status}")
    if document.status == "processing":
        logger.info("Document is currently being processed.")
        raise ValueError("Document is currently being processed.")
    if document.status not in ["uploaded", "failed"]:
        logger.info("Document has already been processed.")
        raise ValueError("Document has already been processed.")

    # Update document status to `processing`
    await update_one(
        collection=db["document"],
        document_model=DocumentDocumentModel,
        id=ObjectId(document_id),
        document=DocumentStateErrorsUpdate(
            status="processing",
        ),
        user_id=user_id,
    )
    # Process document contents into chunks
    error_message = None
    try:
        if len(document.workspaces) == 0:
            raise ValueError("Document has no workspaces assigned.")

        logger.info("Processing document chunks")
        await process_chunks(
            content=content,
            document_id=ObjectId(document_id),
            db=db,
            llm_client=llm_client,
            workspace_id=ObjectId(document.workspaces[0]),
            user_id=ObjectId(document.created_by),
            extension=document.metadata.format,
        )
        await update_one(
            collection=db["document"],
            document_model=DocumentDocumentModel,
            id=ObjectId(document_id),
            document=DocumentStateErrorsUpdate(
                status="processed",
            ),
            user_id=user_id,
        )
    except ValueError as ve:
        error_message = str(ve)
        logger.error(f"Error processing document: {ve}")
    except Exception as e:
        error_message = str(e)
        logger.error(f"Error processing document: {e}")
    finally:
        # Update the document state to 'failed' with a specific error message
        if error_message:
            await update_one(
                collection=db["document"],
                document_model=DocumentDocumentModel,
                id=ObjectId(document_id),
                document=DocumentStateErrorsUpdate(
                    status="failed",
                    errors=[
                        ErrorDetails(
                            message=error_message,
                            level="critical",
                        )
                    ],
                ),
                user_id=user_id,
            )
            raise Exception(error_message)


async def assign_documents_to_workspace(
    document_ids: List[ObjectId],
    workspace_id: ObjectId,
    db: AsyncIOMotorDatabase,
    user_id: ObjectId,
) -> DocumentAssignments:
    """Assign documents to workspace."""
    results = DocumentAssignments(
        assigned=[], not_found=[], already_assigned=[]
    )

    documents = await db.document.find(
        {"_id": {"$in": document_ids}, "created_by": user_id},
        {"_id": 1, "workspaces": 1},
    ).to_list(None)

    found_document_ids = {doc["_id"] for doc in documents}
    document_bulk_operations = []
    chunk_bulk_operations = []

    for doc in documents:
        if workspace_id not in doc["workspaces"]:
            document_bulk_operations.append(
                UpdateOne(
                    {"_id": doc["_id"], "created_by": user_id},
                    {"$push": {"workspaces": workspace_id}},
                )
            )
            chunk_bulk_operations.append(
                UpdateMany(
                    {"document": doc["_id"], "created_by": user_id},
                    {"$push": {"workspaces": workspace_id}},
                )
            )
            results.assigned.append(str(doc["_id"]))
        else:
            results.already_assigned.append(str(doc["_id"]))

    # Check for documents not found
    for document_id in document_ids:
        if document_id not in found_document_ids:
            results.not_found.append(str(document_id))

    # Perform all document updates in one bulk operation if there are any to perform
    if document_bulk_operations:
        await db.document.bulk_write(document_bulk_operations)

    # Perform all related chunk updates in one bulk operation if there are any to perform
    if chunk_bulk_operations:
        await db.chunk.bulk_write(chunk_bulk_operations)

    return results


async def unassign_documents_from_workspace(
    document_ids: List[ObjectId],
    workspace_id: ObjectId,
    db: AsyncIOMotorDatabase,
    db_client: AsyncIOMotorClient,
    user_id: ObjectId,
) -> DocumentUnassignments:
    """Unassign documents from workspace."""
    results = DocumentUnassignments(
        unassigned=[], not_found=[], not_found_in_workspace=[]
    )

    documents = await db.document.find(
        {"_id": {"$in": document_ids}, "created_by": user_id},
        {"_id": 1, "workspaces": 1},
    ).to_list(None)

    document_ids_found = {doc["_id"] for doc in documents}
    document_ids_to_delete = []
    for doc in documents:
        if workspace_id in doc["workspaces"]:
            document_ids_to_delete.append(doc["_id"])
        else:
            results.not_found_in_workspace.append(str(doc["_id"]))

    for doc_id in document_ids:
        if doc_id not in document_ids_found:
            results.not_found.append(str(doc_id))

    logger.info(f"document_ids_to_delete: {document_ids_to_delete}")

    async with await db_client.start_session() as session:
        async with session.start_transaction():
            try:
                # Get chunks that are marked for deletion
                chunks = await db.chunk.find(
                    {
                        "document": {"$in": document_ids_to_delete},
                        "created_by": user_id,
                    }
                ).to_list(None)
                chunk_ids_to_delete = [ObjectId(c["_id"]) for c in chunks]

                # Unset and delete nodes
                await perform_node_chunk_unassignment(
                    db=db,
                    session=session,
                    chunk_ids_to_delete=chunk_ids_to_delete,
                    user_id=user_id,
                )

                # Unset and delete triples
                await perform_triple_chunk_unassignment(
                    db=db,
                    session=session,
                    chunk_ids_to_delete=chunk_ids_to_delete,
                    user_id=user_id,
                )

                # Unset workspace from chunks
                await db.chunk.update_many(
                    {
                        "document": {"$in": document_ids_to_delete},
                        "created_by": user_id,
                    },
                    {"$pull": {"workspaces": ObjectId(workspace_id)}},
                )

                # Unset workspace from documents
                await db.document.update_many(
                    {
                        "_id": {"$in": document_ids_to_delete},
                        "created_by": user_id,
                    },
                    {"$pull": {"workspaces": ObjectId(workspace_id)}},
                )
                results.unassigned.extend(
                    [str(i) for i in document_ids_to_delete]
                )
                # Commit the transaction
                await session.commit_transaction()
            except:
                logger.error(
                    "An error occurred during the document unassignment process",
                    exc_info=True,
                )
                raise

    return results
