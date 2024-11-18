"""Chunk CRUD operations."""

import json
import logging
import sys
from io import BytesIO, StringIO
from typing import Any, Callable, Dict, List, Tuple, get_args

import pandas as pd
from bson import ObjectId
from langchain.text_splitter import RecursiveCharacterTextSplitter
from motor.motor_asyncio import (
    AsyncIOMotorClient,
    AsyncIOMotorClientSession,
    AsyncIOMotorCollection,
    AsyncIOMotorDatabase,
)
from pandas.errors import EmptyDataError, ParserError
from pymongo import InsertOne, UpdateOne
from pymongo.errors import BulkWriteError
from pypdf import PdfReader

from whyhow_api.config import Settings
from whyhow_api.models.common import LLMClient
from whyhow_api.schemas.base import (
    AllowedChunkContentTypes,
    ErrorDetails,
    File_Extensions,
)
from whyhow_api.schemas.chunks import (
    AddChunkModel,
    ChunkAssignments,
    ChunkDocumentModel,
    ChunkMetadata,
    ChunkOut,
    ChunksOutWithWorkspaceDetails,
    ChunkUnassignments,
    UpdateChunkModel,
)
from whyhow_api.schemas.documents import (
    DocumentDocumentModel,
    DocumentStateErrorsUpdate,
)
from whyhow_api.services.crud.base import update_one
from whyhow_api.utilities.common import embed_texts

logger = logging.getLogger(__name__)

settings = Settings()


async def get_chunks(
    collection: AsyncIOMotorCollection,
    user_id: ObjectId,
    llm_client: LLMClient | None = None,
    include_embeddings: bool = False,
    filters: Dict[str, Any] = {},
    skip: int = 0,
    limit: int = 10,
    order: int = 1,
    populate: bool = True,
) -> List[ChunksOutWithWorkspaceDetails] | List[ChunkDocumentModel]:
    """Get chunks for a user with optional population of related data."""
    seed_concept = filters.pop("seed_concept", None)

    pipeline = []

    if seed_concept:
        if llm_client is None:
            raise ValueError(
                "OpenAI client must be supplied to use vector search"
            )
        logger.info(
            f"Using vector similarity search with seed concept: {seed_concept}"
        )
        query_vector_list = await embed_texts(
            llm_client=llm_client, texts=[seed_concept]
        )
        query_vector = query_vector_list[0]
        # logger.info(f"Query vector length: {len(query_vector)}")
        pipeline.append(
            {
                "$vectorSearch": {
                    "index": "vector_search_index",
                    "filter": {
                        "created_by": {"$eq": user_id},
                        "workspaces": filters["workspaces"],
                        **(
                            {"data_type": filters["data_type"]}
                            if "data_type" in filters
                            else {}
                        ),
                    },
                    "path": "embedding",
                    "queryVector": query_vector,
                    "numCandidates": 512,
                    "limit": limit,
                }
            }
        )

    # logger.info(f"Applying filters: {filters}")
    pipeline.append({"$match": {"created_by": user_id, **filters}})

    if not include_embeddings:
        # logger.debug("Excluding embeddings from the results")
        pipeline.append({"$project": {"embedding": 0}})

    if populate:
        # logger.debug("Populating related data")
        pipeline.extend(
            [
                {
                    "$lookup": {
                        "from": "workspace",
                        "localField": "workspaces",
                        "foreignField": "_id",
                        "as": "workspaces",
                    }
                },
                {
                    "$lookup": {
                        "from": "document",
                        "localField": "document",
                        "foreignField": "_id",
                        "as": "document",
                    }
                },
                {
                    "$set": {
                        "workspaces": {
                            "$map": {
                                "input": "$workspaces",
                                "as": "workspace",
                                "in": {
                                    "_id": "$$workspace._id",
                                    "name": "$$workspace.name",
                                },
                            }
                        },
                        "document": {
                            "$map": {
                                "input": "$document",
                                "as": "doc",
                                "in": {
                                    "_id": "$$doc._id",
                                    "filename": {
                                        "$getField": {
                                            "field": "filename",
                                            "input": "$$doc.metadata",
                                        }
                                    },
                                },
                            }
                        },
                    }
                },
                {
                    "$unwind": {
                        "path": "$document",
                        "preserveNullAndEmptyArrays": True,
                    }
                },
            ]
        )

    # logger.debug(f"Sorting chunks by created_at in order: {order}")
    pipeline.extend(
        [
            {"$sort": {"created_at": order, "_id": order}},
            {"$skip": skip},  # type: ignore[dict-item]
        ]
    )

    if limit >= 0:
        # logger.debug(f"Limiting results to {limit} chunks")
        pipeline.append({"$limit": limit})  # type: ignore[dict-item]

    # logger.info(f"Running query: {pipeline}")
    chunks = await collection.aggregate(pipeline).to_list(None)

    if chunks is None:
        logger.warning("No chunks found.")
        return []

    logger.info(
        f"GET_CHUNKS :: retrieved {len(chunks)} chunks with limit {limit}"
    )

    if populate:
        # logger.debug("Returning chunks with populated workspace details")
        return [ChunksOutWithWorkspaceDetails(**c) for c in chunks]
    else:
        # logger.debug("Returning chunks without populated workspace details")
        return [ChunkDocumentModel(**c) for c in chunks]


def split_text_into_chunks(
    text: str, page_number: int | None = None
) -> List[Dict[str, Any]]:
    """Split the given text into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.api.max_chars_per_chunk, chunk_overlap=0
    )
    chunks = []
    loc = 0
    if page_number is not None:
        for d in text_splitter.split_text(text):
            _length = len(d)
            chunks.append(
                {
                    "content": d,
                    "metadata": {
                        "start": loc,
                        "end": loc + _length,
                        "page": page_number,
                    },
                }
            )
            loc += _length
    else:
        for d in text_splitter.split_text(text):
            _length = len(d)
            chunks.append(
                {
                    "content": d,
                    "metadata": {"start": loc, "end": loc + _length},
                }
            )
            loc += _length
    return chunks


def prepare_chunks(
    chunks: List[AddChunkModel], workspace_id: ObjectId, user_id: ObjectId
) -> List[ChunkDocumentModel]:
    """Prepare chunks for insertion.

    Used for preparing POST body chunks for insertion into the database.
    """
    return [
        ChunkDocumentModel(
            workspaces=[workspace_id],
            content=c.content,
            data_type="string" if isinstance(c.content, str) else "object",
            tags={str(workspace_id): c.tags} if c.tags else {},
            metadata=ChunkMetadata(
                length=len(
                    c.content
                    if isinstance(c.content, str)
                    else c.content.keys()
                ),
                size=sys.getsizeof(c.content),
                data_source_type="manual",
            ),
            user_metadata=(
                {str(workspace_id): c.user_metadata} if c.user_metadata else {}
            ),
            created_by=user_id,
        )
        for c in chunks
    ]


async def add_chunks(
    db: AsyncIOMotorDatabase,
    llm_client: LLMClient,
    chunks: List[ChunkDocumentModel],
) -> list[ChunkOut]:
    """Add chunks to the database.

    All data types are embedded (`objects` are stringified). Populates the `chunk` collection.
    """
    try:
        # Embeds text if string, otherwise stringifies the object.
        embeddings = await embed_texts(
            llm_client=llm_client,
            texts=[
                (
                    c.content
                    if c.data_type == "string"
                    else json.dumps(obj=c.content)  # type: ignore[misc]
                )
                for c in chunks
            ],
        )

        # Add embeddings to chunks and create operation objects
        operations = []
        chunks_ids = []
        for idx, c in enumerate(chunks):
            c.embedding = embeddings[idx]
            c.id = ObjectId()
            chunks_ids.append(c.id)
            operations.append(
                InsertOne(
                    c.model_dump(by_alias=True, exclude_none=True),
                )
            )

        if operations:
            await db.chunk.bulk_write(operations)
            inserted_chunks = await db.chunk.find(
                {"_id": {"$in": chunks_ids}},
                {"embedding": 0},
            ).to_list(None)
        else:
            inserted_chunks = []

        return [ChunkOut(**c) for c in inserted_chunks]

    except BulkWriteError as e:
        logger.error(f"Database operation failed: {e.details}")
        raise Exception(f"Database operation failed: {e.details}")


def validate_and_convert(value: Any) -> Any:
    """Validate and convert a single value based on a fix set of allowable data types."""
    if not isinstance(value, get_args(AllowedChunkContentTypes)):
        return str(value)
    return value


def create_structured_chunks(
    content: bytes,
    document_id: ObjectId,
    workspace_id: ObjectId,
    user_id: ObjectId,
    file_type: str,
) -> List[ChunkDocumentModel]:
    """General function to create chunks from structured files (CSV or JSON)."""
    content_decoded = content.decode("utf-8")
    chunks: list[ChunkDocumentModel] = []
    if file_type == "csv":
        df = pd.read_csv(StringIO(content_decoded))
    elif file_type == "json":
        df = pd.read_json(StringIO(content_decoded))
    else:
        raise ValueError("Unsupported file type")

    df = df.map(validate_and_convert)

    # Convert NaN to None
    df = df.replace({float("nan"): None})

    # Convert the dataframe to a list of dictionaries for further processing
    # Each dictionary in the list represents a row in the dataframe
    data = df.to_dict(orient="records")

    for idx, obj in enumerate(data):
        chunks.append(
            ChunkDocumentModel(
                document=document_id,
                workspaces=[workspace_id],
                data_type="object",
                content=obj,
                tags={},
                metadata=ChunkMetadata(
                    length=len(obj.keys()),
                    size=sys.getsizeof(obj),
                    data_source_type="automatic",
                    index=idx,
                ),
                user_metadata={},
                created_by=user_id,
            )
        )
    logger.info(f"Created {len(chunks)} chunks from structured data")
    return chunks


async def process_structured_chunks(
    content: bytes,
    document_id: ObjectId,
    db: AsyncIOMotorDatabase,
    llm_client: LLMClient,
    workspace_id: ObjectId,
    user_id: ObjectId,
    file_type: str,
) -> None:
    """Process structured chunks content from CSV or JSON files."""
    error_message = None
    try:
        chunks = create_structured_chunks(
            content,
            document_id,
            workspace_id,
            user_id,
            file_type,
        )
        await add_chunks(db, llm_client, chunks)
    except ValueError as ve:
        error_message = "Unsupported file type selected. Please choose either 'csv' or 'json'."
        logger.error(f"ValueError: {ve}")
    except ParserError as pe:
        error_message = "There was an error parsing the file. Please check the file format and contents."
        logger.error(f"ParserError: {pe}")
    except UnicodeDecodeError as ude:
        error_message = "Failed to decode the file. Please ensure the file encoding is correct (e.g., UTF-8)."
        logger.error(f"UnicodeDecodeError: {ude}")
    except EmptyDataError as ede:
        error_message = (
            "The provided file is empty. Please check the file content."
        )
        logger.error(f"EmptyDataError: {ede}")
    except Exception as e:
        error_message = "An unexpected error occurred during file processing."
        logger.error(f"Unexpected error: {e}")
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


def create_unstructured_chunks(
    content: bytes,
    document_id: ObjectId,
    workspace_id: ObjectId,
    user_id: ObjectId,
    file_type: str,
) -> List[ChunkDocumentModel]:
    """Create chunks from text-based files (PDF or TXT)."""
    text_chunks = []
    if file_type == "pdf":
        reader = PdfReader(BytesIO(content))
        pages = [page.extract_text() for page in reader.pages]
        for page_text, page_number in zip(pages, range(len(pages))):
            text_chunks.extend(split_text_into_chunks(page_text, page_number))
    elif file_type == "txt":
        text = content.decode("utf-8")
        text_chunks.extend(split_text_into_chunks(text))
    else:
        raise ValueError("Unsupported file type")

    chunks = []
    for chunk in text_chunks:
        chunks.append(
            ChunkDocumentModel(
                document=document_id,
                workspaces=[workspace_id],
                data_type="string",
                content=chunk["content"],
                tags={},
                metadata=ChunkMetadata(
                    length=len(chunk["content"]),
                    size=sys.getsizeof(chunk["content"]),
                    data_source_type="automatic",
                    **chunk["metadata"],
                ),
                user_metadata={},
                created_by=user_id,
            )
        )
    logger.info(f"Created {len(chunks)} chunks from unstructured data")
    return chunks


async def process_unstructured_chunks(
    content: bytes,
    document_id: ObjectId,
    db: AsyncIOMotorDatabase,
    llm_client: LLMClient,
    workspace_id: ObjectId,
    user_id: ObjectId,
    file_type: str,
) -> None:
    """Process unstructured content from PDF or TXT files."""
    error_message = None
    try:
        chunks = create_unstructured_chunks(
            content,
            document_id,
            workspace_id,
            user_id,
            file_type,
        )
        await add_chunks(db, llm_client, chunks)
    except ValueError as ve:
        error_message = "Unsupported file type selected. Please choose either 'pdf' or 'txt'."
        logger.error(f"ValueError: {ve}")
    except ParserError as pe:
        error_message = "There was an error parsing the file. Please check the file format and contents."
        logger.error(f"ParserError: {pe}")
    except UnicodeDecodeError as ude:
        error_message = "Failed to decode the file. Please ensure the file encoding is correct (e.g., UTF-8)."
        logger.error(f"UnicodeDecodeError: {ude}")
    except EmptyDataError as ede:
        error_message = (
            "The provided file is empty. Please check the file content."
        )
        logger.error(f"EmptyDataError: {ede}")
    except Exception as e:
        error_message = "An unexpected error occurred during file processing."
        logger.error(f"Unexpected error: {e}")
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


SUPPORTED_EXTENSIONS = {
    "csv": process_structured_chunks,
    "json": process_structured_chunks,
    "pdf": process_unstructured_chunks,
    "txt": process_unstructured_chunks,
}


async def process_chunks(
    content: bytes,
    document_id: ObjectId,
    db: AsyncIOMotorDatabase,
    llm_client: LLMClient,
    workspace_id: ObjectId,
    user_id: ObjectId,
    extension: File_Extensions,
) -> None:
    """Process chunks based on the file extension."""
    process_func: Callable = SUPPORTED_EXTENSIONS[extension]  # type: ignore[type-arg]

    await process_func(
        content=content,
        document_id=document_id,
        db=db,
        workspace_id=workspace_id,
        user_id=user_id,
        llm_client=llm_client,
        file_type=extension,
    )


async def assign_chunks_to_workspace(
    db: AsyncIOMotorDatabase,
    chunk_ids: List[ObjectId],
    workspace_id: ObjectId,
    user_id: ObjectId,
) -> ChunkAssignments:
    """Assign a chunks to a workspace."""
    results = ChunkAssignments(assigned=[], not_found=[], already_assigned=[])

    chunks = await db.chunk.find(
        {"_id": {"$in": chunk_ids}, "created_by": user_id},
        {"_id": 1, "workspaces": 1},
    ).to_list(None)

    found_chunk_ids = {chunk["_id"] for chunk in chunks}
    bulk_operations = []

    for chunk in chunks:
        if workspace_id not in chunk["workspaces"]:
            bulk_operations.append(
                UpdateOne(
                    {"_id": chunk["_id"], "created_by": user_id},
                    {"$push": {"workspaces": workspace_id}},
                )
            )
            results.assigned.append(str(chunk["_id"]))
        else:
            results.already_assigned.append(str(chunk["_id"]))

    # Check for chunks not found
    for chunk_id in chunk_ids:
        if chunk_id not in found_chunk_ids:
            results.not_found.append(str(chunk_id))

    if bulk_operations:
        await db.chunk.bulk_write(bulk_operations)

    return results


async def perform_node_chunk_unassignment(
    db: AsyncIOMotorDatabase,
    session: AsyncIOMotorClientSession,
    chunk_ids_to_delete: list[ObjectId],
    user_id: ObjectId,
) -> None:
    """Perform unassignment of chunks from nodes."""
    await db.node.update_many(
        {
            "chunks": {"$in": chunk_ids_to_delete},
            "created_by": user_id,
        },
        {"$pull": {"chunks": {"$in": chunk_ids_to_delete}}},
        session=session,
    )


async def perform_triple_chunk_unassignment(
    db: AsyncIOMotorDatabase,
    session: AsyncIOMotorClientSession,
    chunk_ids_to_delete: List[ObjectId],
    user_id: ObjectId,
) -> None:
    """Perform unassignment of chunks from triples."""
    await db.triple.update_many(
        {
            "chunks": {"$in": chunk_ids_to_delete},
            "created_by": user_id,
        },
        {"$pull": {"chunks": {"$in": chunk_ids_to_delete}}},
        session=session,
    )


async def unassign_chunks_from_workspace(
    chunk_ids: List[ObjectId],
    workspace_id: ObjectId,
    db: AsyncIOMotorDatabase,
    db_client: AsyncIOMotorClient,
    user_id: ObjectId,
) -> ChunkUnassignments:
    """
    Unassigns chunks from a specified workspace and cleans up any nodes or triples that no longer contain chunks.

    Parameters
    ----------
    - chunk_ids (List[ObjectId]): List of chunk IDs to unassign.
    - workspace_id (ObjectId): ID of the workspace from which chunks are being unassigned.
    - db (AsyncIOMotorDatabase): Database instance for accessing chunks.
    - db_client (AsyncIOMotorDatabase): Database client used for managing transactions.
    - user_id (ObjectId): ID of the user performing the unassignment.

    Returns
    -------
    - ChunkUnassignments: Object containing the lists of unassigned, not found, and not found in workspace chunk IDs.
    """
    results = ChunkUnassignments(
        unassigned=[], not_found=[], not_found_in_workspace=[]
    )

    chunks = await db.chunk.find(
        {"_id": {"$in": chunk_ids}, "created_by": user_id},
        {"_id": 1, "workspaces": 1},
    ).to_list(None)

    chunk_ids_found = {chunk["_id"] for chunk in chunks}
    chunk_ids_to_delete = []
    for chunk in chunks:
        if workspace_id in chunk["workspaces"]:
            chunk_ids_to_delete.append(chunk["_id"])
        else:
            results.not_found_in_workspace.append(str(chunk["_id"]))

    for chunk_id in chunk_ids:
        if chunk_id not in chunk_ids_found:
            results.not_found.append(str(chunk_id))

    logger.info(f"chunk_ids_to_delete: {chunk_ids_to_delete}")

    async with await db_client.start_session() as session:
        async with session.start_transaction():
            try:
                await perform_node_chunk_unassignment(
                    db=db,
                    session=session,
                    chunk_ids_to_delete=chunk_ids_to_delete,
                    user_id=user_id,
                )

                await perform_triple_chunk_unassignment(
                    db=db,
                    session=session,
                    chunk_ids_to_delete=chunk_ids_to_delete,
                    user_id=user_id,
                )

                # Unset workspace from chunks
                await db.chunk.update_many(
                    {
                        "_id": {"$in": chunk_ids_to_delete},
                        "created_by": user_id,
                    },
                    {"$pull": {"workspaces": ObjectId(workspace_id)}},
                )
                results.unassigned.extend(
                    [str(i) for i in chunk_ids_to_delete]
                )
                logger.info(
                    f"Chunks successfully unassigned: {results.unassigned}"
                )
                # Commit the transaction
                await session.commit_transaction()
            except:
                logger.error(
                    "An error occurred during the chunk unassignment process",
                    exc_info=True,
                )
                raise

    return results


async def update_chunk(
    chunk_id: ObjectId,
    workspace_id: ObjectId,
    body: UpdateChunkModel,
    user_id: ObjectId,
    db: AsyncIOMotorDatabase,
) -> Tuple[
    str, List[ChunksOutWithWorkspaceDetails] | List[ChunkDocumentModel]
]:
    """Update a chunk.

    Todo
    ----
    - Review: Currently does not update the `updated_at` field as tags and user_metadata are at the
    workspace level.
    """
    # logger.info(f"Updating chunk: {chunk_id} within workspace: {workspace_id}")

    update_body = {}
    for k, v in body.model_dump(exclude_none=True).items():
        update_body[f"{k}.{workspace_id}"] = v

    update_chunk_obj = {"$set": update_body}

    updated_obj = await db.chunk.update_one(
        filter={
            "_id": ObjectId(chunk_id),
            "created_by": user_id,
            "workspaces": workspace_id,
        },
        update=update_chunk_obj,
    )
    if updated_obj.matched_count == 0:
        message = "No chunk found to update"
    elif updated_obj.modified_count == 0:
        message = "No changes made to the chunk"
    else:
        message = "Chunk updated successfully"

    chunks = await get_chunks(
        collection=db["chunk"],
        user_id=user_id,
        include_embeddings=False,
        filters={"_id": ObjectId(chunk_id)},
    )

    return message, chunks


async def delete_chunk(
    chunk_id: ObjectId,
    db_client: AsyncIOMotorClient,
    db: AsyncIOMotorDatabase,
    user_id: ObjectId,
) -> ChunkOut | None:
    """Delete a chunk.

    Deletes a chunk and unsets it from any nodes/triples that reference it.
    """
    async with await db_client.start_session() as session:
        async with session.start_transaction():

            chunk = await db.chunk.find_one(
                {"_id": chunk_id, "created_by": user_id},
                {"embedding": 0},
                session=session,
            )

            # Delete the chunk
            await db.chunk.delete_one(
                {"_id": chunk_id, "created_by": user_id},
                session=session,
            )

            # Unset chunk from nodes
            await perform_node_chunk_unassignment(
                db=db,
                session=session,
                chunk_ids_to_delete=[chunk_id],
                user_id=user_id,
            )
            # Unset chunk from triples
            await perform_triple_chunk_unassignment(
                db=db,
                session=session,
                chunk_ids_to_delete=[chunk_id],
                user_id=user_id,
            )

            # Commit the transaction
            await session.commit_transaction()

            if chunk:
                return ChunkOut(**chunk)
            return None


async def get_chunks_with_ws_and_doc_details(
    db: AsyncIOMotorDatabase,
    user_id: ObjectId,
    workspace_id: ObjectId | None = None,
    workspace_name: str | None = None,
    document_id: ObjectId | None = None,
    document_filename: str | None = None,
    data_type: str | None = None,
    skip: int = 0,
    limit: int = 10,
    order: int = 1,
    include_embeddings: bool = False,
) -> Tuple[List[ChunksOutWithWorkspaceDetails], int]:
    """Get chunks with populated workspace and document details."""
    if workspace_id and workspace_name:
        raise ValueError(
            "Both workspace_id and workspace_name cannot be provided."
        )
    if document_id and document_filename:
        raise ValueError(
            "Both document_id and document_filename cannot be provided."
        )

    collection = db["chunk"]
    pre_filters: Dict[str, Any] = {"created_by": user_id}
    post_filters: Dict[str, Any] = {}
    if data_type:
        pre_filters["data_type"] = data_type
    if workspace_id:
        pre_filters["workspaces"] = ObjectId(workspace_id)
    if document_id:
        pre_filters["document"] = ObjectId(document_id)
    if workspace_name:
        post_filters["workspaces.name"] = workspace_name
    if document_filename:
        post_filters["document.filename"] = document_filename

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
        {
            "$lookup": {
                "from": "document",
                "localField": "document",
                "foreignField": "_id",
                "as": "document",
            }
        },
        {
            "$set": {
                "workspaces": {
                    "$map": {
                        "input": "$workspaces",
                        "as": "workspace",
                        "in": {
                            "_id": "$$workspace._id",
                            "name": "$$workspace.name",
                        },
                    }
                },
                "document": {
                    "$map": {
                        "input": "$document",
                        "as": "doc",
                        "in": {
                            "_id": "$$doc._id",
                            "filename": {
                                "$getField": {
                                    "field": "filename",
                                    "input": "$$doc.metadata",
                                }
                            },
                        },
                    }
                },
            }
        },
        {
            "$unwind": {
                "path": "$document",
                "preserveNullAndEmptyArrays": True,
            }
        },
        {
            "$match": post_filters,
        },
        {
            "$facet": {
                "chunks": [
                    {"$sort": {"created_at": order, "_id": order}},
                    {"$skip": skip},
                    ({"$limit": limit} if limit != -1 else {}),
                ],
                "totalCount": [{"$count": "count"}],
            }
        },
        {
            "$project": {
                "chunks": 1,
                "totalCount": {"$arrayElemAt": ["$totalCount.count", 0]},
            }
        },
    ]

    if not include_embeddings:
        pipeline.insert(0, {"$project": {"embedding": 0}})

    chunks_and_count = await collection.aggregate(pipeline).to_list(
        length=None
    )
    if workspace_id:
        chunks_and_count[0]["chunks"] = [
            {
                **c,
                "user_metadata": c["user_metadata"].get(
                    str(c["workspaces"][0]["_id"]), {}
                ),
                "tags": c["tags"].get(str(c["workspaces"][0]["_id"]), []),
            }
            for c in chunks_and_count[0]["chunks"]
        ]
    chunks = [
        ChunksOutWithWorkspaceDetails(**c)
        for c in chunks_and_count[0].get("chunks", [])
    ]

    total_count = chunks_and_count[0].get("totalCount", 0)

    return chunks, total_count
