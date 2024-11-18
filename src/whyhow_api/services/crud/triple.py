"""CRUD operations for triples."""

import logging
from typing import Any, Dict, List, Tuple

import logfire
from bson import ObjectId
from motor.core import AgnosticClientSession
from motor.motor_asyncio import AsyncIOMotorCollection, AsyncIOMotorDatabase
from pymongo import UpdateOne

from whyhow_api.models.common import LLMClient, Triple
from whyhow_api.schemas.chunks import ChunksOutWithWorkspaceDetails
from whyhow_api.utilities.common import clean_text

logger = logging.getLogger(__name__)


async def delete_triple(
    db: AsyncIOMotorDatabase,
    user_id: ObjectId,
    triple_id: ObjectId,
) -> None:
    """Delete a triple.

    Deletes a triple, allowing the nodes connected to it to be orphaned.
    """
    await db.triple.delete_one({"_id": triple_id, "created_by": user_id})


async def get_triple_chunks(
    collection: AsyncIOMotorCollection,
    id: ObjectId,
    user_id: ObjectId | None,
    graph_id: ObjectId | None,
    skip: int = 0,
    limit: int = 100,
    order: int = -1,
) -> Tuple[List[ChunksOutWithWorkspaceDetails], int]:
    """
    Get the chunks of a triple.

    Parameters
    ----------
    collection : AsyncIOMotorCollection
        The collection to query.
    id : ObjectId
        The ID of the triple.
    user_id : ObjectId
        The ID of the user.
    skip : int, optional
        The number of documents to skip, by default 0.
    limit : int, optional
        The number of documents to return, by default 100.
    order : int, optional
        The order of the documents, by default -1.

    Returns
    -------
    Tuple[List[ChunkDocumentModel], int]
        A tuple containing the list of chunks and the total number of documents.

    Todo
    ----
    - Rethink the output model; should it be an array of workspaces, tags dict, etc?
    """
    pipeline_1: List[Dict[str, Any]] = [
        {"$match": {"_id": ObjectId(id)}},
        {
            "$lookup": {
                "from": "graph",
                "localField": "graph",
                "foreignField": "_id",
                "as": "graph",
            }
        },
        {"$unwind": {"path": "$graph", "preserveNullAndEmptyArrays": True}},
        {
            "$lookup": {
                "from": "workspace",
                "localField": "graph.workspace",
                "foreignField": "_id",
                "as": "workspace",
            }
        },
        {
            "$unwind": {
                "path": "$workspace",
                "preserveNullAndEmptyArrays": True,
            }
        },
        {
            "$lookup": {
                "from": "chunk",
                "localField": "chunks",
                "foreignField": "_id",
                "as": "chunks",
            }
        },
        {"$project": {"chunks.embedding": 0}},
        {"$unwind": "$chunks"},
        {"$addFields": {"chunks.workspaces": ["$workspace"]}},
        {"$replaceRoot": {"newRoot": "$chunks"}},
        {
            "$lookup": {
                "from": "document",
                "localField": "document",
                "foreignField": "_id",
                "as": "documentInfo",
            }
        },
        {
            "$unwind": {
                "path": "$documentInfo",
                "preserveNullAndEmptyArrays": True,
            }
        },
        # TODO: optimise this.
        {
            "$addFields": {
                "document": {
                    "_id": "$documentInfo._id",
                    "filename": "$documentInfo.metadata.filename",
                }
            }
        },
        {
            "$addFields": {
                "document": {
                    "$cond": {
                        "if": {"$eq": ["$document", {}]},
                        "then": None,
                        "else": "$document",
                    }
                }
            }
        },
        {"$project": {"documentInfo": 0}},
    ]

    if user_id is None and graph_id is None:
        raise ValueError("Either user_id or graph_id must be provided.")

    if user_id:
        pipeline_1.insert(0, {"$match": {"created_by": user_id}})
    if graph_id:
        pipeline_1.insert(0, {"$match": {"graph": graph_id}})

    # Get the count of documents
    cursor = collection.aggregate(pipeline_1)
    total = len(await cursor.to_list(None))

    pipeline_2: List[Dict[str, Any]] = [
        {"$sort": {"created_at": order, "_id": order}},
        {"$skip": skip},
    ]

    if limit >= 0:
        pipeline_2.append({"$limit": limit})

    pipeline = pipeline_1 + pipeline_2

    db_chunks = await collection.aggregate(pipeline).to_list(None)

    chunks = [
        {
            **c,
            "user_metadata": c["user_metadata"].get(
                str(c["workspaces"][0]["_id"]), {}
            ),
            "tags": c["tags"].get(str(c["workspaces"][0]["_id"]), []),
        }
        for c in db_chunks
    ]

    return [ChunksOutWithWorkspaceDetails(**c) for c in chunks], total


def convert_triple_to_text(
    triple: dict[str, Any], include_chunks: bool
) -> str:
    """Convert a triple to a natural language text string.

    Convert a triple to a natural language text string including verbalization of any existing properties,
    while removing unwanted characters from property keys and values.

    Parameters
    ----------
    - triple (dict[str, Any]): A dictionary representing the triple with potential properties.
    - include_chunks (bool): A boolean flag to include chunk content in the text if True.

    Returns
    -------
    - str: A natural language string representation of the triple including its properties if they exist.
    """
    # Start constructing the sentence
    sentence = f"{clean_text(triple['head'])} which is a {clean_text(triple['head_type'])}"

    # Add properties to head if any
    if "head_properties" in triple and triple["head_properties"]:
        head_props = " with " + ", ".join(
            f"{clean_text(key)} of {clean_text(str(value))}"
            for key, value in triple["head_properties"].items()
        )
        sentence += head_props

    # Add the relation and the tail
    sentence += f" {clean_text(triple['relation'])} {clean_text(triple['tail'])}, a {clean_text(triple['tail_type'])}"

    # Add properties to tail if any
    if "tail_properties" in triple and triple["tail_properties"]:
        tail_props = " with " + ", ".join(
            f"{clean_text(key)} of {clean_text(str(value))}"
            for key, value in triple["tail_properties"].items()
        )
        sentence += tail_props

    # If there are relation properties, add them in a way that integrates into the sentence
    if "relation_properties" in triple and triple["relation_properties"]:
        relation_props = " due to " + ", ".join(
            f"{clean_text(key)} of {clean_text(str(value))}"
            for key, value in triple["relation_properties"].items()
        )
        sentence += relation_props

    # Optionally add chunks content
    try:
        if include_chunks and "chunks_content" in triple:
            chunks_content = [
                chunk["content"]
                for chunk in triple["chunks_content"]
                if "content" in chunk
            ]
            chunks_text_list = []
            for chunk in chunks_content:
                if isinstance(chunk, str):
                    # Handle plain text chunk
                    chunks_text_list.append(clean_text(chunk))
                elif isinstance(chunk, dict):
                    # Handle structured object chunk
                    structured_chunk_text = ", ".join(
                        f"{clean_text(key)}: {clean_text(str(value))}"
                        for key, value in chunk.items()
                    )
                    chunks_text_list.append(structured_chunk_text)
                else:
                    raise ValueError(
                        "chunk should be either a string or a dictionary"
                    )

            chunks_text = (
                ". This is further explained by the chunks: "
                + " | ".join(chunks_text_list)
            )
            sentence += chunks_text
            logger.debug("Chunks content added to the sentence.")
        else:
            logger.debug("No chunks content to add.")
    except Exception as e:
        logger.error(f"Error while adding chunks content: {e}")

    return sentence


async def embed_triples(
    llm_client: LLMClient,
    triples: list[dict[str, Any]],
    batch_size: int = 2048,
) -> list[list[float]]:
    """Embed triples with batching."""
    texts = [convert_triple_to_text(t, include_chunks=False) for t in triples]
    logger.info(f"Total triples converted to text: {len(texts)}")

    # Logfire trace of LLM client
    logfire.instrument_openai(llm_client.client)

    # Batch processing of embeddings
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        logger.info(f"Processing batch from {i} to {i + batch_size}")

        if len(batch_texts) > 2048:
            raise RuntimeError("Texts must be 2048 items or less.")

        # Get embeddings for the batch of texts
        response = await llm_client.client.embeddings.create(
            input=batch_texts,
            model=(
                llm_client.metadata.embedding_name
                if llm_client.metadata.embedding_name
                else "text-embedding-3-small"
            ),
            dimensions=1024,  # ONLY WORKS FOR TEXT-EMBEDDING-3-* models
        )
        # Extract embeddings and add to the list
        batch_embeddings = [d.embedding for d in response.data]
        all_embeddings.extend(batch_embeddings)

    return all_embeddings


async def update_triple_embeddings(
    db: AsyncIOMotorDatabase,
    llm_client: LLMClient,
    triple_ids: list[ObjectId],
    user_id: ObjectId,
    session: AgnosticClientSession | None = None,
) -> None:
    """Update the embedding of triples."""
    triples: List[dict[str, Any]] = await db.triple.aggregate(
        [
            {
                "$match": {
                    "_id": {"$in": triple_ids},
                    "created_by": ObjectId(user_id),
                }
            },
            {
                "$lookup": {
                    "from": "node",
                    "localField": "head_node",
                    "foreignField": "_id",
                    "as": "head_node",
                }
            },
            {
                "$lookup": {
                    "from": "node",
                    "localField": "tail_node",
                    "foreignField": "_id",
                    "as": "tail_node",
                }
            },
            {"$unwind": "$head_node"},
            {"$unwind": "$tail_node"},
            {
                "$project": {
                    "_id": 0,
                    "head": "$head_node.name",
                    "head_type": "$head_node.type",
                    "head_properties": "$head_node.properties",
                    "relation": "$type",
                    "relation_properties": "$properties",
                    "tail": "$tail_node.name",
                    "tail_type": "$tail_node.type",
                    "tail_properties": "$tail_node.properties",
                }
            },
        ],
        session=session,
    ).to_list(None)

    if not triples:
        raise ValueError("No triples found.")

    triple_embed_objs = [
        Triple(
            head=triple["head"],
            head_type=triple["head_type"],
            head_properties=triple["head_properties"],
            relation=triple["relation"],
            relation_properties=triple["relation_properties"],
            tail=triple["tail"],
            tail_type=triple["tail_type"],
            tail_properties=triple["tail_properties"],
        )
        for triple in triples
    ]

    # Embed the triples
    triple_embeddings = await embed_triples(
        llm_client=llm_client,
        triples=[triple.model_dump() for triple in triple_embed_objs],
    )

    # Update the triples with the embeddings
    update_operations = [
        UpdateOne(
            {"_id": ObjectId(triple_id), "created_by": user_id},
            {"$set": {"embedding": triple_embeddings[i]}},
        )
        for i, triple_id in enumerate(triple_ids)
    ]

    if update_operations:
        result = await db.triple.bulk_write(update_operations, session=session)

    if result.matched_count != len(triple_ids):
        raise ValueError("Triple embedding not updated for all triples.")
