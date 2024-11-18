"""CRUD operations for Node model."""

import logging
from typing import Any, List

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from whyhow_api.models.common import LLMClient
from whyhow_api.schemas.chunks import ChunksOutWithWorkspaceDetails
from whyhow_api.schemas.nodes import NodeDocumentModel, NodeUpdate
from whyhow_api.services.crud.base import update_one
from whyhow_api.services.crud.triple import update_triple_embeddings

logger = logging.getLogger(__name__)


async def update_node(
    db: AsyncIOMotorDatabase,
    db_client: AsyncIOMotorClient,
    llm_client: LLMClient,
    user_id: ObjectId,
    node_id: ObjectId,
    node: NodeDocumentModel,
    update: NodeUpdate,
) -> NodeDocumentModel:
    """Update a node."""
    try:
        async with await db_client.start_session() as session:
            async with session.start_transaction():
                update.graph = ObjectId(update.graph) if update.graph else None
                updated_node = await update_one(
                    collection=db["node"],
                    document_model=NodeDocumentModel,
                    id=ObjectId(node.id),
                    document=update,
                    user_id=user_id,
                    session=session,
                )
                if updated_node is None:
                    raise ValueError(f"Node {node_id} not found.")

                # Update embeddings of all associated triples
                if update.name or update.type or update.properties:
                    triples = await db.triple.find(
                        {
                            "$or": [
                                {"head_node": node_id, "created_by": user_id},
                                {"tail_node": node_id, "created_by": user_id},
                            ]
                        }
                    ).to_list(None)

                    if triples:
                        await update_triple_embeddings(
                            db=db,
                            triple_ids=[
                                ObjectId(triple["_id"]) for triple in triples
                            ],
                            llm_client=llm_client,
                            user_id=user_id,
                            session=session,
                        )

                await session.commit_transaction()

        logger.info(f"Node {node_id} was successfully updated.")
        return NodeDocumentModel(**updated_node.model_dump())
    except Exception as e:
        logger.error(f"Failed to update node {node_id} due to error: {str(e)}")
        raise


async def delete_node(
    db: AsyncIOMotorDatabase,
    db_client: AsyncIOMotorClient,
    user_id: ObjectId,
    node_id: ObjectId,
) -> None:
    """Delete node including its associated triples."""
    try:
        async with await db_client.start_session() as session:
            async with session.start_transaction():
                # Delete the node
                await db.node.delete_one(
                    {"_id": node_id, "created_by": user_id}, session=session
                )

                # Delete associated triples e.g. those that connect to the node.
                await db.triple.delete_many(
                    {
                        "$or": [
                            {
                                "$and": [
                                    {"head_node": node_id},
                                    {"created_by": user_id},
                                ]
                            },
                            {
                                "$and": [
                                    {"tail_node": node_id},
                                    {"created_by": user_id},
                                ]
                            },
                        ]
                    },
                    session=session,
                )

                logger.info(f"Node {node_id} was successfully deleted.")
    except Exception as e:
        logger.error(f"Failed to delete node {node_id} due to error: {str(e)}")
        raise


async def get_node_chunks(
    db: AsyncIOMotorDatabase, id: ObjectId, user_id: ObjectId
) -> List[ChunksOutWithWorkspaceDetails]:
    """Get the chunks of a triple.

    Todo
    ----
    - Rethink the output model; should it be an array of workspaces, tags dict, etc?
    """
    pipeline: list[dict[str, Any]] = [
        {"$match": {"_id": ObjectId(id), "created_by": user_id}},
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
        {
            "$addFields": {
                "document": {
                    "$cond": {
                        "if": {"$ifNull": ["$documentInfo", False]},
                        "then": {
                            "_id": "$documentInfo._id",
                            "filename": "$documentInfo.metadata.filename",
                        },
                        "else": None,
                    }
                }
            }
        },
        {"$project": {"documentInfo": 0}},
    ]
    chunks = await db["node"].aggregate(pipeline).to_list(None)
    _chunks = [
        {
            **c,
            "user_metadata": c["user_metadata"].get(
                str(c["workspaces"][0]["_id"]), {}
            ),
            "tags": (
                {
                    k: v
                    for k, v in c["tags"].items()
                    if k == str(c["workspaces"][0]["_id"])
                }
                if c["tags"]
                else {}
            ),
        }
        for c in chunks
    ]

    return [ChunksOutWithWorkspaceDetails(**c) for c in _chunks]


async def get_nodes_by_ids(
    db: AsyncIOMotorDatabase,
    node_ids: list[ObjectId],
    graph_id: ObjectId,
    user_id: ObjectId,
) -> list[NodeDocumentModel]:
    """
    Get nodes by their ids.

    Parameters
    ----------
    db : AsyncIOMotorDatabase
        The database instance.
    node_ids : list[ObjectId]
        List of node ids.
    graph_id : ObjectId
        The graph id.
    user_id : ObjectId
        The user id.

    Returns
    -------
    list[NodeDocumentModel]
        List of node document models.
    """
    nodes = await db.node.find(
        {"_id": {"$in": node_ids}, "created_by": user_id, "graph": graph_id}
    ).to_list(None)
    return [NodeDocumentModel(**node) for node in nodes]
