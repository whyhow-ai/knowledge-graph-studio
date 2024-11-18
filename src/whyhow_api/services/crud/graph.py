"""CRUD operations for the graphs."""

import itertools
import logging
from typing import Any, Dict, List, Tuple

from bson import ObjectId
from motor.motor_asyncio import (
    AsyncIOMotorClient,
    AsyncIOMotorClientSession,
    AsyncIOMotorCollection,
    AsyncIOMotorDatabase,
)

from whyhow_api.schemas.chunks import ChunksOutWithWorkspaceDetails
from whyhow_api.schemas.graphs import DetailedGraphDocumentModel
from whyhow_api.schemas.nodes import NodeWithId
from whyhow_api.schemas.triples import TripleWithId

logger = logging.getLogger(__name__)


async def delete_graphs(
    db: AsyncIOMotorDatabase,
    user_id: ObjectId,
    graph_ids: list[ObjectId],
    db_client: AsyncIOMotorClient | None = None,
    session: AsyncIOMotorClientSession | None = None,
) -> None:
    """Delete a graph.

    Deletes a single graph and its associated triples,
    nodes, and queries.
    """
    if session is None:
        if db_client is None:
            raise ValueError("Either db_client or session must be provided.")

        async with await db_client.start_session() as new_session:
            async with new_session.start_transaction():
                await delete_graphs(
                    db=db,
                    user_id=user_id,
                    graph_ids=graph_ids,
                    session=new_session,
                )
                await new_session.commit_transaction()
    else:
        await db.triple.delete_many(
            {"graph": {"$in": graph_ids}, "created_by": user_id},
            session=session,
        )
        await db.node.delete_many(
            {"graph": {"$in": graph_ids}, "created_by": user_id},
            session=session,
        )
        await db.query.delete_many(
            {"graph": {"$in": graph_ids}, "created_by": user_id},
            session=session,
        )
        await db.graph.delete_many(
            {"_id": {"$in": graph_ids}, "created_by": user_id}, session=session
        )
        logger.info(
            f"All related items for graphs {graph_ids} were successfully deleted."
        )


async def list_relations(
    collection: AsyncIOMotorCollection,
    user_id: ObjectId | None,
    graph_id: ObjectId,
    skip: int = 0,
    limit: int = 100,
    order: int = -1,
) -> Tuple[List[str], int]:
    """List graph relations.

    List all of the distinct relations on the provided graph by name.

    Parameters
    ----------
    collection : AsyncIOMotorCollection
        The collection where triples are stored to query.
    graph_id : ObjectId
        The ID of the graph to retrieve.
    skip : int, optional
        Number of documents to skip.
    limit : int, optional
        Number of documents to limit the results to.
    order : int, optional
        Sort order, -1 for descending, 1 for ascending.

    Returns
    -------
    Tuple[List[str], int]
        A list of relations and the total number of relations.
    """
    pipeline_1: list[dict[str, Any]] = [
        {"$match": {"graph": graph_id, "type": {"$ne": "Contains"}}},
        {
            "$group": {
                "_id": "$type",
                "properties": {"$first": "$properties"},
            }
        },
    ]

    if user_id:
        pipeline_1.insert(0, {"$match": {"created_by": user_id}})

    # Calculate the number of documents
    cursor = collection.aggregate(pipeline_1)
    total = len(await cursor.to_list(length=None))

    pipeline_2: list[dict[str, Any]] = [
        {
            "$project": {
                "_id": 0,
                "type": "$_id",
                "properties": 1,
            }
        },
        {"$sort": {"created_at": order, "type": order}},
        {"$skip": skip},
    ]
    if limit >= 0:
        pipeline_2.append({"$limit": limit})

    pipeline = pipeline_1 + pipeline_2
    cursor = collection.aggregate(pipeline)
    relations = [t["type"] for t in await cursor.to_list(length=None)]

    return relations, total


async def list_nodes(
    collection: AsyncIOMotorCollection,
    user_id: ObjectId | None,
    graph_id: ObjectId,
    skip: int = 0,
    limit: int = 100,
    order: int = -1,
) -> tuple[List[NodeWithId], int]:
    """List graph nodes.

    List all of the distinct nodes on the provided graph by name.

    Parameters
    ----------
    collection : AsyncIOMotorCollection
        The collection where nodes are stored to query.
    graph_id : ObjectId
        The ID of the graph to retrieve.
    skip : int, optional
        Number of documents to skip.
    limit : int, optional
        Number of documents to limit the results to.
    order : int, optional
        Sort order, -1 for descending, 1 for ascending.

    Returns
    -------
    List[NodeWithId]
        A list of nodes.
    """
    nodes_pipeline = [
        {"$sort": {"created_at": order, "id": order}},
        {"$skip": skip},
    ]
    if limit != -1:
        nodes_pipeline.append({"$limit": limit})
    if user_id is not None:
        nodes_pipeline.insert(0, {"$match": {"created_by": user_id}})

    pipeline: list[dict[str, Any]] = [
        {"$match": {"graph": graph_id}},
        {
            "$project": {
                "id": "$_id",
                "name": "$name",
                "label": "$type",
                "properties": 1,
                "chunks": 1,
            }
        },
        {
            "$facet": {
                "nodes": nodes_pipeline,
                "totalCount": [{"$count": "count"}],
            }
        },
        {
            "$project": {
                "nodes": 1,
                "totalCount": {"$arrayElemAt": ["$totalCount.count", 0]},
            }
        },
    ]

    nodes_and_count = await collection.aggregate(pipeline).to_list(None)

    nodes = [NodeWithId(**n) for n in nodes_and_count[0].get("nodes", [])]
    total_count = nodes_and_count[0].get("totalCount", 0)

    return nodes, total_count


async def get_graph(
    collection: AsyncIOMotorCollection,
    graph_id: ObjectId,
    user_id: ObjectId | None,
    public: bool = False,
) -> DetailedGraphDocumentModel | None:
    """Get graph.

    Retrieve one distinct graph for the user.

    Parameters
    ----------
    collection : AsyncIOMotorCollection
        The MongoDB collection where graphs are stored.
    graph_id : ObjectId
        The ID of the graph to retrieve.
    user_id : ObjectId
        The ID of the user to filter by.

    Returns
    -------
    DetailedGraphDocumentModel | None
        The graph if found, None otherwise.
    """
    pipeline: list[dict[str, Any]] = [
        {
            "$match": {
                "_id": graph_id,
                **({"created_by": user_id} if user_id else {}),
                **({"public": True} if public else {}),
            }
        },
        {
            "$lookup": {
                "from": "schema",
                "localField": "schema_id",
                "foreignField": "_id",
                "as": "_schema",
            }
        },
        {
            "$lookup": {
                "from": "workspace",
                "localField": "workspace",
                "foreignField": "_id",
                "as": "workspace",
            }
        },
        {"$unwind": "$_schema"},
        {"$unwind": "$workspace"},
        {
            "$addFields": {
                "schema_": {
                    "_id": {"$toString": "$_schema._id"},
                    "name": "$_schema.name",
                }
            }
        },
        {
            "$project": {
                "schema_id": 0,
                "_schema": 0,
                "workspace.created_at": 0,
                "workspace.updated_at": 0,
                "workspace.created_by": 0,
            }
        },
    ]

    cursor = await collection.aggregate(pipeline).to_list(length=None)

    if cursor is None or len(cursor) == 0:
        logger.info("No graph data found")
        return None

    return DetailedGraphDocumentModel(**cursor[0])


async def list_all_graphs(
    collection: AsyncIOMotorCollection,
    user_id: ObjectId,
    filters: Dict[str, Any] | None = None,
    skip: int = 0,
    limit: int = 100,
    order: int = -1,
) -> List[DetailedGraphDocumentModel] | None:
    """
    List all graphs.

    List all of the distinct graphs for the user.

    Parameters
    ----------
    collection : AsyncIOMotorCollection
        The MongoDB collection where graphs are stored.
    user_id : ObjectId
        The ID of the user to filter by.
    filters : dict, optional
        Additional filters to apply to the query.
    skip : int, optional
        Number of documents to skip.
    limit : int, optional
        Number of documents to limit the results to.
    order : int, optional
        Sort order, -1 for descending, 1 for ascending.

    Returns
    -------
    List[DetailedGraphDocumentModel] | None
        A list of graphs or None if no graphs are found.
    """
    query = {"$match": {"created_by": user_id}}
    if filters:
        query["$match"].update(filters)

    pipeline: list[dict[str, Any]] = [
        query,
        {
            "$lookup": {
                "from": "schema",
                "localField": "schema_id",
                "foreignField": "_id",
                "as": "_schema",
            }
        },
        {
            "$lookup": {
                "from": "workspace",
                "localField": "workspace",
                "foreignField": "_id",
                "as": "workspace",
            }
        },
        {"$unwind": "$_schema"},
        {"$unwind": "$workspace"},
        {
            "$addFields": {
                "schema_": {
                    "_id": {"$toString": "$_schema._id"},
                    "name": "$_schema.name",
                }
            }
        },
        {
            "$project": {
                "schema_id": 0,
                "_schema": 0,
                "workspace.created_at": 0,
                "workspace.updated_at": 0,
                "workspace.created_by": 0,
            }
        },
        {"$sort": {"created_at": order, "_id": order}},
        {"$skip": skip},
    ]
    if limit >= 0:
        pipeline.append({"$limit": limit})

    graphs = await collection.aggregate(pipeline).to_list(length=None)
    graph_out = [DetailedGraphDocumentModel(**g) for g in graphs]

    if graph_out is None:
        logger.info("No graph data found")
        return None

    return graph_out


async def list_triples(
    collection: AsyncIOMotorCollection,
    user_id: ObjectId | None,
    graph_id: ObjectId,
    skip: int = 0,
    limit: int = 100,
    order: int = -1,
) -> Tuple[List[TripleWithId], int]:
    """
    List graph triples and calculate their total count.

    List all the distinct triples in the graph and calculate the total count of triples.

    Parameters
    ----------
    collection : AsyncIOMotorCollection
        The collection where triples are stored to query.
    graph_id : ObjectId
        The ID of the graph to retrieve.
    skip : int, optional
        Number of documents to skip.
    limit : int, optional
        Number of documents to limit the results to.
    order : int, optional
        Sort order, -1 for descending, 1 for ascending.

    Returns
    -------
    Tuple[List[TripleWithId], int]
        A list of triples and the total number of triples.
    """
    query = {"graph": graph_id, "type": {"$ne": "Contains"}}

    if user_id:
        query["created_by"] = user_id

    # Create base pipeline
    base_pipeline: list[dict[str, Any]] = [
        {"$match": query},
        {
            "$lookup": {
                "from": "node",
                "localField": "head_node",
                "foreignField": "_id",
                "as": "head_node",
            }
        },
        {
            "$unwind": {
                "path": "$head_node",
                "preserveNullAndEmptyArrays": True,
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
        {
            "$unwind": {
                "path": "$tail_node",
                "preserveNullAndEmptyArrays": True,
            }
        },
        {
            "$project": {
                "_id": 1,
                "head_node": {
                    "_id": "$head_node._id",
                    "name": "$head_node.name",
                    "label": "$head_node.type",
                    "properties": "$head_node.properties",
                    "chunks": "$head_node.chunks",
                },
                "relation": {
                    "name": "$type",
                    "properties": "$properties",
                },
                "tail_node": {
                    "_id": "$tail_node._id",
                    "name": "$tail_node.name",
                    "label": "$tail_node.type",
                    "properties": "$tail_node.properties",
                    "chunks": "$tail_node.chunks",
                },
                "chunks": "$chunks",
            }
        },
    ]

    # Add count stage
    base_pipeline.append(
        {
            "$facet": {
                "triples": [
                    {"$sort": {"created_at": order, "_id": order}},
                    {"$skip": skip},
                ],
                "total_count": [{"$count": "total"}],
            }
        }
    )
    if limit >= 0:
        base_pipeline[-1]["$facet"]["triples"].append({"$limit": limit})

    # Execute the aggregation pipeline
    cursor = collection.aggregate(base_pipeline)
    result = await cursor.to_list(length=1)

    if not result:
        return [], 0

    triples = [TripleWithId(**f) for f in result[0]["triples"]]
    total_count = (
        result[0]["total_count"][0]["total"] if result[0]["total_count"] else 0
    )
    return triples, total_count


async def list_triples_by_ids(
    db: AsyncIOMotorDatabase,
    user_id: ObjectId,
    graph_id: ObjectId,
    triple_ids: List[ObjectId],
) -> List[TripleWithId]:
    """List graph triples by IDs.

    List all the distinct triples in the graph by their IDs.
    """
    query = {
        "created_by": user_id,
        "graph": graph_id,
        "_id": {"$in": triple_ids},
    }

    # Fetch all relationships from the database
    pipeline: list[dict[str, Any]] = [
        {"$match": query},
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
        {
            "$project": {
                "head": {"$arrayElemAt": ["$head_node.name", 0]},
                "head_type": {"$arrayElemAt": ["$head_node.type", 0]},
                "head_id": {"$arrayElemAt": ["$head_node._id", 0]},
                "head_properties": {
                    "$arrayElemAt": ["$head_node.properties", 0]
                },
                "head_chunks": {"$arrayElemAt": ["$head_node.chunks", 0]},
                "relation": "$type",
                "relation_properties": "$properties",
                "relation_chunks": "$chunks",
                "tail": {"$arrayElemAt": ["$tail_node.name", 0]},
                "tail_type": {"$arrayElemAt": ["$tail_node.type", 0]},
                "tail_id": {"$arrayElemAt": ["$tail_node._id", 0]},
                "tail_properties": {
                    "$arrayElemAt": ["$tail_node.properties", 0]
                },
                "tail_chunks": {"$arrayElemAt": ["$tail_node.chunks", 0]},
            }
        },
        {
            "$replaceRoot": {
                "newRoot": {
                    "_id": "$_id",
                    "chunks": "$relation_chunks",
                    "head_node": {
                        "_id": "$head_id",
                        "name": "$head",
                        "label": "$head_type",
                        "properties": "$head_properties",
                        "chunks": "$head_chunks",
                    },
                    "relation": {
                        "name": "$relation",
                        "properties": "$relation_properties",
                    },
                    "tail_node": {
                        "_id": "$tail_id",
                        "name": "$tail",
                        "label": "$tail_type",
                        "properties": "$tail_properties",
                        "chunks": "$tail_chunks",
                    },
                }
            }
        },
    ]

    cursor = db.triple.aggregate(pipeline)
    triples = [TripleWithId(**f) for f in await cursor.to_list(length=None)]
    return triples


async def get_graph_chunks(
    db: AsyncIOMotorDatabase,
    user_id: ObjectId | None,
    graph_id: ObjectId,
    workspace_id: ObjectId,
    skip: int = 0,
    limit: int = 100,
    order: int = -1,
) -> Tuple[list[ChunksOutWithWorkspaceDetails], int]:
    """Get graph chunks.

    Todo
    ----
    - Optimise these mongodb operations.
    """
    logger.info(
        f"fetching chunks for graph {graph_id} in workspace {workspace_id} skip={skip} limit={limit}"
    )

    node_triple_match = {"graph": graph_id}
    if user_id:
        node_triple_match["created_by"] = user_id

    node_chunks = await db.node.find(node_triple_match, {"chunks": 1}).to_list(
        None
    )
    triple_chunks = await db.triple.find(
        node_triple_match, {"chunks": 1}
    ).to_list(None)

    chunk_ids = list(
        set(
            itertools.chain.from_iterable(
                chunks["chunks"] for chunks in node_chunks + triple_chunks
            )
        )
    )

    pipeline: list[dict[str, Any]] = [
        {"$project": {"embedding": 0}},
        {"$match": {"_id": {"$in": chunk_ids}}},
    ]

    if user_id:
        pipeline.insert(
            0, {"$match": {"created_by": user_id, "workspaces": workspace_id}}
        )

    chunks_pipeline = [
        {"$sort": {"created_at": order}},
        {"$skip": skip},
    ]

    if limit >= 0:
        chunks_pipeline.append({"$limit": limit})

    chunks_pipeline.extend(
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
                    "as": "documentInfo",
                }
            },
            {
                "$addFields": {
                    "document": {
                        "_id": {"$arrayElemAt": ["$documentInfo._id", 0]},
                        "filename": {
                            "$arrayElemAt": [
                                "$documentInfo.metadata.filename",
                                0,
                            ]
                        },
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
    )

    pipeline.extend(
        [
            {
                "$facet": {
                    "chunks": chunks_pipeline,
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
    )

    db_chunks_and_count = await db.chunk.aggregate(pipeline).to_list(None)
    db_chunks = db_chunks_and_count[0].get("chunks", [])
    total_count = db_chunks_and_count[0].get("totalCount", 0)

    chunks = [
        {
            **c,
            "workspaces": [
                w for w in c["workspaces"] if w["_id"] == workspace_id
            ],
            "user_metadata": c["user_metadata"].get(str(workspace_id), {}),
            "tags": c["tags"].get(str(workspace_id), []),
        }
        for c in db_chunks
    ]

    return [ChunksOutWithWorkspaceDetails(**c) for c in chunks], total_count
