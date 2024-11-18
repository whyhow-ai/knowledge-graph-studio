"""Node CRUD routes."""

import logging
from typing import Annotated, Any, Dict, List

from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException, Query
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from whyhow_api.dependencies import (
    get_db,
    get_db_client,
    get_llm_client,
    get_user,
    valid_graph_id,
    valid_node_id,
)
from whyhow_api.models.common import LLMClient
from whyhow_api.schemas.nodes import (
    NodeChunksResponse,
    NodeCreate,
    NodeDocumentModel,
    NodeOut,
    NodesResponse,
    NodeUpdate,
)
from whyhow_api.schemas.schemas import SchemaDocumentModel
from whyhow_api.services.crud.base import (
    create_one,
    get_all,
    get_all_count,
    get_one,
)
from whyhow_api.services.crud.node import (
    delete_node,
    get_node_chunks,
    update_node,
)
from whyhow_api.services.graph_service import extend_schema
from whyhow_api.utilities.routers import order_query

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Nodes"], prefix="/nodes")


def get_all_nodes_response(
    nodes: list[NodeOut], total_count: int
) -> NodesResponse:
    """Get all nodes response."""
    return NodesResponse(
        message="Nodes retrieved successfully",
        status="success",
        nodes=nodes,
        count=total_count,
    )


def get_node_response(node: NodeOut) -> NodesResponse:
    """Get node response."""
    return NodesResponse(
        message="Node retrieved successfully",
        status="success",
        count=1,
        nodes=[node],
    )


def update_node_response(node: NodeOut) -> NodesResponse:
    """Update node response."""
    return NodesResponse(
        message="Node updated successfully",
        status="success",
        count=1,
        nodes=[node],
    )


@router.post("", response_model=NodesResponse)
async def create_node(
    body: NodeCreate,
    db: AsyncIOMotorDatabase = Depends(get_db),
    user_id: ObjectId = Depends(get_user),
) -> NodesResponse:
    """Create node."""
    graph = await valid_graph_id(
        graph_id=str(body.graph), user_id=user_id, db=db
    )

    # Check whether the provided node type is supported by the graph schema.
    # If `body.strict_mode` is True, invalid node will throw an exception.
    # If `body.strict_mode` is False, invalid node will be added and the schema will be extended.
    schema = await get_one(
        collection=db["schema"],
        user_id=user_id,
        document_model=SchemaDocumentModel,
        id=ObjectId(graph.schema_.id),
    )
    schema = SchemaDocumentModel.model_validate(schema)
    if schema is None:
        raise HTTPException(
            status_code=404,
            detail="Graph schema not found",
        )
    entities = {e.name for e in schema.entities}

    if body.type not in entities:
        if body.strict_mode:
            raise HTTPException(
                status_code=400,
                detail=f"Node type '{body.type}' is not supported by the graph schema. Allowed types: {list(entities)}",
            )
        else:
            await extend_schema(
                db=db,
                schema_id=ObjectId(graph.schema_.id),
                user_id=user_id,
                entity_types=set([body.type]),
            )

    # Check if node already exists in the graph
    node = await db.node.find_one(
        {
            "name": body.name,
            "type": body.type,
            "created_by": user_id,
            "graph": ObjectId(graph.id),
        }
    )

    if node:
        raise HTTPException(
            status_code=400,
            detail=f"Node with name '{body.name}' and type '{body.type}' already exists in the graph '{graph.name}' ({graph.id}).",
        )

    node = await create_one(
        collection=db["node"],
        document_model=NodeDocumentModel,
        user_id=user_id,
        document=body,
    )
    return NodesResponse(
        message="Node created successfully",
        status="success",
        count=1,
        nodes=[NodeOut.model_validate(node)],
    )


@router.get("", response_model=NodesResponse)
async def read_nodes_endpoint(
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=-1, le=50),
    order: int = Depends(order_query),
    name: Annotated[
        str | None, Query(description="The name of the node(s)")
    ] = None,
    type: Annotated[
        str | None, Query(description="The type of the nodes(s)")
    ] = None,
    workspace_name: Annotated[
        str | None,
        Query(
            description="The name of the workspace that contains the node(s)"
        ),
    ] = None,
    workspace_id: Annotated[
        str | None,
        Query(description="The id of the workspace that contains the node(s)"),
    ] = None,
    graph_name: Annotated[
        str | None,
        Query(description="The name of the graph that contains the node(s)"),
    ] = None,
    graph_id: Annotated[
        str | None,
        Query(description="The id of the graph that contains the node(s)"),
    ] = None,
    chunk_ids: Annotated[
        List[str] | None,
        Query(description="The ids of the chunks that contains the node(s)"),
    ] = None,
    db: AsyncIOMotorDatabase = Depends(get_db),
    user_id: ObjectId = Depends(get_user),
) -> NodesResponse:
    """Read nodes."""
    if graph_name and not (workspace_name or workspace_id):
        raise HTTPException(
            status_code=400,
            detail="Workspace name or id is required when graph name is provided.",
        )

    collection = db["node"]
    pre_filters: Dict[str, Any] = {}
    post_filters: Dict[str, Any] = {}

    if name:
        pre_filters["name"] = name
    if type:
        pre_filters["type"] = type
    if graph_id:
        pre_filters["graph"] = ObjectId(graph_id)
    if workspace_id:
        post_filters["graph.workspace._id"] = ObjectId(workspace_id)
    if workspace_name:
        post_filters["graph.workspace.name"] = workspace_name
    if graph_name:
        post_filters["graph.name"] = graph_name
    if chunk_ids:
        pre_filters["chunks"] = {"$in": [ObjectId(c_id) for c_id in chunk_ids]}

    pipeline = [
        {"$match": pre_filters},
        {
            "$lookup": {
                "from": "graph",
                "localField": "graph",
                "foreignField": "_id",
                "as": "graph",
            }
        },
        {
            "$unwind": {
                "path": "$graph",
                "preserveNullAndEmptyArrays": False,
            }
        },
        {
            "$lookup": {
                "from": "workspace",
                "localField": "graph.workspace",
                "foreignField": "_id",
                "as": "graph.workspace",
            }
        },
        {
            "$unwind": {
                "path": "$graph.workspace",
                "preserveNullAndEmptyArrays": False,
            }
        },
        {"$match": post_filters},
        {
            "$addFields": {"graph": "$graph._id"}
        },  # TODO: determine whether we want to populate the graph field with _id and name.
    ]

    nodes = await get_all(
        collection=collection,
        document_model=NodeDocumentModel,
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

    return get_all_nodes_response(
        nodes=[NodeOut.model_validate(n) for n in nodes],
        total_count=total_count,
    )


@router.get("/{node_id}", response_model=NodesResponse)
async def read_node_endpoint(
    node: NodeDocumentModel = Depends(valid_node_id),
) -> NodesResponse:
    """Read node."""
    return get_node_response(NodeOut.model_validate(node))


@router.put("/{node_id}", response_model=NodesResponse)
async def update_node_endpoint(
    body: NodeUpdate,
    node: NodeDocumentModel = Depends(valid_node_id),
    db: AsyncIOMotorDatabase = Depends(get_db),
    db_client: AsyncIOMotorClient = Depends(get_db_client),
    user_id: ObjectId = Depends(get_user),
    llm_client: LLMClient = Depends(get_llm_client),
) -> NodesResponse:
    """Update node."""
    updated_node = await update_node(
        db=db,
        db_client=db_client,
        llm_client=llm_client,
        user_id=user_id,
        node_id=ObjectId(node.id),
        node=node,
        update=body,
    )
    return update_node_response(NodeOut.model_validate(updated_node))


@router.delete(
    "/{node_id}",
    response_model=NodesResponse,
    description="Delete node and associated triples and orphaned nodes",
    response_model_exclude_none=True,
)
async def delete_node_endpoint(
    node: NodeDocumentModel = Depends(valid_node_id),
    db: AsyncIOMotorDatabase = Depends(get_db),
    db_client: AsyncIOMotorClient = Depends(get_db_client),
    user_id: ObjectId = Depends(get_user),
) -> NodesResponse:
    """Delete node and associated triples."""
    await delete_node(
        db=db,
        db_client=db_client,
        user_id=user_id,
        node_id=ObjectId(node.id),
    )

    return NodesResponse(
        message="Node deleted successfully",
        status="success",
        nodes=[NodeOut.model_validate(node)],
        count=1,
    )


@router.get(
    "/{node_id}/chunks",
    response_model=NodeChunksResponse,
)
async def read_node_with_chunks_endpoint(
    node: NodeDocumentModel = Depends(valid_node_id),
    user_id: ObjectId = Depends(get_user),
    db: AsyncIOMotorDatabase = Depends(get_db),
) -> NodeChunksResponse:
    """Read node with chunks."""
    node_chunks = await get_node_chunks(
        db=db, id=ObjectId(node.id), user_id=user_id
    )

    if node_chunks is None:
        raise HTTPException(
            status_code=404,
            detail="No chunks found for the node.",
        )

    return NodeChunksResponse(
        message="Node with chunks retrieved successfully.",
        status="success",
        count=len(node_chunks),
        chunks=node_chunks,
    )
