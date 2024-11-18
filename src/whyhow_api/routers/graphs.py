"""Graphs router."""

import logging
from typing import Annotated, Any, Dict, List

from bson import ObjectId
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    HTTPException,
    Query,
    status,
)
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo.errors import DuplicateKeyError

from whyhow_api.config import Settings
from whyhow_api.dependencies import (  # valid_create_graph,
    LLMClient,
    get_db,
    get_db_client,
    get_llm_client,
    get_settings,
    get_user,
    valid_graph_id,
    valid_public_graph_id,
)
from whyhow_api.exceptions import NotFoundException
from whyhow_api.schemas.base import Default_Entity_Type, Graph_Status
from whyhow_api.schemas.chunks import (
    ChunksResponseWithWorkspaceDetails,
    PublicChunksResponseWithWorkspaceDetails,
)
from whyhow_api.schemas.graphs import (
    AddChunksToGraphBody,
    CreateGraphBody,
    CreateGraphDetailsResponse,
    CreateGraphFromTriplesBody,
    CypherResponse,
    DetailedGraphDocumentModel,
    DetailedGraphOut,
    DetailedGraphsResponse,
    GraphDocumentModel,
    GraphOut,
    GraphsDetailedNodeResponse,
    GraphsDetailedTripleResponse,
    GraphsResponse,
    GraphsSimilarNodesResponse,
    GraphUpdate,
    MergeNodesRequest,
    PublicDetailedGraphOut,
    PublicGraphsDetailedNodeResponse,
    PublicGraphsResponse,
    PublicGraphsTripleResponse,
    QueryGraphRequest,
)
from whyhow_api.schemas.queries import QueryOut
from whyhow_api.schemas.rules import (
    MergeNodesRule,
    RuleCreate,
    RuleOut,
    RulesResponse,
)
from whyhow_api.schemas.workspaces import WorkspaceDocumentModel
from whyhow_api.services import graph_service
from whyhow_api.services.crud.base import (
    get_all,
    get_all_count,
    get_one,
    update_one,
)
from whyhow_api.services.crud.graph import (
    delete_graphs,
    get_graph_chunks,
    list_nodes,
    list_relations,
    list_triples,
)
from whyhow_api.services.crud.node import get_nodes_by_ids
from whyhow_api.services.crud.rule import create_rule, get_graph_rules
from whyhow_api.services.graph_service import MixedQueryProcessor
from whyhow_api.utilities.routers import order_query

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Graphs"], prefix="/graphs")


def update_graph_response(graph: GraphOut) -> GraphsResponse:
    """Update graph response."""
    return GraphsResponse(
        message="Graph updated successfully.",
        status="success",
        graphs=[graph],
        count=1,
    )


@router.get(
    "", response_model=DetailedGraphsResponse, response_model_exclude_none=True
)
async def list_all_graphs_endpoint(
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=-1, le=50),
    order: int = Depends(order_query),
    name: Annotated[
        str | None, Query(description="The name of the graph")
    ] = None,
    workspace_id: Annotated[
        str | None, Query(description="The id of the workspace")
    ] = None,
    workspace_name: Annotated[
        str | None, Query(description="The name of the workspace")
    ] = None,
    schema_id: Annotated[
        str | None, Query(description="The id of the schema")
    ] = None,
    schema_name: Annotated[
        str | None, Query(description="The name of the schema")
    ] = None,
    status: Annotated[
        Graph_Status | None, Query(description="The status of the graph")
    ] = None,
    db: AsyncIOMotorDatabase = Depends(get_db),
    user_id: ObjectId = Depends(get_user),
) -> DetailedGraphsResponse:
    """Read graphs."""
    if workspace_id and workspace_name:
        raise HTTPException(
            status_code=400,
            detail="Both workspace_id and workspace_name cannot be provided.",
        )
    if schema_id and schema_name:
        raise HTTPException(
            status_code=400,
            detail="Both schema_id and schema_name cannot be provided.",
        )

    collection = db["graph"]
    pre_filters: Dict[str, Any] = {}
    post_filters: Dict[str, Any] = {}
    if name:
        pre_filters["name"] = {
            "$regex": name,
            "$options": "i",
        }  # Case-insensitive search
    if status:
        pre_filters["status"] = status
    if workspace_id:
        pre_filters["workspace"] = ObjectId(workspace_id)
    if schema_id:
        pre_filters["schema_id"] = ObjectId(schema_id)

    if workspace_name:
        post_filters["workspace.name"] = {
            "$regex": workspace_name,
            "$options": "i",
        }  # Case-insensitive search
        logger.info(f"Workspace name: {workspace_name}")
    if schema_name:
        post_filters["schema_.name"] = {
            "$regex": schema_name,
            "$options": "i",
        }  # Case-insensitive search

    pipeline: List[Dict[str, Any]] = [
        {"$match": pre_filters},
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
        {"$match": post_filters},
    ]

    # Get total count of items in db
    total_count = await get_all_count(
        collection=collection, user_id=user_id, aggregation_query=pipeline
    )

    if total_count == 0:
        return DetailedGraphsResponse(
            message="No graphs found.",
            status="success",
            graphs=[],
            count=total_count,
        )
    else:
        graphs = await get_all(
            collection=collection,
            document_model=DetailedGraphDocumentModel,
            user_id=user_id,
            aggregation_query=pipeline,
            skip=skip,
            limit=limit,
            order=order,
        )

        return DetailedGraphsResponse(
            message="Successfully retrieved graph.",
            status="success",
            graphs=[
                DetailedGraphOut.model_validate(g.model_dump(by_alias=True))
                for g in (graphs if graphs is not None else [])
            ],
            count=total_count,
        )


@router.get(
    "/{graph_id}",
    response_model=DetailedGraphsResponse,
    # response_model_exclude_none=True,
)
async def read_graph_endpoint(
    graph: DetailedGraphDocumentModel = Depends(valid_graph_id),
) -> DetailedGraphsResponse:
    """Read graph."""
    return DetailedGraphsResponse(
        message="Successfully retrieved graph.",
        status="success",
        count=1,
        graphs=[
            DetailedGraphOut.model_validate(graph.model_dump(by_alias=True))
        ],
    )


@router.get(
    "/public/{graph_id}",
    response_model_exclude_none=True,
    response_model=DetailedGraphsResponse,
)
async def read_public_graph_endpoint(
    graph: GraphDocumentModel = Depends(valid_public_graph_id),
) -> DetailedGraphsResponse:
    """Read a public graph."""
    return DetailedGraphsResponse(
        message="Successfully retrieved graph.",
        status="success",
        count=1,
        graphs=[
            PublicDetailedGraphOut.model_validate(
                graph.model_dump(by_alias=True)
            )
        ],
    )


@router.put(
    "/add_chunks",
    response_model=GraphsResponse,
    response_model_exclude_none=True,
)
async def add_chunks_to_graph_endpoint(
    background_tasks: BackgroundTasks,
    body: AddChunksToGraphBody,
    db: AsyncIOMotorDatabase = Depends(get_db),
    db_client: AsyncIOMotorClient = Depends(get_db_client),
    user_id: ObjectId = Depends(get_user),
    llm_client: LLMClient = Depends(get_llm_client),
    settings: Settings = Depends(get_settings),
) -> GraphsResponse:
    """Add chunks to a graph."""
    graph = await db.graph.find_one(
        {"created_by": user_id, "_id": ObjectId(body.graph)}
    )
    if graph is None:
        logger.info(f"Graph with id '{body.graph}' does not exist.")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Graph not found.",
        )

    background_tasks.add_task(
        graph_service.create_or_update_graph,
        db=db,
        db_client=db_client,
        llm_client=llm_client,
        user_id=user_id,
        graph_id=ObjectId(body.graph),
        workspace_id=ObjectId(graph["workspace"]),
        schema_id=ObjectId(graph["schema_id"]),
        settings=settings,
        filters=body.filters,
    )
    return GraphsResponse(
        message="Hold tight - your graph is being created!",
        status="success",
        graphs=[GraphOut.model_validate(graph)],
        count=1,
    )


@router.post(
    "/from_triples",
    response_model=GraphsResponse,
    response_model_exclude_none=True,
)
async def create_graph_from_triples_endpoint(
    background_tasks: BackgroundTasks,
    body: CreateGraphFromTriplesBody,
    db: AsyncIOMotorDatabase = Depends(get_db),
    db_client: AsyncIOMotorClient = Depends(get_db_client),
    user_id: ObjectId = Depends(get_user),
    llm_client: LLMClient = Depends(get_llm_client),
) -> GraphsResponse:
    """Build a graph from triples.

    Builds a graph from triples, if a schema is not provided, one is auto-generated.
    Provided triples are validated against the schema before the graph is built.
    The chunks of head nodes, relations, and tail nodes can be provided by passing
    the chunk IDs as a list to the "chunks" field in the "head_properties",
    "relation_properties", and "tail_properties" respectively.
    """
    try:
        graph = await graph_service.create_base_graph(
            name=body.name,
            user_id=user_id,
            workspace_id=ObjectId(body.workspace),
            schema_id=ObjectId(body.schema_) if body.schema_ else None,
            db=db,
        )
        await graph_service.create_or_update_graph_from_triples(
            background_tasks=background_tasks,
            triples=body.triples,
            db=db,
            db_client=db_client,
            user_id=user_id,
            llm_client=llm_client,
            graph_id=ObjectId(graph.id) if graph.id else None,
        )
        return GraphsResponse(
            message="Hold tight - your graph is being built!",
            status="success",
            graphs=[GraphOut.model_validate(graph)],
            count=1,
        )
    except DuplicateKeyError as dke:
        logger.info(f"Error: {dke}")
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Graph with the same name already exists for this workspace.",
        )
    except NotFoundException as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except ValueError as e:
        await delete_graphs(
            user_id=user_id,
            graph_ids=[ObjectId(graph.id)],
            db=db,
            db_client=db_client,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.post("/create_details", response_model=CreateGraphDetailsResponse)
async def get_graph_create_details_endpoint(
    body: CreateGraphBody,
    user_id: ObjectId = Depends(get_user),
    db: AsyncIOMotorDatabase = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> CreateGraphDetailsResponse:
    """Get graph creation details."""
    details = await graph_service.get_graph_create_details(
        db=db, body=body, settings=settings, user_id=user_id
    )

    if details is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to retrieve graph creation details.",
        )
    chunks_selected, chunks_allowed, est_cost, est_time = details

    return CreateGraphDetailsResponse(
        message="Graph creation details processed successfully.",
        status="success",
        count=1,
        chunks_selected=chunks_selected,
        chunks_allowed=chunks_allowed,
        cost=est_cost,
        time=est_time,
    )


@router.put(
    "/{graph_id}",
    response_model=GraphsResponse,
    response_model_exclude_none=True,
)
async def update_graph_endpoint(
    body: GraphUpdate,
    graph: DetailedGraphDocumentModel = Depends(valid_graph_id),
    db: AsyncIOMotorDatabase = Depends(get_db),
    user_id: ObjectId = Depends(get_user),
) -> GraphsResponse:
    """Update graph."""
    updated_graph = await update_one(
        collection=db["graph"],
        document_model=GraphDocumentModel,
        id=ObjectId(graph.id),
        document=body,
        user_id=user_id,
    )
    return update_graph_response(GraphOut.model_validate(updated_graph))


@router.delete(
    "/{graph_id}",
    response_model=DetailedGraphsResponse,
    response_model_exclude_none=True,
)
async def delete_graph_endpoint(
    graph: DetailedGraphDocumentModel = Depends(valid_graph_id),
    db: AsyncIOMotorDatabase = Depends(get_db),
    db_client: AsyncIOMotorClient = Depends(get_db_client),
    user_id: ObjectId = Depends(get_user),
) -> DetailedGraphsResponse:
    """Delete graph and associated entities."""
    await delete_graphs(
        db=db,
        db_client=db_client,
        user_id=user_id,
        graph_ids=[ObjectId(graph.id)],
    )
    return DetailedGraphsResponse(
        message="Graph deleted successfully.",
        status="success",
        graphs=[
            DetailedGraphOut.model_validate(graph.model_dump(by_alias=True))
        ],
        count=1,
    )


@router.post(
    "/{graph_id}/merge_nodes",
    response_model=DetailedGraphsResponse,
    description="Merge nodes on a graph.",
)
async def merge_nodes_endpoint(
    request: MergeNodesRequest,
    graph: DetailedGraphDocumentModel = Depends(valid_graph_id),
    db: AsyncIOMotorDatabase = Depends(get_db),
    user_id: ObjectId = Depends(get_user),
) -> DetailedGraphsResponse:
    """Merge nodes on a graph."""
    from_nodes = request.from_nodes
    to_node = request.to_node
    if from_nodes is None or to_node is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Both from_nodes and to_node are required.",
        )
    if request.save_as_rule:
        from_nodes_ids = [ObjectId(n) for n in from_nodes]
        to_node_ids = [ObjectId(to_node)]
        from_node_names = [
            n.name
            for n in await get_nodes_by_ids(
                db, from_nodes_ids, ObjectId(graph.id), user_id
            )
        ]
        to_node_name = [
            n.name
            for n in await get_nodes_by_ids(
                db, to_node_ids, ObjectId(graph.id), user_id
            )
        ][0]
    try:
        merged_node = await graph_service.merge_nodes(
            db=db,
            graph_id=ObjectId(graph.id),
            user_id=user_id,
            from_nodes=[ObjectId(n) for n in from_nodes],
            to_node=ObjectId(to_node),
        )
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to merge nodes. Check that the nodes are existent and have the same type.",
        )
    if request.save_as_rule:
        rule = RuleCreate(
            workspace=graph.workspace.id,
            rule=MergeNodesRule(
                rule_type="merge_nodes",
                from_node_names=from_node_names,
                to_node_name=to_node_name,
                node_type=merged_node.label or Default_Entity_Type,
            ),
        )
        await create_rule(
            db=db,
            rule=rule,
            user_id=user_id,
        )
    return DetailedGraphsResponse(
        message="Nodes merged successfully.",
        status="success",
        graphs=[
            DetailedGraphOut.model_validate(graph.model_dump(by_alias=True))
        ],
        nodes=[merged_node],
        count=1,
    )


@router.post(
    "/{graph_id}/query",
    response_model=DetailedGraphsResponse,
    response_model_exclude_none=True,
    description="Query a graph using either a natural language query (unstructured) or set of specific entities and relations (structured).",
)
async def graph_query_endpoint(
    request: QueryGraphRequest,
    graph: DetailedGraphDocumentModel = Depends(valid_graph_id),
    llm_client: LLMClient = Depends(get_llm_client),
    db: AsyncIOMotorDatabase = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> DetailedGraphsResponse:
    """Query a graph."""
    try:
        query_processor = MixedQueryProcessor(
            db=db,
            graph_id=ObjectId(graph.id),
            user_id=ObjectId(graph.created_by),
            workspace_id=ObjectId(graph.workspace.id),
            schema_id=ObjectId(graph.schema_.id),
            llm_client=llm_client,
            settings=settings,
        )
        response = await query_processor.query(request=request)

        success_message = (
            "Graph query successful." if response else "No answer found."
        )
        query_results = [QueryOut.model_validate(response)] if response else []

        return DetailedGraphsResponse(
            message=success_message,
            status="success",
            count=1 if response else 0,
            graphs=[
                DetailedGraphOut.model_validate(
                    graph.model_dump(by_alias=True)
                )
            ],
            queries=query_results,
        )
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to perform graph query.",
        )


@router.get(
    "/{graph_id}/nodes",
    response_model=GraphsDetailedNodeResponse,
    description="Get all of the distinct nodes on a graph.",
)
async def get_graph_nodes_endpoint(
    graph: DetailedGraphDocumentModel = Depends(valid_graph_id),
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=-1, le=50),
    order: int = Depends(order_query),
    db: AsyncIOMotorDatabase = Depends(get_db),
) -> GraphsDetailedNodeResponse:
    """Get nodes on a graph."""
    collection = db["node"]
    nodes, total_count = await list_nodes(
        collection=collection,
        graph_id=ObjectId(graph.id),
        skip=skip,
        limit=limit,
        order=order,
        user_id=None,
    )
    if nodes is None or len(nodes) == 0:
        return GraphsDetailedNodeResponse(
            message="No nodes found.",
            status="success",
            graphs=[
                DetailedGraphOut.model_validate(
                    graph.model_dump(by_alias=True)
                )
            ],
            nodes=[],
            count=0,
        )

    return GraphsDetailedNodeResponse(
        message="Graph nodes retrieved successfully.",
        status="success",
        graphs=[
            DetailedGraphOut.model_validate(graph.model_dump(by_alias=True))
        ],
        nodes=nodes,
        count=total_count,
    )


@router.get(
    "/{graph_id}/resolve",
    response_model=GraphsSimilarNodesResponse,
    description="Get similar nodes on a graph.",
)
async def get_similar_nodes_endpoint(
    limit: int = Query(10, ge=-1, le=25),
    graph: DetailedGraphDocumentModel = Depends(valid_graph_id),
    db: AsyncIOMotorDatabase = Depends(get_db),
    user_id: ObjectId = Depends(get_user),
) -> GraphsSimilarNodesResponse:
    """Get similar nodes on a graph."""
    similar_nodes = await graph_service.get_similar_nodes(
        db=db,
        graph_id=ObjectId(graph.id),
        user_id=user_id,
        limit=limit,
    )

    return GraphsSimilarNodesResponse(
        message="Similar nodes retrieved successfully.",
        status="success",
        graphs=[
            DetailedGraphOut.model_validate(graph.model_dump(by_alias=True))
        ],
        similar_nodes=similar_nodes,
        count=len(similar_nodes) if similar_nodes else 0,
    )


@router.get(
    "/{graph_id}/relations",
    response_model=DetailedGraphsResponse,
    description="Get all of the distinct relations on a graph.",
    response_model_exclude_none=True,
)
async def get_graph_relations_endpoint(
    graph: DetailedGraphDocumentModel = Depends(valid_graph_id),
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=-1, le=50),
    order: str = Depends(order_query),
    db: AsyncIOMotorDatabase = Depends(get_db),
    user_id: ObjectId = Depends(get_user),
) -> DetailedGraphsResponse:
    """Get relations on a graph."""
    relations, total_count = await list_relations(collection=db["triple"], user_id=user_id, graph_id=graph.id, skip=skip, limit=limit, order=order)  # type: ignore[arg-type]
    workspace = await get_one(
        collection=db["workspace"],
        document_model=WorkspaceDocumentModel,
        user_id=user_id,
        id=ObjectId(graph.workspace.id),
    )
    if workspace is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workspace not found.",
        )
    return DetailedGraphsResponse(
        message="Graph relations retrieved successfully.",
        status="success",
        graphs=[
            DetailedGraphOut.model_validate(graph.model_dump(by_alias=True))
        ],
        relations=relations,
        count=total_count,
    )


@router.get(
    "/{graph_id}/triples",
    response_model=GraphsDetailedTripleResponse,
    description="Get all of the distint triples on a graph.",
)
async def get_graph_triples_endpoint(
    graph: DetailedGraphDocumentModel = Depends(valid_graph_id),
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=-1, le=50),
    order: str = Depends(order_query),
    db: AsyncIOMotorDatabase = Depends(get_db),
    user_id: ObjectId = Depends(get_user),
) -> GraphsDetailedTripleResponse:
    """Get graph triples."""
    triples, total_count = await list_triples(collection=db["triple"], graph_id=graph.id, skip=skip, limit=limit, order=order, user_id=user_id)  # type: ignore[arg-type]
    return GraphsDetailedTripleResponse(
        message="Graph triples retrieved successfully.",
        status="success",
        graphs=[
            DetailedGraphOut.model_validate(graph.model_dump(by_alias=True))
        ],
        triples=triples,
        count=total_count,
    )


@router.get(
    "/public/{graph_id}/nodes",
    response_model=PublicGraphsDetailedNodeResponse,
    description="Get all of the distinct nodes on a graph.",
    response_model_exclude_none=True,
)
async def get_public_graph_nodes_endpoint(
    graph: DetailedGraphDocumentModel = Depends(valid_public_graph_id),
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=-1, le=50),
    order: int = Depends(order_query),
    db: AsyncIOMotorDatabase = Depends(get_db),
) -> PublicGraphsDetailedNodeResponse:
    """Get nodes on a graph."""
    collection = db["node"]
    nodes, total_count = await list_nodes(
        collection=collection,
        graph_id=ObjectId(graph.id),
        skip=skip,
        limit=limit,
        order=order,
        user_id=None,
    )
    if nodes is None or len(nodes) == 0:
        return PublicGraphsDetailedNodeResponse(
            message="No nodes found.",
            status="success",
            graphs=[
                DetailedGraphOut.model_validate(
                    graph.model_dump(by_alias=True)
                )
            ],
            nodes=[],
            count=0,
        )
    return PublicGraphsDetailedNodeResponse(
        message="Graph nodes retrieved successfully.",
        status="success",
        graphs=[
            DetailedGraphOut.model_validate(graph.model_dump(by_alias=True))
        ],
        nodes=nodes,
        count=total_count,
    )


@router.get(
    "/public/{graph_id}/relations",
    response_model=PublicGraphsResponse,
    description="Get all of the distinct relations on a public graph.",
    response_model_exclude_none=True,
)
async def get_public_graph_relations(
    graph: DetailedGraphDocumentModel = Depends(valid_public_graph_id),
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=-1, le=50),
    order: str = Depends(order_query),
    db: AsyncIOMotorDatabase = Depends(get_db),
) -> PublicGraphsResponse:
    """Get relations on a graph."""
    relations, total_count = await list_relations(collection=db["triple"], user_id=None, graph_id=graph.id, skip=skip, limit=limit, order=order)  # type: ignore[arg-type]
    return PublicGraphsResponse(
        message="Graph relations retrieved successfully.",
        status="success",
        graphs=[
            DetailedGraphOut.model_validate(graph.model_dump(by_alias=True))
        ],
        relations=relations,
        count=total_count,
    )


@router.get(
    "/public/{graph_id}/triples",
    response_model=PublicGraphsTripleResponse,
)
async def get_public_graph_triples_endpoint(
    graph: DetailedGraphDocumentModel = Depends(valid_public_graph_id),
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=-1, le=50),
    order: str = Depends(order_query),
    db: AsyncIOMotorDatabase = Depends(get_db),
) -> PublicGraphsTripleResponse:
    """Get public graph triples."""
    triples, total_count = await list_triples(collection=db["triple"], graph_id=graph.id, skip=skip, limit=limit, order=order, user_id=None)  # type: ignore[arg-type]
    return PublicGraphsTripleResponse(
        message="Graph triples retrieved successfully.",
        status="success",
        graphs=[
            DetailedGraphOut.model_validate(graph.model_dump(by_alias=True))
        ],
        triples=triples,
        count=total_count,
    )


@router.get(
    "/{graph_id}/export/cypher",
    response_model=CypherResponse,
    description="Export graph as Cypher statements.",
)
async def export_graph_to_cypher_endpoint(
    graph: DetailedGraphDocumentModel = Depends(valid_graph_id),
    db: AsyncIOMotorDatabase = Depends(get_db),
    user_id: ObjectId = Depends(get_user),
) -> CypherResponse:
    """Export graph as Cypher statements."""
    try:
        cypher = await graph_service.export_graph_to_cypher(
            db=db,
            graph_id=ObjectId(graph.id),
            user_id=user_id,
        )
        if cypher is None:
            raise HTTPException(
                status_code=400,
                detail="No cypher to export for this graph.",
            )

        cypher_text = " ".join(cypher)

        return CypherResponse(
            message="Cypher text successfully generated.",
            status="success",
            count=1,
            cypher_text=cypher_text,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to export graph to Cypher: {str(e)}",
        )


@router.get(
    "/{graph_id}/chunks", response_model=ChunksResponseWithWorkspaceDetails
)
async def get_graph_chunks_endpoint(
    graph: DetailedGraphDocumentModel = Depends(valid_graph_id),
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=-1),
    order: int = Depends(order_query),
    db: AsyncIOMotorDatabase = Depends(get_db),
    user_id: ObjectId = Depends(get_user),
) -> ChunksResponseWithWorkspaceDetails:
    """Get graph chunks."""
    chunks, total_count = await get_graph_chunks(
        db=db,
        user_id=user_id,
        graph_id=ObjectId(graph.id),
        workspace_id=ObjectId(graph.workspace.id),
        skip=skip,
        limit=limit,
        order=order,
    )

    return ChunksResponseWithWorkspaceDetails(
        message="Chunks retrieved successfully.",
        status="success",
        count=total_count,
        chunks=chunks,
    )


@router.get(
    "/public/{graph_id}/chunks",
    response_model=PublicChunksResponseWithWorkspaceDetails,
)
async def get_public_graph_chunks_endpoint(
    graph: DetailedGraphDocumentModel = Depends(valid_public_graph_id),
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=-1),
    order: int = Depends(order_query),
    db: AsyncIOMotorDatabase = Depends(get_db),
) -> PublicChunksResponseWithWorkspaceDetails:
    """Get public graph chunks."""
    chunks, total_count = await get_graph_chunks(
        db=db,
        user_id=None,
        graph_id=ObjectId(graph.id),
        workspace_id=ObjectId(graph.workspace.id),
        skip=skip,
        limit=limit,
        order=order,
    )

    return PublicChunksResponseWithWorkspaceDetails(
        message="Chunks retrieved successfully.",
        status="success",
        count=total_count,
        chunks=chunks,
    )


@router.get(
    "/{graph_id}/rules",
    response_model_exclude_none=True,
    response_model=RulesResponse,
)
async def read_graph_rules_endpoint(
    graph: GraphDocumentModel = Depends(valid_graph_id),
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=-1, le=50),
    order: int = Depends(order_query),
    db: AsyncIOMotorDatabase = Depends(get_db),
    user_id: ObjectId = Depends(get_user),
) -> RulesResponse:
    """Get all graph rules."""
    rules, total_count = await get_graph_rules(
        db=db,
        graph_id=ObjectId(graph.id),
        user_id=user_id,
        skip=skip,
        limit=limit,
        order=order,
    )
    return RulesResponse(
        message="Rules retrieved successfully.",
        status="success",
        count=total_count,
        rules=[RuleOut.model_validate(rule) for rule in rules],
    )


@router.get(
    "/public/{graph_id}/rules",
    response_model_exclude_none=True,
    response_model=RulesResponse,
)
async def read_public_graph_rules_endpoint(
    graph: GraphDocumentModel = Depends(valid_public_graph_id),
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=-1, le=50),
    order: int = Depends(order_query),
    db: AsyncIOMotorDatabase = Depends(get_db),
) -> RulesResponse:
    """Get all public graph rules."""
    rules, total_count = await get_graph_rules(
        db=db,
        graph_id=ObjectId(graph.id),
        skip=skip,
        limit=limit,
        order=order,
    )
    return RulesResponse(
        message="Rules retrieved successfully.",
        status="success",
        count=total_count,
        rules=[RuleOut.model_validate(rule) for rule in rules],
    )
