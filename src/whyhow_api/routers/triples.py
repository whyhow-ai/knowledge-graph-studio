"""Triple CRUD router."""

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

from whyhow_api.dependencies import (
    get_db,
    get_db_client,
    get_llm_client,
    get_user,
    valid_public_graph_id,
    valid_triple_id,
)
from whyhow_api.exceptions import NotFoundException
from whyhow_api.models.common import LLMClient
from whyhow_api.schemas.chunks import PublicChunksOutWithWorkspaceDetails
from whyhow_api.schemas.graphs import (
    GraphDocumentModel,
    GraphStateErrorsUpdate,
    Triple,
)
from whyhow_api.schemas.tasks import TaskOut, TaskResponse
from whyhow_api.schemas.triples import (
    PublicTripleChunksResponse,
    TripleChunksResponse,
    TripleCreateNode,
    TripleDocumentModel,
    TripleOut,
    TriplesCreate,
    TriplesResponse,
)
from whyhow_api.services import graph_service
from whyhow_api.services.crud.base import get_all, get_all_count, update_one
from whyhow_api.services.crud.triple import delete_triple, get_triple_chunks
from whyhow_api.utilities.routers import order_query

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Triples"], prefix="/triples")


def get_all_triples_response(
    triples: list[TripleOut], total_count: int
) -> TriplesResponse:
    """Get all triples response."""
    return TriplesResponse(
        message="Triples retrieved successfully.",
        status="success",
        triples=triples,
        count=total_count,
    )


def get_triple_response(triple: TripleOut) -> TriplesResponse:
    """Get triple response."""
    return TriplesResponse(
        message="Triple retrieved successfully.",
        status="success",
        count=1,
        triples=[triple],
    )


@router.get(
    "", response_model=TriplesResponse, response_model_exclude_none=True
)
async def read_triples_endpoint(
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=-1, le=50),
    order: int = Depends(order_query),
    type: Annotated[
        str | None, Query(description="The type of the triple(s)")
    ] = None,
    graph_id: Annotated[
        str | None,
        Query(description="The graph id associated with the triple(s)"),
    ] = None,
    graph_name: Annotated[
        str | None,
        Query(description="The graph name associated with the triple(s)"),
    ] = None,
    chunk_ids: Annotated[
        List[str] | None,
        Query(description="A set of chunk ids associated with the triple(s)"),
    ] = None,
    head_node_id: Annotated[
        str | None,
        Query(description="The head node id associated with the triple(s)"),
    ] = None,
    tail_node_id: Annotated[
        str | None,
        Query(description="The tail node id associated with the triple(s)"),
    ] = None,
    db: AsyncIOMotorDatabase = Depends(get_db),
    user_id: ObjectId = Depends(get_user),
) -> TriplesResponse:
    """Read triples."""
    if graph_id and graph_name:
        raise HTTPException(
            status_code=400,
            detail="Both graph_id and graph_name cannot be provided.",
        )

    collection = db["triple"]
    pre_filters: Dict[str, Any] = {}
    post_filters: Dict[str, Any] = {}

    if type:
        pre_filters["type"] = type
    if chunk_ids:
        pre_filters["chunks"] = {"$in": [ObjectId(c_id) for c_id in chunk_ids]}
    if head_node_id:
        pre_filters["head_node"] = ObjectId(head_node_id)
    if tail_node_id:
        pre_filters["tail_node"] = ObjectId(tail_node_id)
    if graph_id:
        pre_filters["graph"] = ObjectId(graph_id)
    if graph_name:
        post_filters["graph.name"] = graph_name

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
        {"$unwind": {"path": "$graph", "preserveNullAndEmptyArrays": False}},
        {"$match": post_filters},
        {
            "$addFields": {"graph": "$graph._id"}
        },  # TODO: Review whether we want to remove this
    ]

    triples = await get_all(
        collection=collection,
        document_model=TripleDocumentModel,
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

    return get_all_triples_response(
        triples=[TripleOut.model_validate(t) for t in triples],
        total_count=total_count,
    )


@router.get("/{triple_id}", response_model=TriplesResponse)
async def read_triple_endpoint(
    triple: TripleDocumentModel = Depends(valid_triple_id),
) -> TriplesResponse:
    """Read triple."""
    return get_triple_response(TripleOut.model_validate(triple))


@router.get(
    "/{triple_id}/chunks",
    response_model=TripleChunksResponse,
)
async def read_triple_with_chunks_endpoint(
    triple: TripleDocumentModel = Depends(valid_triple_id),
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=-1, le=50),
    order: int = Depends(order_query),
    db: AsyncIOMotorDatabase = Depends(get_db),
    user_id: ObjectId = Depends(get_user),
) -> TripleChunksResponse:
    """Read triple with chunks."""
    chunks, total_count = await get_triple_chunks(
        collection=db["triple"],
        id=ObjectId(triple.id),
        user_id=user_id,
        graph_id=None,
        skip=skip,
        limit=limit,
        order=order,
    )

    if chunks is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No chunks found.",
        )

    return TripleChunksResponse(
        message="Triple chunks retrieved successfully.",
        status="success",
        count=total_count,
        chunks=chunks,
    )


@router.get(
    "/public/{triple_id}/chunks",
    response_model=PublicTripleChunksResponse,
)
async def read_public_triple_with_chunks_endpoint(
    triple_id: str,
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=-1, le=50),
    order: int = Depends(order_query),
    db: AsyncIOMotorDatabase = Depends(get_db),
) -> PublicTripleChunksResponse:
    """Read public graph triple with chunks."""
    triple = await db.triple.find_one(
        {"_id": ObjectId(triple_id)}, {"embedding": 0}
    )
    if triple is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Triple not found.",
        )

    graph_id = triple.get("graph")
    await valid_public_graph_id(db=db, graph_id=graph_id)
    chunks, total_count = await get_triple_chunks(
        collection=db["triple"],
        id=ObjectId(triple_id),
        user_id=None,
        graph_id=ObjectId(graph_id),
        skip=skip,
        limit=limit,
        order=order,
    )

    if chunks is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No chunks found.",
        )

    return PublicTripleChunksResponse(
        message="Triple chunks retrieved successfully.",
        status="success",
        count=total_count,
        chunks=[
            PublicChunksOutWithWorkspaceDetails.model_validate(c)
            for c in chunks
        ],
    )


@router.post("", response_model=TaskResponse)
async def create_triples_endpoint(
    background_tasks: BackgroundTasks,
    body: TriplesCreate,
    db: AsyncIOMotorDatabase = Depends(get_db),
    db_client: AsyncIOMotorClient = Depends(get_db_client),
    user_id: ObjectId = Depends(get_user),
    llm_client: LLMClient = Depends(get_llm_client),
) -> TaskResponse:
    """Create triples."""
    # Check if the graph exists
    db_graph = await db.graph.find_one(
        {"_id": ObjectId(body.graph), "created_by": user_id}
    )
    if db_graph is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Graph not found.",
        )

    # Prepare triples
    triples: list[Triple] = []
    for triple in body.triples:
        if type(triple.head_node) is str:
            node = await db.node.find_one(
                {
                    "_id": ObjectId(triple.head_node),
                    "created_by": user_id,
                    "graph": ObjectId(body.graph),
                }
            )
            if node is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Node not found.",
                )
            head_node = TripleCreateNode(
                name=node["name"],
                type=node["type"],
                properties=node["properties"],
            )
        else:
            head_node = triple.head_node  # type: ignore

        if type(triple.tail_node) is str:
            node = await db.node.find_one(
                {
                    "_id": ObjectId(triple.tail_node),
                    "created_by": user_id,
                    "graph": ObjectId(body.graph),
                }
            )
            if node is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Node not found.",
                )
            tail_node = TripleCreateNode(
                name=node["name"],
                type=node["type"],
                properties=node["properties"],
            )
        else:
            tail_node = triple.tail_node  # type: ignore

        relation_properties = triple.properties
        relation_properties["chunks"] = triple.chunks
        triples.append(
            Triple(
                head=head_node.name,
                head_type=head_node.type,
                head_properties=head_node.properties,
                tail=tail_node.name,
                tail_type=tail_node.type,
                tail_properties=tail_node.properties,
                relation=triple.type,
                relation_properties=relation_properties,
            )
        )

    try:
        await update_one(
            collection=db["graph"],
            document_model=GraphDocumentModel,
            id=ObjectId(body.graph),
            document=GraphStateErrorsUpdate(status="updating"),
            user_id=user_id,
        )
        task_doc = await graph_service.create_or_update_graph_from_triples(
            background_tasks=background_tasks,
            triples=triples,
            db=db,
            db_client=db_client,
            user_id=user_id,
            llm_client=llm_client,
            graph_id=ObjectId(body.graph),
            strict_mode=body.strict_mode,
        )
        task = TaskOut.model_validate(task_doc)
        task.id = str(task.id)
        task.created_by = str(task.created_by)
        return TaskResponse(
            message="Triples creation task started successfully.",
            status="success",
            task=task,
            count=1,
        )
    except NotFoundException as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.delete(
    "/{triple_id}",
    response_model=TriplesResponse,
)
async def delete_triple_endpoint(
    triple: TripleDocumentModel = Depends(valid_triple_id),
    db: AsyncIOMotorDatabase = Depends(get_db),
    user_id: ObjectId = Depends(get_user),
) -> TriplesResponse:
    """Delete a triple."""
    logger.info(f"Deleting triple: {triple.id}")
    await delete_triple(
        db=db,
        user_id=user_id,
        triple_id=ObjectId(triple.id),
    )
    return TriplesResponse(
        message="Triple deleted successfully.",
        status="success",
        triples=[TripleOut.model_validate(triple)],
        count=1,
    )
