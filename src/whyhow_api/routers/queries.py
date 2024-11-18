"""Queries CRUD router."""

import logging
from typing import Annotated, Any, Dict

from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException, Query
from motor.motor_asyncio import AsyncIOMotorDatabase

from whyhow_api.dependencies import get_db, get_user, valid_query_id
from whyhow_api.schemas.base import Status
from whyhow_api.schemas.queries import (
    QueryDocumentModel,
    QueryOut,
    QueryResponse,
)
from whyhow_api.services.crud.base import delete_one, get_all, get_all_count
from whyhow_api.utilities.routers import order_query

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Queries"], prefix="/queries")


@router.get("", response_model=QueryResponse)
async def read_queries_endpoint(
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=-1, le=50),
    order: int = Depends(order_query),
    status: Annotated[
        Status | None, Query(description="The status of the query(-ies)")
    ] = None,
    graph_id: Annotated[
        str | None,
        Query(
            description="The id of the graph associated with the query(-ies)"
        ),
    ] = None,
    graph_name: Annotated[
        str | None,
        Query(
            description="The name of the graph associated with the query(-ies)"
        ),
    ] = None,
    db: AsyncIOMotorDatabase = Depends(get_db),
    user_id: ObjectId = Depends(get_user),
) -> QueryResponse:
    """Read queries."""
    if graph_id and graph_name:
        raise HTTPException(
            status_code=400,
            detail="Both graph_id and graph_name cannot be provided.",
        )

    collection = db["query"]
    pre_filters: Dict[str, Any] = {}
    post_filters: Dict[str, Any] = {}

    if graph_id:
        pre_filters["graph"] = ObjectId(graph_id)
    if status:
        pre_filters["status"] = status
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

    queries = await get_all(
        collection=collection,
        document_model=QueryDocumentModel,
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

    return QueryResponse(
        message="Queries retrieved successfully.",
        status="success",
        count=total_count,
        queries=[QueryOut.model_validate(q) for q in queries],
    )


@router.get("/{query_id}", response_model=QueryResponse)
async def read_query_endpoint(
    query: QueryDocumentModel = Depends(valid_query_id),
) -> QueryResponse:
    """Get query."""
    return QueryResponse(
        message="Query retrieved successfully.",
        status="success",
        count=1,
        queries=[QueryOut.model_validate(query)],
    )


@router.delete("/{query_id}", response_model=QueryResponse)
async def delete_query_endpoint(
    query: QueryDocumentModel = Depends(valid_query_id),
    db: AsyncIOMotorDatabase = Depends(get_db),
    user_id: ObjectId = Depends(get_user),
) -> QueryResponse:
    """Delete query."""
    deleted_query = await delete_one(
        collection=db["query"],
        document_model=QueryDocumentModel,
        id=ObjectId(query.id),
        user_id=user_id,
    )
    return QueryResponse(
        message="Query deleted successfully.",
        status="success",
        count=1,
        queries=[QueryOut.model_validate(deleted_query)],
    )
