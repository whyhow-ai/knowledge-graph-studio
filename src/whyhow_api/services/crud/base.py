"""Base CRUD operations."""

import logging
from typing import Any, Dict, List, Type

from bson import ObjectId
from motor.core import AgnosticClientSession
from motor.motor_asyncio import AsyncIOMotorCollection
from pydantic import BaseModel

from whyhow_api.utilities.routers import list_aggregation

logger = logging.getLogger(__name__)


async def get_one(
    collection: AsyncIOMotorCollection,
    document_model: Type[BaseModel],
    user_id: ObjectId,
    id: ObjectId | None = None,
    filters: Dict[str, Any] | None = None,
    remove_fields: List[str] | None = None,
) -> BaseModel | None:
    """Get a single object."""
    #  User collection does not have a created_by field
    if collection.name == "user":
        query = {}
    else:
        query = {"created_by": user_id}
    if id:
        query["_id"] = id
    if filters:
        query.update(filters)

    obj = await collection.find_one(
        query, {field: 0 for field in remove_fields} if remove_fields else None
    )
    if obj:
        return document_model(**obj)
    return None


async def get_all(
    collection: AsyncIOMotorCollection,
    document_model: Type[BaseModel],
    user_id: ObjectId,
    aggregation_query: List[Dict[str, Any]] = [],
    skip: int = 0,
    limit: int = 10,
    order: int = -1,
) -> List[BaseModel]:
    """Get all objects."""
    pipeline = list_aggregation(
        user_id=user_id,
        aggregation_query=aggregation_query,
        skip=skip,
        limit=limit,
        order=order,
    )

    items = await collection.aggregate(pipeline).to_list(
        length=None if limit == -1 else limit
    )
    return [document_model(**i) for i in items]


async def get_all_count(
    collection: AsyncIOMotorCollection,
    user_id: ObjectId | None,
    aggregation_query: List[Dict[str, Any]] = [],
) -> int:
    """Get count of all objects."""
    pipeline = list_aggregation(
        user_id=user_id, aggregation_query=aggregation_query, count=True
    )

    result = await collection.aggregate(pipeline).to_list(None)
    return result[0]["total"] if result else 0


async def create_one(
    collection: AsyncIOMotorCollection,
    document_model: Type[BaseModel],
    document: BaseModel,
    user_id: ObjectId,
    session: AgnosticClientSession | None = None,
) -> BaseModel:
    """Create a new object."""
    document_dict = document.model_dump(by_alias=True, exclude_none=True)
    if "created_by" not in document_dict:
        document_dict["created_by"] = user_id
    document_data = document_model(**document_dict).model_dump(
        by_alias=True, exclude_none=True
    )
    result = await collection.insert_one(document_data, session=session)
    document_data["_id"] = result.inserted_id
    return document_model(**document_data)


async def update_one(
    collection: AsyncIOMotorCollection,
    document_model: Type[BaseModel],
    id: ObjectId,
    document: BaseModel,
    user_id: ObjectId,
    session: AgnosticClientSession | None = None,
) -> BaseModel | None:
    """Update an object."""
    update_data = {
        k: v for k, v in document.model_dump().items() if v is not None
    }
    await collection.update_one(
        {"_id": id}, {"$set": update_data}, session=session
    )
    #  User collection does not have a created_by field
    if collection.name == "user":
        updated_obj = await collection.find_one({"_id": id}, session=session)
    else:
        updated_obj = await collection.find_one(
            {"_id": id, "created_by": user_id}, session=session
        )
    if updated_obj:
        return document_model(**updated_obj)
    else:
        return None


async def delete_one(
    collection: AsyncIOMotorCollection,
    document_model: Type[BaseModel],
    id: ObjectId,
    user_id: ObjectId,
) -> BaseModel | None:
    """Delete a single object.

    Returns the deleted object if successful.
    """
    obj = await collection.find_one({"_id": id, "created_by": user_id})
    if obj:
        await collection.delete_one({"_id": id, "created_by": user_id})
        return document_model(**obj)
    else:
        return None


async def delete_all() -> None:
    """Delete all objects."""
    raise NotImplementedError(
        "Delete all is not implemented yet."
    )  # noqa: E501
