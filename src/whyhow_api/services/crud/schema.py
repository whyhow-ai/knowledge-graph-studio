"""Schema CRUD operations."""

import logging

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

logger = logging.getLogger(__name__)


async def delete_schema(
    db: AsyncIOMotorDatabase,
    db_client: AsyncIOMotorClient,
    user_id: ObjectId,
    schema_id: ObjectId,
) -> bool:
    """Delete a schema.

    Delete schema if it is not associated with any graph.
    """
    graphs = await db.graph.find(
        {"schema_id": schema_id, "created_by": user_id},
        {"_id": 1},
    ).to_list(None)
    graph_ids = [g["_id"] for g in graphs]

    if graph_ids:
        return False

    await db.schema.delete_one({"_id": schema_id, "created_by": user_id})
    return True
