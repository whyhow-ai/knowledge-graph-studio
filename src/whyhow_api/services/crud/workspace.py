"""Workspace CRUD operations."""

import logging

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from whyhow_api.services.crud.graph import delete_graphs

logger = logging.getLogger(__name__)


async def delete_workspace(
    db: AsyncIOMotorDatabase,
    db_client: AsyncIOMotorClient,
    user_id: ObjectId,
    workspace_id: ObjectId,
) -> None:
    """Delete a workspace.

    Delete workspace including its chunks, graphs, schemas, queries.
    """
    async with await db_client.start_session() as session:
        async with session.start_transaction():
            # Find graphs first to delete them
            graphs = await db.graph.find(
                {"workspace": workspace_id, "created_by": user_id},
                {"_id": 1},
                session=session,
            ).to_list(None)
            graph_ids = [g["_id"] for g in graphs]
            await delete_graphs(
                db=db,
                user_id=user_id,
                graph_ids=graph_ids,
                session=session,
            )

            # Deleting dependent entities
            await db.schema.delete_many(
                {"workspace": workspace_id, "created_by": user_id},
                session=session,
            )

            # Remove workspace from documents including tags and user_metadata
            await db.document.update_many(
                {"workspaces": workspace_id, "created_by": user_id},
                {
                    "$pull": {"workspaces": workspace_id},
                    "$unset": {  # Remove keys from tags and user_metadata
                        f"tags.{str(workspace_id)}": "",
                        f"user_metadata.{str(workspace_id)}": "",
                    },
                },
                session=session,
            )
            await db.chunk.update_many(
                {"workspaces": workspace_id, "created_by": user_id},
                {
                    "$pull": {"workspaces": workspace_id},
                    "$unset": {  # Remove keys from tags and user_metadata
                        f"tags.{str(workspace_id)}": "",
                        f"user_metadata.{str(workspace_id)}": "",
                    },
                },
                session=session,
            )

            # Finally, delete the workspace itself
            await db.workspace.delete_one(
                {"_id": workspace_id, "created_by": user_id}, session=session
            )

            logger.info(f"Workspace {workspace_id} was successfully deleted.")
