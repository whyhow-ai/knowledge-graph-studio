from unittest.mock import AsyncMock, MagicMock

import pytest
from bson import ObjectId

from whyhow_api.services.crud.workspace import delete_workspace


@pytest.mark.asyncio
async def test_delete_workspace_success():
    workspace_id = ObjectId()
    user_id = ObjectId()

    graphs = [{"_id": ObjectId()}, {"_id": ObjectId()}]
    graph_ids = [g["_id"] for g in graphs]

    db = MagicMock()
    db.graph.find.return_value.to_list = AsyncMock(return_value=graphs)
    db.node.delete_many = AsyncMock(return_value=None)
    db.triple.delete_many = AsyncMock(return_value=None)
    db.query.delete_many = AsyncMock(return_value=None)
    db.graph.delete_many = AsyncMock(return_value=None)
    db.schema.delete_many = AsyncMock(return_value=None)
    db.document.update_many = AsyncMock(return_value=None)
    db.chunk.update_many = AsyncMock(return_value=None)
    db.workspace.delete_one = AsyncMock(return_value=None)

    session = MagicMock()
    session.start_transaction.return_value = AsyncMock()
    session.commit_transaction = AsyncMock()

    db_client = AsyncMock()
    db_client.start_session.return_value.__aenter__.return_value = session

    await delete_workspace(db, db_client, user_id, workspace_id)

    db.node.delete_many.assert_awaited_once_with(
        {"graph": {"$in": graph_ids}, "created_by": user_id}, session=session
    )
    db.triple.delete_many.assert_awaited_once_with(
        {"graph": {"$in": graph_ids}, "created_by": user_id}, session=session
    )
    db.query.delete_many.assert_awaited_once_with(
        {"graph": {"$in": graph_ids}, "created_by": user_id}, session=session
    )
    db.graph.delete_many.assert_awaited_once_with(
        {"_id": {"$in": graph_ids}, "created_by": user_id}, session=session
    )
    db.schema.delete_many.assert_awaited_once_with(
        {"workspace": workspace_id, "created_by": user_id}, session=session
    )
    db.document.update_many.assert_awaited_once_with(
        {"workspaces": workspace_id, "created_by": user_id},
        {
            "$pull": {"workspaces": workspace_id},
            "$unset": {
                f"tags.{str(workspace_id)}": "",
                f"user_metadata.{str(workspace_id)}": "",
            },
        },
        session=session,
    )
    db.chunk.update_many.assert_awaited_once_with(
        {"workspaces": workspace_id, "created_by": user_id},
        {
            "$pull": {"workspaces": workspace_id},
            "$unset": {
                f"tags.{str(workspace_id)}": "",
                f"user_metadata.{str(workspace_id)}": "",
            },
        },
        session=session,
    )
    db.workspace.delete_one.assert_awaited_once_with(
        {"_id": workspace_id, "created_by": user_id}, session=session
    )
