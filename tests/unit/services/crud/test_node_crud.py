from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from bson import ObjectId

from whyhow_api.models.common import LLMClient
from whyhow_api.schemas.nodes import NodeDocumentModel
from whyhow_api.services.crud.node import (
    delete_node,
    get_nodes_by_ids,
    update_node,
)


@pytest.mark.asyncio
async def test_delete_node_success():
    fake_node_id = ObjectId()
    user_id = ObjectId()
    fake_node = {
        "_id": fake_node_id,
        "name": "test node",
        "graph": ObjectId(),
        "created_by": user_id,
        "document": ObjectId(),
        "workspaces": [ObjectId()],
    }
    triple_1 = {
        "_id": ObjectId(),
        "head_node": fake_node_id,
        "tail_node": ObjectId(),
    }
    triple_2 = {
        "_id": ObjectId(),
        "head_node": ObjectId(),
        "tail_node": fake_node_id,
    }

    db = MagicMock()
    db.node.find_one = AsyncMock(return_value=fake_node)
    db.triple.find.return_value.to_list = AsyncMock(
        side_effect=[
            [triple_1],  # First find for head_node
            [triple_2],  # Second find for tail_node
            [],  # Check orphaned node head_node
            [],  # Check orphaned node tail_node
            [],  # Check for triples with head_node as orphaned node
            [],  # Check for triples with tail_node as orphaned node
        ]
    )
    db.triple.delete_many = AsyncMock(return_value=None)
    db.node.delete_many = AsyncMock(return_value=None)
    db.node.delete_one = AsyncMock(return_value=None)

    session = MagicMock()
    session.start_transaction.return_value = AsyncMock()
    session.commit_transaction = AsyncMock()

    db_client = AsyncMock()
    db_client.start_session.return_value.__aenter__.return_value = session

    await delete_node(db, db_client, user_id, fake_node_id)

    db.node.delete_one.assert_awaited_once_with(
        {"_id": fake_node_id, "created_by": user_id}, session=session
    )
    db.triple.delete_many.assert_awaited_once_with(
        {
            "$or": [
                {
                    "$and": [
                        {"head_node": fake_node_id},
                        {"created_by": user_id},
                    ]
                },
                {
                    "$and": [
                        {"tail_node": fake_node_id},
                        {"created_by": user_id},
                    ]
                },
            ]
        },
        session=session,
    )


@pytest.mark.asyncio
async def test_get_nodes_by_ids():
    db_mock = MagicMock()
    node_ids_mock = [ObjectId(), ObjectId()]
    graph_id_mock = ObjectId()
    user_id_mock = ObjectId()

    # Mock the database response
    mock_nodes = [
        {
            "_id": node_ids_mock[0],
            "name": "node1",
            "type": "label1",
            "properties": {},
            "created_by": user_id_mock,
            "graph": graph_id_mock,
        },
        {
            "_id": node_ids_mock[1],
            "name": "node2",
            "type": "label2",
            "properties": {},
            "created_by": user_id_mock,
            "graph": graph_id_mock,
        },
    ]
    find_mock = MagicMock()
    find_mock.to_list = AsyncMock(return_value=mock_nodes)
    db_mock.node.find.return_value = find_mock

    result = await get_nodes_by_ids(
        db_mock, node_ids_mock, graph_id_mock, user_id_mock
    )

    assert len(result) == 2
    assert isinstance(result[0], NodeDocumentModel)
    assert result[0].id == node_ids_mock[0]
    assert result[0].name == "node1"
    assert result[0].type == "label1"
    assert result[0].properties == {}
    assert result[1].id == node_ids_mock[1]
    assert result[1].name == "node2"
    assert result[1].type == "label2"
    assert result[1].properties == {}

    db_mock.node.find.assert_called_once_with(
        {
            "_id": {"$in": node_ids_mock},
            "created_by": user_id_mock,
            "graph": graph_id_mock,
        }
    )


@pytest.mark.asyncio
@patch("whyhow_api.services.crud.node.update_one", new_callable=AsyncMock)
@patch(
    "whyhow_api.services.crud.node.update_triple_embeddings",
    new_callable=AsyncMock,
)
async def test_update_node_success(
    mock_update_triple_embeddings, mock_update_one
):
    fake_node_id = ObjectId()
    user_id = ObjectId()
    fake_node = {
        "_id": fake_node_id,
        "name": "test node",
        "graph": ObjectId(),
        "created_by": user_id,
        "document": ObjectId(),
        "workspaces": [ObjectId()],
    }
    updated_node_data = {
        "name": "updated node",
        "type": "updated type",
        "properties": {"key": "value"},
    }

    llm_client = MagicMock(spec=LLMClient)

    db = MagicMock()
    db.triple.find.return_value.to_list = AsyncMock(
        return_value=[{"_id": ObjectId()}]
    )
    update_one_return = MagicMock()
    fake_updated_node = fake_node.copy()
    fake_updated_node.update(updated_node_data)
    update_one_return.model_dump.return_value = fake_updated_node
    mock_update_one.return_value = update_one_return

    session = MagicMock()
    session.start_transaction.return_value = AsyncMock()
    session.commit_transaction = AsyncMock()

    db_client = AsyncMock()
    db_client.start_session.return_value.__aenter__.return_value = session

    node = NodeDocumentModel(**fake_node)
    update = MagicMock()
    update.name = updated_node_data["name"]
    update.type = updated_node_data["type"]
    update.properties = updated_node_data["properties"]
    update.graph = fake_node["graph"]

    result = await update_node(
        db, db_client, llm_client, user_id, fake_node_id, node, update
    )

    mock_update_one.assert_awaited_once_with(
        collection=db["node"],
        document_model=NodeDocumentModel,
        id=fake_node_id,
        document=update,
        user_id=user_id,
        session=session,
    )
    db.triple.find.assert_called_once()
    mock_update_triple_embeddings.assert_called_once()

    assert result.name == updated_node_data["name"]
    assert result.type == updated_node_data["type"]
    assert result.properties == updated_node_data["properties"]

    session.commit_transaction.assert_awaited_once()


@pytest.mark.asyncio
@patch("whyhow_api.services.crud.node.update_one", new_callable=AsyncMock)
@patch(
    "whyhow_api.services.crud.node.update_triple_embeddings",
    new_callable=AsyncMock,
)
async def test_update_node_not_found(
    mock_update_triple_embeddings, mock_update_one
):
    fake_node_id = ObjectId()
    user_id = ObjectId()

    fake_node = {
        "_id": fake_node_id,
        "name": "test node",
        "graph": ObjectId(),
        "created_by": user_id,
        "document": ObjectId(),
        "workspaces": [ObjectId()],
    }

    node = NodeDocumentModel(**fake_node)
    update = MagicMock()
    update.name = "updated node"
    update.type = "updated type"
    update.properties = {"key": "value"}
    update.graph = fake_node["graph"]

    db = MagicMock()
    mock_update_one.return_value = None
    db.triple.find.return_value.to_list = AsyncMock(return_value=[])

    session = MagicMock()
    session.start_transaction.return_value = AsyncMock()
    session.commit_transaction = AsyncMock()

    db_client = AsyncMock()
    db_client.start_session.return_value.__aenter__.return_value = session

    llm_client = MagicMock(spec=LLMClient)

    with pytest.raises(ValueError, match=f"Node {fake_node_id} not found."):
        await update_node(
            db, db_client, llm_client, user_id, fake_node_id, node, update
        )

    mock_update_one.assert_called_once()
    db.triple.find.assert_not_called()
    session.commit_transaction.assert_not_awaited()


@pytest.mark.asyncio
@patch("whyhow_api.services.crud.node.update_one", new_callable=AsyncMock)
@patch(
    "whyhow_api.services.crud.node.update_triple_embeddings",
    new_callable=AsyncMock,
)
async def test_update_node_with_triples(
    mock_update_triple_embeddings, mock_update_one
):
    fake_node_id = ObjectId()
    user_id = ObjectId()
    fake_node = {
        "_id": fake_node_id,
        "name": "test node",
        "graph": ObjectId(),
        "created_by": user_id,
        "document": ObjectId(),
        "workspaces": [ObjectId()],
    }
    updated_node_data = {
        "name": "updated node",
        "type": "updated type",
        "properties": {"key": "value"},
    }
    triple_1 = {
        "_id": ObjectId(),
        "head_node": fake_node_id,
        "tail_node": ObjectId(),
    }

    db = MagicMock()
    db.triple.find.return_value.to_list = AsyncMock(return_value=[triple_1])
    update_one_return = MagicMock()
    fake_updated_node = fake_node.copy()
    fake_updated_node.update(updated_node_data)
    update_one_return.model_dump.return_value = fake_updated_node
    mock_update_one.return_value = update_one_return

    session = MagicMock()
    session.start_transaction.return_value = AsyncMock()
    session.commit_transaction = AsyncMock()

    db_client = AsyncMock()
    db_client.start_session.return_value.__aenter__.return_value = session

    llm_client = MagicMock(spec=LLMClient)

    node = NodeDocumentModel(**fake_node)
    update = MagicMock()
    update.name = updated_node_data["name"]
    update.type = updated_node_data["type"]
    update.properties = updated_node_data["properties"]
    update.graph = fake_node["graph"]

    result = await update_node(
        db, db_client, llm_client, user_id, fake_node_id, node, update
    )

    mock_update_one.assert_awaited_once_with(
        collection=db["node"],
        document_model=NodeDocumentModel,
        id=fake_node_id,
        document=update,
        user_id=user_id,
        session=session,
    )
    db.triple.find.assert_called_once_with(
        {
            "$or": [
                {"head_node": fake_node_id, "created_by": user_id},
                {"tail_node": fake_node_id, "created_by": user_id},
            ]
        }
    )
    mock_update_triple_embeddings.assert_awaited_once_with(
        db=db,
        triple_ids=[ObjectId(triple_1["_id"])],
        llm_client=llm_client,
        user_id=user_id,
        session=session,
    )

    assert result.name == updated_node_data["name"]
    assert result.type == updated_node_data["type"]
    assert result.properties == updated_node_data["properties"]

    session.commit_transaction.assert_awaited_once()
