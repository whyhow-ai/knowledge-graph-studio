from unittest.mock import AsyncMock

import pytest
from bson import ObjectId

from whyhow_api.dependencies import (
    get_db,
    get_db_client,
    get_user,
    valid_chunk_id,
    valid_workspace_id,
)
from whyhow_api.schemas.chunks import ChunkOut
from whyhow_api.schemas.workspaces import WorkspaceDocumentModel


@pytest.fixture
def fake_chunk_out():
    fake_chunk_id = ObjectId()
    user_id = ObjectId()

    return ChunkOut(
        id=fake_chunk_id,
        created_by=user_id,
        document=ObjectId(),
        workspaces=[ObjectId()],
        data_type="object",
        content="test content",
        tags={str(user_id): ["tag1"]},
        user_metadata={"test": {"test": "test"}},
        metadata={
            "language": "en",
            "size": 10,
            "data_source_type": "manual",
        },
    )


@pytest.fixture
def workspace_object_mock():
    return WorkspaceDocumentModel(
        _id=ObjectId(), name="workspace_object_mock", created_by=ObjectId()
    )


# ROUTERS UPDATE Chunks Unit Tests


def test_routers_update_chunk_not_found(
    client, monkeypatch, fake_chunk_out, workspace_object_mock
):
    chunk_id_mock = ObjectId()
    workspace_id_mock = ObjectId()

    fake_update = AsyncMock()
    fake_update.return_value = ["", None]
    monkeypatch.setattr("whyhow_api.routers.chunks.update_chunk", fake_update)

    client.app.dependency_overrides[get_db] = lambda: AsyncMock()
    client.app.dependency_overrides[get_user] = lambda: ObjectId()
    client.app.dependency_overrides[valid_chunk_id] = lambda: fake_chunk_out
    client.app.dependency_overrides[valid_workspace_id] = (
        lambda: workspace_object_mock
    )
    fake_object_to_update = {
        "metadata.language": "fr",
        "metadata.size": 15,
        "metadata.data_source_type": "manual",
    }

    response = client.put(
        f"/chunks/{chunk_id_mock}/{workspace_id_mock}",
        json=fake_object_to_update,
    )
    assert response.status_code == 404
    assert response.json() == {"detail": "Chunk not found."}


# ROUTERS DELETE Chunks Unit Tests
def test_routers_delete_chunk_successful(client, monkeypatch, fake_chunk_out):

    chunk_id_mock = ObjectId()

    fake_delete = AsyncMock()
    fake_delete.return_value = fake_chunk_out
    monkeypatch.setattr("whyhow_api.routers.chunks.delete_chunk", fake_delete)

    client.app.dependency_overrides[get_db] = lambda: AsyncMock()
    client.app.dependency_overrides[get_user] = lambda: ObjectId()
    client.app.dependency_overrides[get_db_client] = lambda: AsyncMock()
    client.app.dependency_overrides[valid_chunk_id] = lambda: fake_chunk_out

    response = client.delete(f"/chunks/{chunk_id_mock}")
    assert response.status_code == 200

    data = response.json()
    assert data["message"] == "Chunk deleted."
    assert data["status"] == "success"
    assert data["count"] == 1
    assert len(data["chunks"]) == 1
    chunk = data["chunks"][0]
    assert chunk["_id"] == fake_chunk_out.id
    assert chunk["created_by"] == str(fake_chunk_out.created_by)
    assert chunk["created_by"] == str(fake_chunk_out.created_by)
    assert chunk["workspaces"] == [
        str(workspace) for workspace in fake_chunk_out.workspaces
    ]
    assert chunk["document"] == str(fake_chunk_out.document)
    assert chunk["data_type"] == fake_chunk_out.data_type
    assert chunk["content"] == fake_chunk_out.content
    assert chunk["tags"] == fake_chunk_out.tags
    assert chunk["user_metadata"] == fake_chunk_out.user_metadata


def test_routers_delete_chunk_not_found(client, monkeypatch, fake_chunk_out):
    chunk_id_mock = ObjectId()

    fake_delete = AsyncMock()
    fake_delete.return_value = None
    monkeypatch.setattr("whyhow_api.routers.chunks.delete_chunk", fake_delete)

    client.app.dependency_overrides[get_db] = lambda: AsyncMock()
    client.app.dependency_overrides[get_user] = lambda: ObjectId()
    client.app.dependency_overrides[get_db_client] = lambda: AsyncMock()
    client.app.dependency_overrides[valid_chunk_id] = lambda: fake_chunk_out

    response = client.delete(f"/chunks/{chunk_id_mock}")
    assert response.status_code == 404
    assert response.json() == {"detail": "Chunk not found."}
