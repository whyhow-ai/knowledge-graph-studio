from unittest.mock import AsyncMock

import pytest
from bson import ObjectId

from whyhow_api.dependencies import get_db, get_user, valid_workspace_id
from whyhow_api.schemas.base import get_utc_now
from whyhow_api.schemas.chunks import (
    AddChunkModel,
    ChunkDocumentModel,
    ChunkMetadata,
)
from whyhow_api.schemas.workspaces import WorkspaceDocumentModel


class TestWorkspacesGetOne:

    def test_get_workspace_successful(self, client):

        workspace_id_mock = ObjectId()
        user_id_mock = ObjectId()
        workspace_object_mock = WorkspaceDocumentModel(
            name="test workspace", created_by=user_id_mock
        )

        client.app.dependency_overrides[get_db] = lambda: AsyncMock()
        client.app.dependency_overrides[get_user] = lambda: ObjectId()
        client.app.dependency_overrides[valid_workspace_id] = (
            lambda: workspace_object_mock
        )

        response = client.get(f"/workspaces/{workspace_id_mock}")
        assert response.status_code == 200

        data = response.json()
        assert data["message"] == "Workspace retrieved successfully."
        assert data["status"] == "success"
        assert data["count"] == 1
        assert data["workspaces"][0]["name"] == "test workspace"
        assert data["workspaces"][0]["created_by"] == str(
            workspace_object_mock.created_by
        )

    def test_get_workspace_failure(self, client, monkeypatch):

        workspace_id_mock = ObjectId()
        monkeypatch.setattr("whyhow_api.dependencies.get_one", None)

        client.app.dependency_overrides[get_db] = lambda: AsyncMock()
        client.app.dependency_overrides[get_user] = lambda: ObjectId()

        response = client.get(f"/workspaces/{workspace_id_mock}")
        assert response.status_code == 404

        data = response.json()
        assert data["detail"] == "Workspace not found"


class TestWorkspacesGetAll:

    def test_get_workspaces_successful(self, client, monkeypatch):

        workspace_object_mock = WorkspaceDocumentModel(
            name="test workspace", created_by=ObjectId()
        )

        fake_get_all = AsyncMock()
        fake_get_all.return_value = [workspace_object_mock.model_dump()]
        monkeypatch.setattr(
            "whyhow_api.routers.workspaces.get_all", fake_get_all
        )

        fake_get_all_count = AsyncMock()
        fake_get_all_count.return_value = 1
        monkeypatch.setattr(
            "whyhow_api.routers.workspaces.get_all_count", fake_get_all_count
        )

        client.app.dependency_overrides[get_db] = lambda: AsyncMock()
        client.app.dependency_overrides[get_user] = lambda: ObjectId()

        response = client.get("/workspaces", params={"skip": 0, "limit": 1})
        assert response.status_code == 200

        data = response.json()
        assert data["message"] == "Workspaces retrieved successfully."
        assert data["status"] == "success"
        assert data["count"] == 1
        assert data["workspaces"][0]["name"] == "test workspace"
        assert data["workspaces"][0]["created_by"] == str(
            workspace_object_mock.created_by
        )


class TestWorkspacesCreate:

    def test_create_workspace_successful(self, client, monkeypatch):

        user_id_mock = ObjectId()
        workspace_object_mock = WorkspaceDocumentModel(
            name="test workspace",
            created_by=user_id_mock,
        )

        fake_create = AsyncMock()
        fake_create.return_value = workspace_object_mock.model_dump()
        monkeypatch.setattr(
            "whyhow_api.routers.workspaces.create_one", fake_create
        )

        client.app.dependency_overrides[get_db] = lambda: AsyncMock()
        client.app.dependency_overrides[get_user] = lambda: user_id_mock

        response = client.post(
            "/workspaces",
            json={
                "name": "test workspace",
            },
        )
        assert response.status_code == 200

        data = response.json()
        assert data["message"] == "Workspace created successfully."
        assert data["status"] == "success"
        assert data["count"] == 1
        assert data["workspaces"][0]["name"] == "test workspace"
        assert data["workspaces"][0]["created_by"] == str(user_id_mock)


class TestWorkspacesUpdate:

    def test_update_workspace_successful(self, client, monkeypatch):

        workspace_id_mock = ObjectId()
        workspace_object_mock = WorkspaceDocumentModel(
            name="test workspace updated", created_by=ObjectId()
        )

        fake_update = AsyncMock()
        fake_update.return_value = workspace_object_mock.model_dump()
        monkeypatch.setattr(
            "whyhow_api.routers.workspaces.update_one", fake_update
        )

        client.app.dependency_overrides[get_db] = lambda: AsyncMock()
        client.app.dependency_overrides[get_user] = lambda: ObjectId()

        response = client.put(
            f"/workspaces/{workspace_id_mock}",
            json={"name": "test workspace updated"},
        )
        assert response.status_code == 200

        data = response.json()
        assert data["message"] == "Workspace updated successfully."
        assert data["status"] == "success"
        assert data["count"] == 1
        assert data["workspaces"][0]["name"] == "test workspace updated"
        assert data["workspaces"][0]["created_by"] == str(
            workspace_object_mock.created_by
        )

    def test_update_workspace_not_found(self, client, monkeypatch):

        fake_update = AsyncMock()
        fake_update.return_value = None
        monkeypatch.setattr(
            "whyhow_api.routers.workspaces.update_one", fake_update
        )

        client.app.dependency_overrides[get_db] = lambda: AsyncMock()
        client.app.dependency_overrides[get_user] = lambda: ObjectId()

        response = client.put(
            f"/workspaces/{ObjectId()}",
            json={"name": "test workspace updated"},
        )
        assert response.status_code == 404

        data = response.json()
        assert data["detail"] == "Workspace not found"


@pytest.mark.skip(reason="Not implemented yet.")
class TestWorkspacesChunkUpdate:

    def test_update_workspace_chunk_successful(self, client, monkeypatch):
        pass

    def test_update_workspace_chunk_not_found(self, client, monkeypatch):
        pass


class TestWorkspacesAddChunks:

    @pytest.fixture
    def add_chunk_mock(self):
        return AddChunkModel(
            content="test chunk",
            user_metadata={"test": "test"},
            tags=["test"],
        )

    @pytest.fixture
    def chunk_document_mock(self):
        return ChunkDocumentModel(
            _id=ObjectId(),
            created_at=get_utc_now(),
            updated_at=get_utc_now(),
            workspace=ObjectId(),
            document=None,
            content="test chunk",
            user_metadata={"test": "test"},
            data_type="string",
            tags=["test"],
            metadata=ChunkMetadata(language="en"),
        )


class TestWorkspacesRemoveChunks:

    pass


class TestWorkspacesListChunks:

    pass


class TestWorkspacesListDocuments:

    pass


class TestWorkspacesAddDocuments:

    pass


class TestWorkspacesRemoveDocuments:

    pass


class TestWorkspaceDelete:

    pass


# class TestWorkspacesGetAllDocuments:

#     @pytest.fixture
#     def document_object_mock(self):
#         return DocumentDocumentModel(
#             _id=ObjectId(),
#             filename="test document",
#             workspaces=[ObjectId()],
#         )

#     def test_get_documents_successful(
#         self, client, monkeypatch, document_object_mock
#     ):

#         fake_get_all = AsyncMock()
#         fake_get_all.return_value = [document_object_mock.model_dump()]
#         monkeypatch.setattr(
#             "whyhow_api.routers.documents.get_all", fake_get_all
#         )

#         fake_get_all_count = AsyncMock()
#         fake_get_all_count.return_value = 1
#         monkeypatch.setattr(
#             "whyhow_api.routers.documents.get_all_count", fake_get_all_count
#         )

#         client.app.dependency_overrides[get_db] = lambda: AsyncMock()
#         client.app.dependency_overrides[get_user] = lambda: ObjectId()

#         response = client.get("/documents", params={"skip": 0, "limit": 1})
#         assert response.status_code == 200

#         data = response.json()
#         assert data["message"] != ""
#         assert data["status"] == "success"
#         assert data["count"] == 1
#         assert data["documents"][0]["name"] == "test document"

#     def test_get_documents_exception(self, client, monkeypatch):

#         fake_get_all = AsyncMock()
#         fake_get_all.side_effect = Exception("Test exception")
#         monkeypatch.setattr(
#             "whyhow_api.routers.documents.get_all", fake_get_all
#         )

#         client.app.dependency_overrides[get_db] = lambda: AsyncMock()
#         client.app.dependency_overrides[get_user] = lambda: ObjectId()

#         response = client.get("/documents")
#         assert response.status_code == 500

#         data = response.json()
#         assert "detail" in data
#         assert data["detail"] != ""
