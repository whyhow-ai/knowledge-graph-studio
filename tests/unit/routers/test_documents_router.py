from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
from botocore.exceptions import ClientError
from bson import ObjectId

from whyhow_api.dependencies import (
    get_db,
    get_db_client,
    get_llm_client,
    get_settings,
    get_user,
    valid_document_id,
    valid_workspace_id,
)
from whyhow_api.routers.graphs import order_query
from whyhow_api.schemas.documents import (
    DocumentMetadata,
    DocumentOutWithWorkspaceDetails,
    DocumentUpdate,
)
from whyhow_api.schemas.workspaces import (
    WorkspaceDetails,
    WorkspaceDocumentModel,
)


class TestDocumentsGetOne:

    @pytest.fixture
    def document_object_mock(self):
        workspace_id = ObjectId()
        return DocumentOutWithWorkspaceDetails(
            _id=ObjectId(),
            created_by=ObjectId(),
            workspaces=[
                WorkspaceDetails(_id=workspace_id, name="test_workspace")
            ],
            status="processed",
            metadata=DocumentMetadata(
                size=1234, format="txt", filename="test_file.txt"
            ),
            tags={str(workspace_id): ["tag1"]},
            user_metadata={str(workspace_id): {"hello": "world"}},
        )

    def test_get_document_successful(self, client, document_object_mock):
        document_id_mock = ObjectId()

        client.app.dependency_overrides[get_db] = lambda: AsyncMock()
        client.app.dependency_overrides[get_user] = lambda: ObjectId()
        client.app.dependency_overrides[valid_document_id] = (
            lambda: document_object_mock
        )

        response = client.get(f"/documents/{document_id_mock}")
        assert response.status_code == 200

        data = response.json()
        assert data["message"] != ""
        assert data["status"] == "success"
        assert data["count"] == 1
        assert data["documents"][0]["metadata"]["filename"] == "test_file.txt"
        assert isinstance(data["documents"][0]["workspaces"][0]["name"], str)
        assert isinstance(data["documents"][0]["workspaces"][0]["_id"], str)

    def test_get_document_not_found(self, client, monkeypatch):

        document_id_mock = ObjectId()

        monkeypatch.setattr("whyhow_api.dependencies.get_one", None)

        client.app.dependency_overrides[get_db] = lambda: AsyncMock()
        client.app.dependency_overrides[get_user] = lambda: ObjectId()

        response = client.get(f"/documents/{document_id_mock}")
        assert response.status_code == 404

        data = response.json()
        assert "detail" in data
        assert data["detail"] != ""


class TestDocumentsGetAll:
    def test_get_documents_with_both_workspace_id_and_workspace_name_error(
        self, client
    ):
        fake_workspace_id = ObjectId()

        client.app.dependency_overrides[order_query] = lambda: 1
        client.app.dependency_overrides[get_db] = lambda: AsyncMock()
        client.app.dependency_overrides[get_user] = lambda: ObjectId()

        params = {
            "workspace_id": fake_workspace_id,
            "workspace_name": "test_name",
        }

        response = client.get("/documents", params=params)
        assert response.status_code == 400

        data = response.json()
        assert "detail" in data
        assert (
            data["detail"]
            == "Both workspace_id and workspace_name cannot be provided."
        )

    def test_get_documents_not_found_count_zero(self, client, monkeypatch):
        fake_workspace_id = ObjectId()

        client.app.dependency_overrides[order_query] = lambda: 1
        client.app.dependency_overrides[get_db] = lambda: AsyncMock()
        client.app.dependency_overrides[get_user] = lambda: ObjectId()

        params = {
            "workspace_id": fake_workspace_id,
            "filename": "test document",
            "status": "processed",
        }

        fake_get_all_count = AsyncMock()
        fake_get_all_count.return_value = 0
        monkeypatch.setattr(
            "whyhow_api.routers.documents.get_all_count", fake_get_all_count
        )

        response = client.get("/documents", params=params)

        data = response.json()
        print(data)
        assert len(data["documents"]) == 0
        assert data["count"] == 0
        assert data["message"] == "No documents found."

    def test_get_documents_not_found(self, client, monkeypatch):
        fake_workspace_id = ObjectId()

        client.app.dependency_overrides[order_query] = lambda: 1
        client.app.dependency_overrides[get_db] = lambda: AsyncMock()
        client.app.dependency_overrides[get_user] = lambda: ObjectId()

        params = {
            "workspace_id": fake_workspace_id,
            "filename": "test document",
            "status": "processed",
        }

        fake_get_all_count = AsyncMock()
        fake_get_all_count.return_value = 1
        monkeypatch.setattr(
            "whyhow_api.routers.documents.get_all_count", fake_get_all_count
        )

        fake_get_all = AsyncMock()
        fake_get_all.return_value = None
        monkeypatch.setattr(
            "whyhow_api.routers.documents.get_all", fake_get_all
        )

        response = client.get("/documents", params=params)

        data = response.json()
        assert response.status_code == 404
        assert "detail" in data
        assert data["detail"] == "No documents found."

    @pytest.fixture
    def document_object_mock(self):
        workspace_id = ObjectId()
        return DocumentOutWithWorkspaceDetails(
            _id=ObjectId(),
            created_by=ObjectId(),
            status="processed",
            metadata=DocumentMetadata(
                size=1234, format="txt", filename="test_file.txt"
            ),
            tags={str(workspace_id): ["tag1"]},
            user_metadata={str(workspace_id): {"hello": "world"}},
            workspaces=[
                WorkspaceDetails(_id=workspace_id, name="workspace_name")
            ],
        )

    def test_get_documents_successful(
        self, client, monkeypatch, document_object_mock
    ):

        client.app.dependency_overrides[order_query] = lambda: 1
        client.app.dependency_overrides[get_db] = lambda: AsyncMock()
        client.app.dependency_overrides[get_user] = lambda: ObjectId()

        params = {
            "workspace_name": "test_workspace",
            "filename": "test document",
            "status": "processed",
        }

        fake_get_all_count = AsyncMock()
        fake_get_all_count.return_value = 1
        monkeypatch.setattr(
            "whyhow_api.routers.documents.get_all_count", fake_get_all_count
        )

        fake_get_all = AsyncMock()
        fake_get_all.return_value = [
            document_object_mock.model_dump(by_alias=True)
        ]
        monkeypatch.setattr(
            "whyhow_api.routers.documents.get_all", fake_get_all
        )

        response = client.get("/documents", params=params)

        data = response.json()
        print(data)
        assert response.status_code == 200


class TestDocumentsUpdateInWorkspace:

    @pytest.fixture
    def document_object_mock(self):
        workspace_id = ObjectId()
        return DocumentOutWithWorkspaceDetails(
            _id=ObjectId(),
            created_by=ObjectId(),
            workspaces=[
                WorkspaceDetails(_id=workspace_id, name="test_workspace")
            ],
            status="processed",
            metadata=DocumentMetadata(
                size=1234, format="txt", filename="test_file.txt"
            ),
            tags={str(workspace_id): ["tag1"]},
            user_metadata={str(workspace_id): {"hello": "world"}},
        )

    @pytest.fixture
    def workspace_object_mock(self):
        return WorkspaceDocumentModel(
            _id=ObjectId(), name="test_workspace", created_by=ObjectId()
        )

    def test_update_document_in_workspace_successful(
        self, client, monkeypatch, document_object_mock, workspace_object_mock
    ):

        document_id_mock = ObjectId()

        fake_update = AsyncMock()
        fake_update.return_value = document_object_mock.model_dump()
        monkeypatch.setattr(
            "whyhow_api.routers.documents.update_document", fake_update
        )

        client.app.dependency_overrides[get_db] = lambda: AsyncMock()
        client.app.dependency_overrides[get_user] = lambda: ObjectId()
        client.app.dependency_overrides[valid_workspace_id] = (
            lambda: workspace_object_mock
        )
        client.app.dependency_overrides[valid_document_id] = (
            lambda: document_object_mock
        )

        update_model = DocumentUpdate(
            user_metadata={"hello": "world"}, tags=["tag1"]
        )
        update_model_dict = update_model.model_dump()

        response = client.put(
            f"/documents/{document_id_mock}/{document_object_mock.workspaces[0]}",
            json=update_model_dict,
        )
        assert response.status_code == 200

        data = response.json()
        assert data["message"] == "Document updated successfully."
        assert data["status"] == "success"
        assert data["count"] == 1
        assert data["documents"][0]["tags"] == document_object_mock.tags
        assert len(data["documents"][0]["tags"].values()) == 1
        assert (
            data["documents"][0]["user_metadata"]
            == document_object_mock.user_metadata
        )

    def test_update_document_in_workspace_not_found(self, client, monkeypatch):
        document_id_mock = ObjectId()
        workspace_id_mock = ObjectId()

        monkeypatch.setattr("whyhow_api.dependencies.get_one", None)

        client.app.dependency_overrides[get_db] = lambda: AsyncMock()
        client.app.dependency_overrides[get_user] = lambda: ObjectId()

        update_model = DocumentUpdate(
            user_metadata={"hello": "world"}, tags=["tag1"]
        )
        update_model_dict = update_model.model_dump()

        response = client.put(
            f"/documents/{document_id_mock}/{workspace_id_mock}",
            json=update_model_dict,
        )
        assert response.status_code == 404

        data = response.json()
        assert "detail" in data
        assert data["detail"] == "Document not found"


class TestDocumentsDelete:
    @pytest.fixture
    def document_object_mock(self):
        workspace_id = ObjectId()
        return DocumentOutWithWorkspaceDetails(
            _id=ObjectId(),
            created_by=ObjectId(),
            workspaces=[
                WorkspaceDetails(_id=workspace_id, name="test_workspace")
            ],
            status="processed",
            metadata=DocumentMetadata(
                size=1234, format="txt", filename="test_file.txt"
            ),
            tags={str(workspace_id): ["tag1"]},
            user_metadata={str(workspace_id): {"hello": "world"}},
        )

    def test_document_delete_not_found(
        self, client, monkeypatch, document_object_mock
    ):
        document_id_mock = ObjectId()
        fake_delete = MagicMock()
        # return None as Not Found for document that will be deleted
        monkeypatch.setattr(
            "whyhow_api.routers.documents.delete_document",
            AsyncMock(return_value=None),
        )
        fake_delete.__getitem__.return_value.delete_document = AsyncMock(
            return_value=None
        )

        client.app.dependency_overrides[get_db] = lambda: AsyncMock()
        client.app.dependency_overrides[get_user] = lambda: ObjectId()
        client.app.dependency_overrides[get_db_client] = lambda: AsyncMock()
        client.app.dependency_overrides[get_settings] = lambda: AsyncMock()
        client.app.dependency_overrides[valid_document_id] = (
            lambda: document_object_mock
        )

        response = client.delete(f"/documents/{document_id_mock}")

        data = response.json()
        assert response.status_code == 404
        assert "detail" in data
        assert data["detail"] == "Document not found."

    def test_document_delete_successful(
        self, client, monkeypatch, document_object_mock
    ):
        document_id_mock = ObjectId()
        # return fake document that will be deleted
        monkeypatch.setattr(
            "whyhow_api.routers.documents.delete_document",
            AsyncMock(return_value=document_object_mock.model_dump()),
        )

        client.app.dependency_overrides[get_db] = lambda: AsyncMock()
        client.app.dependency_overrides[get_user] = lambda: ObjectId()
        client.app.dependency_overrides[get_db_client] = lambda: AsyncMock()
        client.app.dependency_overrides[get_settings] = lambda: AsyncMock()
        client.app.dependency_overrides[valid_document_id] = (
            lambda: document_object_mock
        )

        response = client.delete(f"/documents/{document_id_mock}")

        data = response.json()
        assert response.status_code == 200
        assert data["message"] == "Document deleted successfully."
        assert data["status"] == "success"
        assert data["count"] == 1


class TestGeneratePresigned:
    def test_object_does_not_exist(self, client, monkeypatch):
        # Dependency injection
        client.app.dependency_overrides[get_user] = lambda: ObjectId()
        client.app.dependency_overrides[get_llm_client] = lambda: MagicMock()

        # Patching
        fake_boto3 = Mock()

        # Mocking the head_object method
        def head_object(*args, **kwargs):
            raise ClientError(error_response={}, operation_name="head_object")

        fake_boto3.client.return_value.head_object.side_effect = head_object
        monkeypatch.setattr("whyhow_api.routers.documents.boto3", fake_boto3)

        # Mocking the generate_presigned_post
        fake_boto3.client.return_value.generate_presigned_post.return_value = {
            "url": "https://test.com",
            "fields": {"hello": "world"},
        }

        # Run test
        data = {"filename": "test.txt", "workspace_id": str(ObjectId)}
        response = client.post("/documents/generate_presigned", json=data)

        assert response.status_code == 200

        assert response.json() == {
            "url": "https://test.com",
            "fields": {"hello": "world"},
        }

    def test_object_does_exist(self, client, monkeypatch):
        # Dependency injection
        client.app.dependency_overrides[get_user] = lambda: ObjectId()
        client.app.dependency_overrides[get_llm_client] = lambda: MagicMock()

        # Patching
        fake_boto3 = Mock()

        # Mocking the head_object method
        fake_boto3.client.return_value.head_object.return_value = {}
        monkeypatch.setattr("whyhow_api.routers.documents.boto3", fake_boto3)

        # Run test
        data = {"filename": "test.txt", "workspace_id": str(ObjectId)}
        response = client.post("/documents/generate_presigned", json=data)

        assert response.status_code == 404
        assert response.json() == {"detail": "Document already exists."}
