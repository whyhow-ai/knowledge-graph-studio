from unittest.mock import AsyncMock

import pytest
from bson import ObjectId

from whyhow_api.dependencies import get_db, get_user, valid_query_id
from whyhow_api.schemas.queries import QueryDocumentModel, QueryParameters
from whyhow_api.utilities.routers import order_query


@pytest.mark.skip("Need to fix the test")
class TestPromptsGetOne:

    def test_get_query_successful(self, client, monkeypatch):

        query_id_mock = ObjectId()
        query_object_mock = QueryDocumentModel(
            query=QueryParameters(content="test query content"),
            created_by=ObjectId(),
            status="success",
        )

        client.app.dependency_overrides[get_db] = lambda: AsyncMock()
        client.app.dependency_overrides[get_user] = lambda: ObjectId()
        client.app.dependency_overrides[valid_query_id] = (
            lambda: query_object_mock
        )

        response = client.get(f"/prompts/{query_id_mock}")
        assert response.status_code == 200

        data = response.json()
        assert data["message"] == "Prompt retrieved successfully."
        assert data["status"] == "success"
        assert data["count"] == 1
        assert data["prompts"][0]["query"]["content"] == "test query content"
        assert data["prompts"][0]["created_by"] == str(
            query_object_mock.created_by
        )

    def test_get_prompt_failure(self, client, monkeypatch):

        query_id_mock = ObjectId()

        monkeypatch.setattr("whyhow_api.dependencies.get_one", None)

        client.app.dependency_overrides[get_db] = lambda: AsyncMock()
        client.app.dependency_overrides[get_user] = lambda: ObjectId()

        response = client.get(f"/prompts/{query_id_mock}")
        assert response.status_code == 404

        data = response.json()
        assert data["detail"] == "Prompt not found"


@pytest.mark.skip("Need to fix the test")
class TestPromptsGetAll:

    def test_get_prompts_successful(self, client, monkeypatch):

        query_object_mock = QueryDocumentModel(
            content="test content",
            type="query",
            created_by=ObjectId(),
            status="success",
        )

        fake_get_all = AsyncMock()
        fake_get_all.return_value = [query_object_mock.model_dump()]
        monkeypatch.setattr("whyhow_api.routers.prompts.get_all", fake_get_all)

        fake_count = 1
        fake_get_all_count = AsyncMock()
        fake_get_all_count.return_value = fake_count
        monkeypatch.setattr(
            "whyhow_api.routers.prompts.get_all_count", fake_get_all_count
        )

        client.app.dependency_overrides[get_db] = lambda: AsyncMock()
        client.app.dependency_overrides[get_user] = lambda: ObjectId()
        # send request with graph_id param
        response = client.get(
            "/prompts",
            params={
                "skip": 0,
                "limit": 1,
                "type": "query",
                "graph_id": ObjectId(),
            },
        )
        assert response.status_code == 200

        data = response.json()
        assert data["message"] == "Prompts retrieved successfully."
        assert data["status"] == "success"
        assert data["count"] == fake_count
        assert data["prompts"][0]["content"] == "test content"
        assert data["prompts"][0]["type"] == "query"
        assert data["prompts"][0]["type"] == "query"
        assert data["prompts"][0]["created_by"] == str(
            query_object_mock.created_by
        )

        # send request with graph_name param
        response = client.get(
            "/prompts",
            params={
                "skip": 0,
                "limit": 1,
                "type": "query",
                "graph_name": "test",
            },
        )
        assert response.status_code == 200

    def test_get_all_prompts_both_graph_id_and_name_specified_error(
        self, client
    ):
        client.app.dependency_overrides[get_db] = lambda: AsyncMock()
        client.app.dependency_overrides[get_user] = lambda: ObjectId()
        client.app.dependency_overrides[order_query] = lambda: 1

        response = client.get(
            "/prompts", params={"graph_id": "123", "graph_name": "test"}
        )
        assert response.status_code == 400
        data = response.json()
        assert (
            data["detail"]
            == "Both graph_id and graph_name cannot be provided."
        )


@pytest.mark.skip("Need to fix the test")
class TestPromptsDelete:

    @pytest.fixture
    def query_object_mock(self):
        return QueryDocumentModel(
            _id=ObjectId(),
            content="test content",
            type="query",
            created_by=ObjectId(),
        )

    def test_delete_prompt_successful(
        self, client, monkeypatch, query_object_mock
    ):

        query_id_mock = ObjectId()

        fake_delete = AsyncMock()
        fake_delete.return_value = query_object_mock
        monkeypatch.setattr(
            "whyhow_api.routers.prompts.delete_one", fake_delete
        )

        client.app.dependency_overrides[get_db] = lambda: AsyncMock()
        client.app.dependency_overrides[get_user] = lambda: ObjectId()
        client.app.dependency_overrides[valid_query_id] = (
            lambda: query_object_mock
        )

        response = client.delete(f"/prompts/{query_id_mock}")
        assert response.status_code == 200

        data = response.json()
        assert data["message"] == "Prompt deleted successfully."
        assert data["status"] == "success"
        assert data["count"] == 1
        assert data["prompts"][0]["content"] == "test content"

    def test_delete_prompt_not_found(self, client, monkeypatch):

        monkeypatch.setattr("whyhow_api.dependencies.get_one", None)

        client.app.dependency_overrides[get_db] = lambda: AsyncMock()
        client.app.dependency_overrides[get_user] = lambda: ObjectId()

        response = client.delete(f"/prompts/{ObjectId()}")
        assert response.status_code == 404

        data = response.json()
        assert data["detail"] == "Prompt not found"

    def test_delete_prompt_failure(self, client, monkeypatch):

        fake_delete = AsyncMock()
        fake_delete.return_value = None
        monkeypatch.setattr(
            "whyhow_api.routers.prompts.delete_one", fake_delete
        )

        client.app.dependency_overrides[get_db] = lambda: AsyncMock()
        client.app.dependency_overrides[get_user] = lambda: ObjectId()

        response = client.delete(f"/prompts/{ObjectId()}")
        assert response.status_code == 404

        data = response.json()
        assert data["detail"] == "Prompt not found"
