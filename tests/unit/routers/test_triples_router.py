from unittest.mock import AsyncMock

import pytest
from bson import ObjectId

from whyhow_api.dependencies import (
    get_db,
    get_db_client,
    get_llm_client,
    get_user,
    valid_triple_id,
)
from whyhow_api.routers.graphs import order_query
from whyhow_api.schemas.base import get_utc_now
from whyhow_api.schemas.chunks import ChunksOutWithWorkspaceDetails
from whyhow_api.schemas.graphs import DetailedGraphDocumentModel
from whyhow_api.schemas.triples import TripleDocumentModel, TripleOut


class TestTriplesGetOne:

    @pytest.fixture
    def triple_object_mock(self):
        return TripleDocumentModel(
            _id=ObjectId(),
            head_node=ObjectId(),
            tail_node=ObjectId(),
            type="test triple type",
            graph=ObjectId(),
            properties={"test": "test"},
            created_by=ObjectId(),
            chunks=[ObjectId(), ObjectId()],
        )

    def test_get_triple_successful(self, client, triple_object_mock):
        triple_id_mock = ObjectId()

        client.app.dependency_overrides[get_db] = lambda: AsyncMock()
        client.app.dependency_overrides[get_user] = lambda: ObjectId()
        client.app.dependency_overrides[valid_triple_id] = (
            lambda: triple_object_mock
        )

        response = client.get(f"/triples/{triple_id_mock}")
        assert response.status_code == 200

        data = response.json()
        assert data["message"] == "Triple retrieved successfully."
        assert data["status"] == "success"
        assert data["count"] == 1
        assert data["triples"][0]["type"] == "test triple type"
        assert data["triples"][0]["properties"] == {"test": "test"}

    def test_get_triple_failure(self, client, monkeypatch):

        triple_id_mock = ObjectId()

        monkeypatch.setattr("whyhow_api.dependencies.get_one", None)

        client.app.dependency_overrides[get_db] = lambda: AsyncMock()
        client.app.dependency_overrides[get_user] = lambda: ObjectId()

        response = client.get(f"/triples/{triple_id_mock}")
        assert response.status_code == 404

        data = response.json()
        assert data["detail"] == "Triple not found"


class TestTriplesGetAll:

    @pytest.fixture
    def triple_object_mock(self):
        return TripleDocumentModel(
            _id=ObjectId(),
            head_node=ObjectId(),
            tail_node=ObjectId(),
            type="test triple type",
            graph=ObjectId(),
            properties={"test": "test"},
            created_by=ObjectId(),
            chunks=[ObjectId(), ObjectId()],
        )

    def test_get_triples_successful(
        self, client, monkeypatch, triple_object_mock
    ):

        fake_get_all = AsyncMock()
        fake_get_all.return_value = [triple_object_mock.model_dump()]
        monkeypatch.setattr("whyhow_api.routers.triples.get_all", fake_get_all)

        fake_get_all_count = AsyncMock()
        fake_get_all_count.return_value = 1
        monkeypatch.setattr(
            "whyhow_api.routers.triples.get_all_count", fake_get_all_count
        )

        client.app.dependency_overrides[get_db] = lambda: AsyncMock()
        client.app.dependency_overrides[get_user] = lambda: ObjectId()
        # verify response with type, graph_id, head_node_id and tail_node_id params
        response = client.get(
            "/triples",
            params={
                "skip": 0,
                "limit": 1,
                "type": "test triple type",
                "graph_id": ObjectId(),
                "head_node_id": ObjectId(),
                "tail_node_id": ObjectId(),
            },
        )
        assert response.status_code == 200

        data = response.json()
        assert data["message"] == "Triples retrieved successfully."
        assert data["status"] == "success"
        assert data["count"] == 1
        assert data["triples"][0]["type"] == "test triple type"
        assert data["triples"][0]["properties"] == {"test": "test"}

        # verify response with graph_name and chunk_ids params
        response = client.get(
            "/triples",
            params={
                "skip": 0,
                "limit": 1,
                "type": "test triple type",
                "graph_name": "test_graph",
            },
        )
        assert response.status_code == 200

        data = response.json()
        assert data["message"] == "Triples retrieved successfully."
        assert data["status"] == "success"
        assert data["count"] == 1
        assert data["triples"][0]["type"] == "test triple type"
        assert data["triples"][0]["properties"] == {"test": "test"}

    def test_get_triples_error_both_graph_id_and_graph_name_specified(
        self, client, monkeypatch, triple_object_mock
    ):

        fake_get_all = AsyncMock()
        fake_get_all.return_value = [triple_object_mock.model_dump()]
        monkeypatch.setattr("whyhow_api.routers.triples.get_all", fake_get_all)

        fake_get_all_count = AsyncMock()
        fake_get_all_count.return_value = 1
        monkeypatch.setattr(
            "whyhow_api.routers.triples.get_all_count", fake_get_all_count
        )

        client.app.dependency_overrides[get_db] = lambda: AsyncMock()
        client.app.dependency_overrides[get_user] = lambda: ObjectId()

        response = client.get(
            "/triples",
            params={
                "skip": 0,
                "limit": 1,
                "graph_name": "test_graph",
                "graph_id": str(ObjectId()),
            },
        )
        assert response.status_code == 400
        assert (
            response.json()["detail"]
            == "Both graph_id and graph_name cannot be provided."
        )


class TestTriplesGetChunks:

    @pytest.fixture
    def triple_object_mock(self):
        return TripleDocumentModel(
            _id=ObjectId(),
            head_node=ObjectId(),
            tail_node=ObjectId(),
            type="test triple type",
            graph=ObjectId(),
            properties={"test": "test"},
            created_by=ObjectId(),
            chunks=[ObjectId(), ObjectId()],
        )

    @pytest.fixture
    def chunk_object_mock(self):
        workspace_id = ObjectId()
        return ChunksOutWithWorkspaceDetails(
            _id=ObjectId(),
            created_by=ObjectId(),
            created_at=get_utc_now(),
            updated_at=get_utc_now(),
            workspaces=[{"_id": workspace_id, "name": "workspace_name"}],
            content="test chunk text",
            document=None,
            data_type="string",
            tags={str(workspace_id): ["test"]},
            metadata={
                "language": "en",
                "size": 10,
                "data_source_type": "manual",
            },
            user_metadata={str(workspace_id): {"test": "test"}},
        )

    def test_get_triples_chunks_successful(
        self, client, monkeypatch, chunk_object_mock, triple_object_mock
    ):
        count = 2
        fake_get_chunks = AsyncMock()
        fake_get_chunks.return_value = ([chunk_object_mock], count)
        monkeypatch.setattr(
            "whyhow_api.routers.triples.get_triple_chunks", fake_get_chunks
        )

        client.app.dependency_overrides[get_db] = lambda: AsyncMock()
        client.app.dependency_overrides[get_user] = lambda: ObjectId()
        client.app.dependency_overrides[valid_triple_id] = (
            lambda: triple_object_mock
        )

        response = client.get(f"/triples/{ObjectId()}/chunks")
        assert response.status_code == 200

        data = response.json()
        assert data["message"] == "Triple chunks retrieved successfully."
        assert data["status"] == "success"
        assert data["count"] == count
        assert data["chunks"][0]["content"] == "test chunk text"
        assert data["chunks"][0]["data_type"] == "string"

    def test_get_triples_chunks_not_found(
        self, client, monkeypatch, triple_object_mock
    ):
        count = 0
        fake_get_chunks = AsyncMock()
        fake_get_chunks.return_value = (None, count)
        monkeypatch.setattr(
            "whyhow_api.routers.triples.get_triple_chunks", fake_get_chunks
        )

        client.app.dependency_overrides[get_db] = lambda: AsyncMock()
        client.app.dependency_overrides[get_user] = lambda: ObjectId()
        client.app.dependency_overrides[valid_triple_id] = (
            lambda: triple_object_mock
        )

        response = client.get(f"/triples/{ObjectId()}/chunks")

        assert response.status_code == 404
        data = response.json()
        assert data["detail"] == "No chunks found."

    @pytest.fixture
    def public_graph_object_mock(self):
        return DetailedGraphDocumentModel(
            _id=ObjectId(),
            name="test graph",
            workspace={"_id": ObjectId(), "name": "workspace"},
            schema_={"_id": ObjectId(), "name": "workspace"},
            status="ready",
            public=True,
            created_by=ObjectId(),
        )

    def test_get_public_triple_with_chunks_successful(
        self,
        client,
        monkeypatch,
        chunk_object_mock,
        triple_object_mock,
        public_graph_object_mock,
    ):
        fake_db = AsyncMock()
        triple_object_mock.graph = public_graph_object_mock.id
        fake_db.triple.find_one.return_value = triple_object_mock.model_dump()

        count = 2
        fake_get_chunks = AsyncMock()
        fake_get_chunks.return_value = ([chunk_object_mock], count)
        monkeypatch.setattr(
            "whyhow_api.routers.triples.get_triple_chunks", fake_get_chunks
        )

        fake_valid_public_graph_id = AsyncMock()
        monkeypatch.setattr(
            "whyhow_api.routers.triples.valid_public_graph_id",
            fake_valid_public_graph_id,
        )

        client.app.dependency_overrides[get_db] = lambda: fake_db
        client.app.dependency_overrides[order_query] = lambda: count

        response = client.get(f"triples/public/{ObjectId()}/chunks")
        assert response.status_code == 200

        data = response.json()
        assert data["message"] == "Triple chunks retrieved successfully."
        assert data["status"] == "success"
        assert data["count"] == count
        assert "chunks" in data
        assert len(data["chunks"]) > 0
        chunk = data["chunks"][0]
        assert chunk["content"] == "test chunk text"
        assert chunk["metadata"]["data_source_type"] == "manual"
        assert chunk["metadata"]["language"] == "en"
        assert chunk["metadata"]["size"] == 10
        assert chunk["user_metadata"] == chunk_object_mock.user_metadata
        assert chunk["tags"] == chunk_object_mock.tags

    def test_get_public_triple_with_chunks_triple_not_found(self, client):
        fake_db = AsyncMock()
        fake_db.triple.find_one.return_value = None

        client.app.dependency_overrides[get_db] = lambda: fake_db

        response = client.get(f"triples/public/{ObjectId()}/chunks")
        assert response.status_code == 404
        data = response.json()
        assert data["detail"] == "Triple not found."

    def test_get_public_triple_with_chunks_not_found(
        self,
        client,
        monkeypatch,
        triple_object_mock,
        public_graph_object_mock,
    ):
        fake_db = AsyncMock()
        triple_object_mock.graph = public_graph_object_mock.id
        fake_db.triple.find_one.return_value = triple_object_mock.model_dump()

        count = 0
        fake_get_chunks = AsyncMock()
        fake_get_chunks.return_value = (None, count)
        monkeypatch.setattr(
            "whyhow_api.routers.triples.get_triple_chunks", fake_get_chunks
        )

        fake_valid_public_graph_id = AsyncMock()
        monkeypatch.setattr(
            "whyhow_api.routers.triples.valid_public_graph_id",
            fake_valid_public_graph_id,
        )

        client.app.dependency_overrides[get_db] = lambda: fake_db
        client.app.dependency_overrides[order_query] = lambda: count

        response = client.get(f"triples/public/{ObjectId()}/chunks")
        assert response.status_code == 404
        data = response.json()
        assert data["detail"] == "No chunks found."


class TestTriplesCreate:

    @pytest.fixture
    def create_triple_body(self):
        return {
            "graph": str(ObjectId()),  # Generate a valid ObjectId string
            "triples": [
                {
                    "head_node": {
                        "name": "triple head name",
                        "type": "triple head type",
                        "properties": {},
                    },
                    "tail_node": str(
                        ObjectId()
                    ),  # Generate a valid ObjectId string
                    "type": "triple type",
                    "properties": {},
                    "chunks": [
                        str(ObjectId())
                    ],  # Generate valid ObjectId strings
                    "graph": str(
                        ObjectId()
                    ),  # Generate a valid ObjectId string
                }
            ],
        }

    def test_create_triple_endpoint_graph_not_found(
        self, client, create_triple_body
    ):
        fake_db = AsyncMock()
        fake_db.graph.find_one.return_value = None

        client.app.dependency_overrides[get_db] = lambda: fake_db
        client.app.dependency_overrides[get_db_client] = lambda: AsyncMock()
        client.app.dependency_overrides[get_user] = lambda: ObjectId()
        client.app.dependency_overrides[get_llm_client] = lambda: AsyncMock()

        response = client.post("/triples", json=create_triple_body)

        assert response.status_code == 404
        data = response.json()
        assert data["detail"] == "Graph not found."

    def test_create_triple_endpoint_node_not_found(
        self, client, create_triple_body
    ):
        fake_db = AsyncMock()
        fake_db.node.find_one.return_value = None

        client.app.dependency_overrides[get_db] = lambda: fake_db
        client.app.dependency_overrides[get_db_client] = lambda: AsyncMock()
        client.app.dependency_overrides[get_user] = lambda: ObjectId()
        client.app.dependency_overrides[get_llm_client] = lambda: AsyncMock()

        response = client.post("/triples", json=create_triple_body)

        assert response.status_code == 404
        data = response.json()
        assert data["detail"] == "Node not found."


class TestTriplesDelete:

    def test_delete_tripple_not_found(self, client, monkeypatch):

        prompt_id_mock = ObjectId()

        monkeypatch.setattr("whyhow_api.dependencies.get_one", None)

        client.app.dependency_overrides[get_db] = lambda: AsyncMock()
        client.app.dependency_overrides[get_user] = lambda: ObjectId()
        client.app.dependency_overrides[get_db_client] = lambda: AsyncMock()

        response = client.delete(f"/triples/{prompt_id_mock}")
        assert response.status_code == 404

        data = response.json()
        assert data["detail"] == "Triple not found"

    @pytest.fixture
    def triple_object_mock(self):
        return TripleDocumentModel(
            head_node=ObjectId(),
            tail_node=ObjectId(),
            type="test triple type",
            graph=ObjectId(),
            properties={"test": "test"},
            created_by=ObjectId(),
            chunks=[ObjectId(), ObjectId()],
        )

    @pytest.fixture
    def triple_out_mock(self):
        return TripleOut(
            id=ObjectId(),
            head_node=ObjectId(),
            tail_node=ObjectId(),
            type="test triple type",
            graph=ObjectId(),
            properties={"test": "test"},
            created_by=ObjectId(),
            chunks=[ObjectId(), ObjectId()],
        )

    @pytest.mark.asyncio
    async def test_delete_triple_success(
        self, client, triple_object_mock, triple_out_mock
    ):
        prompt_id_mock = ObjectId()

        client.app.dependency_overrides[get_db] = lambda: AsyncMock()
        client.app.dependency_overrides[get_user] = lambda: ObjectId()
        client.app.dependency_overrides[get_db_client] = lambda: AsyncMock()
        client.app.dependency_overrides[valid_triple_id] = (
            lambda: triple_object_mock
        )

        response = client.delete(f"/triples/{prompt_id_mock}")

        assert response.status_code == 200

        data = response.json()
        triples = data["triples"]

        assert data["message"] == "Triple deleted successfully."
        assert data["status"] == "success"
        assert len(triples) == 1
