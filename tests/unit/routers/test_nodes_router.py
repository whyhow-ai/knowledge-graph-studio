from unittest.mock import AsyncMock

import pytest
from bson import ObjectId

from whyhow_api.dependencies import (
    get_db,
    get_db_client,
    get_llm_client,
    get_user,
    valid_node_id,
)
from whyhow_api.schemas.base import get_utc_now
from whyhow_api.schemas.chunks import (
    ChunkMetadata,
    ChunksOutWithWorkspaceDetails,
)
from whyhow_api.schemas.graphs import DetailedGraphDocumentModel
from whyhow_api.schemas.nodes import NodeDocumentModel
from whyhow_api.schemas.schemas import (
    SchemaDocumentModel,
    SchemaEntity,
    SchemaRelation,
    SchemaTriplePattern,
)
from whyhow_api.schemas.workspaces import WorkspaceDocumentModel


class TestNodesGetOne:

    @pytest.fixture
    def node_object_mock(self):
        return NodeDocumentModel(
            _id=ObjectId(),
            name="test node name",
            type="test node type",
            graph=ObjectId(),
            properties={"test": "test"},
            created_by=ObjectId(),
        )

    def test_get_node_successful(self, client, node_object_mock):
        node_id_mock = ObjectId()

        client.app.dependency_overrides[get_db] = lambda: AsyncMock()
        client.app.dependency_overrides[get_user] = lambda: ObjectId()
        client.app.dependency_overrides[valid_node_id] = (
            lambda: node_object_mock
        )

        response = client.get(f"/nodes/{node_id_mock}")
        assert response.status_code == 200

        data = response.json()
        assert data["message"] == "Node retrieved successfully"
        assert data["status"] == "success"
        assert data["count"] == 1
        assert data["nodes"][0]["name"] == "test node name"
        assert data["nodes"][0]["type"] == "test node type"
        assert data["nodes"][0]["properties"] == {"test": "test"}

    def test_get_node_failure(self, client, monkeypatch):

        node_id_mock = ObjectId()

        monkeypatch.setattr("whyhow_api.dependencies.get_one", None)

        client.app.dependency_overrides[get_db] = lambda: AsyncMock()
        client.app.dependency_overrides[get_user] = lambda: ObjectId()

        response = client.get(f"/nodes/{node_id_mock}")
        assert response.status_code == 404

        data = response.json()
        assert data["detail"] == "Node not found"


class TestNodesGetAll:

    @pytest.fixture
    def node_object_mock(self):
        return NodeDocumentModel(
            _id=ObjectId(),
            name="test node name",
            type="test node type",
            graph=ObjectId(),
            properties={"test": "test"},
            created_by=ObjectId(),
        )

    def test_get_nodes_successful(self, client, monkeypatch, node_object_mock):

        fake_get_all = AsyncMock()
        fake_get_all.return_value = [node_object_mock.model_dump()]
        monkeypatch.setattr("whyhow_api.routers.nodes.get_all", fake_get_all)

        fake_get_all_count = AsyncMock()
        fake_get_all_count.return_value = 1
        monkeypatch.setattr(
            "whyhow_api.routers.nodes.get_all_count", fake_get_all_count
        )

        client.app.dependency_overrides[get_db] = lambda: AsyncMock()
        client.app.dependency_overrides[get_user] = lambda: ObjectId()

        response = client.get("/nodes", params={"skip": 0, "limit": 1})
        assert response.status_code == 200

        data = response.json()
        assert data["message"] == "Nodes retrieved successfully"
        assert data["status"] == "success"
        assert data["count"] == 1


class TestNodeGetChunks:

    @pytest.fixture
    def workspace_object_mock(self):
        user_id_mock = str(ObjectId())
        return WorkspaceDocumentModel(
            name="test workspace", created_by=user_id_mock
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
            content={"chunk1": "value of chunk"},
            document=None,
            data_type="string",
            tags={str(workspace_id): ["test"]},
            metadata=ChunkMetadata(
                language="en",
                size=10,
                length=14,
                data_source_type="manual",
                index=1,
                page=1,
                start=0,
                end=10,
            ),
            user_metadata={"test": {"test": "test"}},
        )

    def test_get_node_with_chunks_not_found(
        self, client, monkeypatch, chunk_object_mock
    ):
        node_id_mock = ObjectId()

        client.app.dependency_overrides[get_db] = lambda: AsyncMock()
        client.app.dependency_overrides[get_user] = lambda: ObjectId()
        client.app.dependency_overrides[valid_node_id] = (
            lambda: chunk_object_mock
        )

        # return None as Not found for node chunks
        fake_find_chunks = AsyncMock()
        fake_find_chunks.return_value = None
        monkeypatch.setattr(
            "whyhow_api.routers.nodes.get_node_chunks", fake_find_chunks
        )

        response = client.get(f"/nodes/{node_id_mock}/chunks")
        assert response.status_code == 404

        data = response.json()
        assert "detail" in data
        assert data["detail"] == "No chunks found for the node."

    def test_get_node_with_chunks_successful(
        self,
        client,
        monkeypatch,
        chunk_object_mock,
    ):
        node_id_mock = ObjectId()

        client.app.dependency_overrides[get_db] = lambda: AsyncMock()
        client.app.dependency_overrides[get_user] = lambda: ObjectId()
        client.app.dependency_overrides[valid_node_id] = (
            lambda: chunk_object_mock
        )

        # return fake node list of chunks object
        fake_find_chunks = AsyncMock()
        fake_find_chunks.return_value = [chunk_object_mock]
        monkeypatch.setattr(
            "whyhow_api.routers.nodes.get_node_chunks", fake_find_chunks
        )

        response = client.get(f"/nodes/{node_id_mock}/chunks")
        assert response.status_code == 200

        data = response.json()
        assert "message" in data
        assert data["message"] == "Node with chunks retrieved successfully."
        assert "status" in data
        assert data["status"] == "success"
        assert "chunks" in data
        assert isinstance(data["chunks"], list)
        assert len(data["chunks"]) == 1

        chunk = data["chunks"][0]
        assert chunk["_id"] == str(chunk_object_mock.id)
        assert chunk["created_by"] == str(chunk_object_mock.created_by)
        assert chunk["workspaces"][0]["_id"] == str(
            chunk_object_mock.workspaces[0].id
        )
        assert (
            chunk["workspaces"][0]["name"]
            == chunk_object_mock.workspaces[0].name
        )
        assert chunk["content"] == chunk_object_mock.content
        assert chunk["document"] == chunk_object_mock.document
        assert chunk["data_type"] == chunk_object_mock.data_type
        assert chunk["tags"] == chunk_object_mock.tags
        assert (
            chunk["metadata"]["language"]
            == chunk_object_mock.metadata.language
        )
        assert chunk["metadata"]["size"] == chunk_object_mock.metadata.size
        assert chunk["metadata"]["length"] == chunk_object_mock.metadata.length
        assert (
            chunk["metadata"]["data_source_type"]
            == chunk_object_mock.metadata.data_source_type
        )
        assert chunk["metadata"]["index"] == chunk_object_mock.metadata.index
        assert chunk["metadata"]["page"] == chunk_object_mock.metadata.page
        assert chunk["metadata"]["start"] == chunk_object_mock.metadata.start
        assert chunk["metadata"]["end"] == chunk_object_mock.metadata.end
        assert chunk["user_metadata"] == chunk_object_mock.user_metadata

    def test_node_get_chunks_not_found(self, client, monkeypatch):
        node_id_mock = ObjectId()
        fake_get_node_chunks = AsyncMock()
        fake_get_node_chunks.return_value = None
        monkeypatch.setattr(
            "whyhow_api.routers.nodes.get_node_chunks", fake_get_node_chunks
        )

        client.app.dependency_overrides[get_db] = lambda: AsyncMock()
        client.app.dependency_overrides[get_user] = lambda: ObjectId()

        response = client.get(f"/{node_id_mock}/chunks")
        data = response.json()
        assert response.status_code == 404
        assert "detail" in data
        assert data["detail"] != ""


class TestNodesCreate:

    @pytest.fixture
    def graph_object_mock(self):
        return DetailedGraphDocumentModel(
            _id=ObjectId(),
            name="test graph",
            workspace={"_id": ObjectId(), "name": "workspace"},
            schema_={"_id": ObjectId(), "name": "workspace"},
            status="ready",
            public=False,
            created_by=ObjectId(),
        )

    @pytest.fixture
    def schema_object_mock(self):
        return SchemaDocumentModel(
            _id=ObjectId(),
            name="test schema",
            workspace=ObjectId(),
            entities=[
                SchemaEntity(
                    name="test node type", description="entity description"
                ),
                SchemaEntity(
                    name="entity2", description="entity2 description"
                ),
            ],
            relations=[
                SchemaRelation(
                    name="relation", description="relation description"
                ),
            ],
            patterns=[
                SchemaTriplePattern(
                    head=SchemaEntity(
                        name="test node type",
                        description="test node type description",
                    ),
                    tail=SchemaEntity(
                        name="entity2", description="entity2 description"
                    ),
                    relation=SchemaRelation(
                        name="relation", description="relation description"
                    ),
                    description="pattern description",
                )
            ],
            created_by=ObjectId(),
        )

    @pytest.fixture
    def node_object_mock(self):
        return NodeDocumentModel(
            _id=ObjectId(),
            name="test node name",
            type="test node type",
            graph=ObjectId(),
            properties={"test": "test"},
            created_by=ObjectId(),
            chunks=[ObjectId()],
        )

    def test_create_node_successful(
        self,
        client,
        monkeypatch,
        node_object_mock,
        graph_object_mock,
        schema_object_mock,
    ):

        fake_valid_graph_id = AsyncMock()
        fake_valid_graph_id.return_value = graph_object_mock
        monkeypatch.setattr(
            "whyhow_api.routers.nodes.valid_graph_id", fake_valid_graph_id
        )

        fake_schema = AsyncMock()
        fake_schema.return_value = schema_object_mock
        monkeypatch.setattr("whyhow_api.routers.nodes.get_one", fake_schema)

        # Mocking db.node.find_one
        fake_find_one = AsyncMock()
        fake_find_one.return_value = None  # Simulate no node found
        db_mock = AsyncMock()
        db_mock.node.find_one = fake_find_one
        client.app.dependency_overrides[get_db] = lambda: db_mock

        fake_create = AsyncMock()
        fake_create.return_value = node_object_mock.model_dump()
        monkeypatch.setattr("whyhow_api.routers.nodes.create_one", fake_create)

        client.app.dependency_overrides[get_user] = (
            lambda: node_object_mock.created_by
        )

        create_node_body = {
            "name": node_object_mock.name,
            "type": node_object_mock.type,
            "properties": node_object_mock.properties,
            "graph": str(node_object_mock.graph),
            "chunks": [str(chunk) for chunk in node_object_mock.chunks],
        }

        response = client.post("/nodes", json=create_node_body)
        assert response.status_code == 200

        data = response.json()
        assert data["message"] == "Node created successfully"
        assert data["status"] == "success"
        assert data["count"] == 1

    def test_create_node_graph_not_found(
        self, client, monkeypatch, node_object_mock
    ):

        fake_get_one = AsyncMock()
        fake_get_one.return_value = None
        monkeypatch.setattr("whyhow_api.routers.nodes.get_one", fake_get_one)

        client.app.dependency_overrides[get_db] = lambda: AsyncMock()
        client.app.dependency_overrides[get_user] = lambda: ObjectId()

        create_node_body = {
            "name": node_object_mock.name,
            "type": node_object_mock.type,
            "properties": node_object_mock.properties,
            "graph": str(node_object_mock.graph),
            "chunks": [str(chunk) for chunk in node_object_mock.chunks],
            "created_by": str(node_object_mock.created_by),
        }

        response = client.post("/nodes", json=create_node_body)
        data = response.json()
        assert response.status_code == 404
        assert "detail" in data
        assert data["detail"] == "Graph not found"


class TestNodesUpdate:

    @pytest.fixture
    def node_object_mock(self):
        return NodeDocumentModel(
            _id=ObjectId(),
            name="test node name",
            type="test node type",
            graph=ObjectId(),
            properties={"test": "test"},
            created_by=ObjectId(),
        )

    def test_update_node_successful(
        self, client, monkeypatch, node_object_mock
    ):

        node_id_mock = ObjectId()

        fake_update = AsyncMock()
        fake_update.return_value = node_object_mock.model_dump()
        monkeypatch.setattr(
            "whyhow_api.routers.nodes.update_node", fake_update
        )

        client.app.dependency_overrides[get_db] = lambda: AsyncMock()
        client.app.dependency_overrides[get_user] = lambda: ObjectId()
        client.app.dependency_overrides[valid_node_id] = (
            lambda: node_object_mock
        )
        client.app.dependency_overrides[get_db_client] = lambda: AsyncMock()
        client.app.dependency_overrides[get_llm_client] = lambda: AsyncMock()

        response = client.put(
            f"/nodes/{node_id_mock}",
            json={
                "name": node_object_mock.name,
                "type": node_object_mock.type,
            },
        )
        assert response.status_code == 200

        data = response.json()
        assert data["message"] != ""
        assert data["status"] == "success"
        assert data["count"] == 1
        assert data["nodes"][0]["name"] == node_object_mock.name
        assert data["nodes"][0]["type"] == node_object_mock.type
        assert data["nodes"][0]["created_by"] == str(
            node_object_mock.created_by
        )

    def test_update_node_not_found(
        self, client, monkeypatch, node_object_mock
    ):

        monkeypatch.setattr("whyhow_api.dependencies.get_one", None)

        client.app.dependency_overrides[get_db] = lambda: AsyncMock()
        client.app.dependency_overrides[get_user] = lambda: ObjectId()

        response = client.put(
            f"/nodes/{ObjectId()}",
            json={"name": node_object_mock.name},
        )
        assert response.status_code == 404

        data = response.json()
        assert "detail" in data
        assert data["detail"] == "Node not found"


class TestNodesDelete:

    @pytest.fixture
    def node_object_mock(self):
        return NodeDocumentModel(
            name="test node name",
            type="test node type",
            properties={"test": "test"},
            graph=ObjectId(),
            chunks=[ObjectId()],
            created_by=ObjectId(),
        )

    def test_delete_node_successful(
        self, client, monkeypatch, node_object_mock
    ):

        client.app.dependency_overrides[get_user] = lambda: ObjectId()
        client.app.dependency_overrides[get_db] = lambda: AsyncMock()
        client.app.dependency_overrides[valid_node_id] = (
            lambda: node_object_mock
        )
        client.app.dependency_overrides[get_db_client] = lambda: AsyncMock()

        node_id_mock = ObjectId()
        fake_delete = AsyncMock()
        fake_delete.return_value = node_object_mock
        monkeypatch.setattr(
            "whyhow_api.routers.nodes.delete_node", fake_delete
        )

        response = client.delete(
            f"/nodes/{node_id_mock}",
        )
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Node deleted successfully"
        assert data["status"] == "success"

    def test_delete_node_not_found(self, client, monkeypatch):

        monkeypatch.setattr("whyhow_api.dependencies.get_one", None)

        client.app.dependency_overrides[get_user] = lambda: ObjectId()
        client.app.dependency_overrides[get_db] = lambda: AsyncMock()
        client.app.dependency_overrides[get_db_client] = lambda: AsyncMock()

        node_id_mock = ObjectId()
        fake_delete = AsyncMock()
        fake_delete.return_value = None
        monkeypatch.setattr(
            "whyhow_api.routers.nodes.delete_node", fake_delete
        )

        response = client.delete(
            f"/nodes/{node_id_mock}",
        )
        assert response.status_code == 404
        data = response.json()
        assert data["detail"] == "Node not found"
