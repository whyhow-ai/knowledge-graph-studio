from unittest.mock import AsyncMock, MagicMock

import pytest
from bson import ObjectId

from whyhow_api.dependencies import (
    get_db,
    get_db_client,
    get_llm_client,
    get_user,
    valid_create_graph,
    valid_graph_id,
    valid_public_graph_id,
)
from whyhow_api.routers.graphs import order_query
from whyhow_api.schemas.graphs import DetailedGraphDocumentModel
from whyhow_api.schemas.nodes import NodeWithId
from whyhow_api.schemas.rules import MergeNodesRule, RuleOut
from whyhow_api.schemas.triples import RelationOut, TripleWithId
from whyhow_api.schemas.workspaces import WorkspaceDocumentModel


class TestGraphsGetOne:

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

    def test_get_graph_successful(self, client, graph_object_mock):
        graph_id_mock = ObjectId()
        client.app.dependency_overrides[get_db] = lambda: AsyncMock()
        client.app.dependency_overrides[get_user] = lambda: ObjectId()
        client.app.dependency_overrides[valid_graph_id] = (
            lambda: graph_object_mock
        )

        response = client.get(f"/graphs/{graph_id_mock}")
        assert response.status_code == 200

        data = response.json()
        assert data["message"] == "Successfully retrieved graph."
        assert data["status"] == "success"
        assert data["count"] == 1
        assert data["graphs"][0]["name"] == "test graph"

    def test_get_public_graph_successful(self, client, graph_object_mock):
        graph_id_mock = ObjectId()
        # set graph as public
        public_graph_object_mock = graph_object_mock
        public_graph_object_mock.public = True

        client.app.dependency_overrides[valid_public_graph_id] = (
            lambda: public_graph_object_mock
        )

        response = client.get(f"/graphs/public/{graph_id_mock}")
        assert response.status_code == 200

        data = response.json()

        assert data["message"] == "Successfully retrieved graph."
        assert data["status"] == "success"
        assert data["count"] == 1
        assert data["graphs"][0]["name"] == "test graph"
        assert data["graphs"][0]["created_by"] == str(
            public_graph_object_mock.created_by
        )
        assert data["graphs"][0]["public"] is True

    def test_get_graph_failure(self, client, monkeypatch):

        graph_id_mock = ObjectId()

        monkeypatch.setattr("whyhow_api.dependencies.get_graph", None)

        client.app.dependency_overrides[get_db] = lambda: AsyncMock()
        client.app.dependency_overrides[get_user] = lambda: ObjectId()

        response = client.get(f"/graphs/{graph_id_mock}")
        assert response.status_code == 404

        data = response.json()
        assert "detail" in data
        assert data["detail"] == "Graph not found"


class TestGraphsGetAll:

    @pytest.fixture
    def graph_object_mock(self):
        return DetailedGraphDocumentModel(
            _id=ObjectId(),
            name="test graph",
            workspace={"_id": ObjectId(), "name": "workspace"},
            schema_={"_id": ObjectId(), "name": "schema"},
            status="ready",
            public=False,
            created_by=ObjectId(),
        )

    def test_get_graphs_successful(
        self, client, monkeypatch, graph_object_mock
    ):

        fake_list_all_graphs = AsyncMock()
        fake_list_all_graphs.return_value = [graph_object_mock]
        monkeypatch.setattr(
            "whyhow_api.routers.graphs.get_all", fake_list_all_graphs
        )

        fake_get_all_count = AsyncMock()
        fake_get_all_count.return_value = 1
        monkeypatch.setattr(
            "whyhow_api.routers.graphs.get_all_count", fake_get_all_count
        )

        client.app.dependency_overrides[get_db] = lambda: AsyncMock()
        client.app.dependency_overrides[get_user] = lambda: ObjectId()
        # test with workspace_name and schema_name
        params = {
            "skip": 0,
            "limit": 1,
            "name": "test graph",
            "status": "ready",
            "workspace_name": "test_workspace",
            "schema_name": "test_schema",
        }

        response = client.get("/graphs", params=params)
        assert response.status_code == 200

        data = response.json()
        assert data["message"] == "Successfully retrieved graph."
        assert data["status"] == "success"
        assert data["count"] == 1
        assert data["graphs"][0]["name"] == "test graph"

        # test with workspace_id and schema_id
        params = {
            "skip": 0,
            "limit": 1,
            "name": "test graph",
            "status": "ready",
            "workspace_id": ObjectId(),
            "schema_id": ObjectId(),
        }

        response = client.get("/graphs", params=params)
        assert response.status_code == 200

        data = response.json()
        assert data["message"] == "Successfully retrieved graph."
        assert data["status"] == "success"
        assert data["count"] == 1
        assert data["graphs"][0]["name"] == "test graph"
        assert data["graphs"][0]["created_by"] == str(
            graph_object_mock.created_by
        )
        assert data["graphs"][0]["public"] is False

    def test_get_graphs_no_graphs(self, client, monkeypatch):
        client.app.dependency_overrides[get_db] = lambda: AsyncMock()
        client.app.dependency_overrides[get_user] = lambda: ObjectId()

        fake_get_all_count = AsyncMock(return_value=0)

        monkeypatch.setattr(
            "whyhow_api.routers.graphs.get_all_count", fake_get_all_count
        )
        params = {"skip": 0, "limit": 1}
        response = client.get("/graphs", params=params)

        assert response.status_code == 200

        data = response.json()
        assert data["message"] == "No graphs found."
        assert data["status"] == "success"
        assert data["count"] == 0
        assert data["graphs"] == []

    def test_get_graphs_with_both_workspace_id_and_workspace_name_error(
        self, client
    ):
        fake_workspace_id = ObjectId()

        client.app.dependency_overrides[order_query] = lambda: 1
        client.app.dependency_overrides[get_db] = lambda: AsyncMock()
        client.app.dependency_overrides[get_user] = lambda: ObjectId()

        params = {
            "workspace_id": fake_workspace_id,
            "workspace_name": "test_workspace_name",
            "type": "test_type",
        }

        response = client.get("/graphs", params=params)
        assert response.status_code == 400

        data = response.json()
        assert "detail" in data
        assert (
            data["detail"]
            == "Both workspace_id and workspace_name cannot be provided."
        )

    def test_get_graphs_with_both_schema_id_and_schema_name_error(
        self, client
    ):
        fake_schema_id = ObjectId()

        client.app.dependency_overrides[order_query] = lambda: 1
        client.app.dependency_overrides[get_db] = lambda: AsyncMock()
        client.app.dependency_overrides[get_user] = lambda: ObjectId()

        params = {
            "schema_id": fake_schema_id,
            "schema_name": "test_schema_name",
            "type": "test_type",
        }

        response = client.get("/graphs", params=params)
        assert response.status_code == 400

        data = response.json()
        assert "detail" in data
        assert (
            data["detail"]
            == "Both schema_id and schema_name cannot be provided."
        )


class TestGraphsCreate:

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

    @pytest.mark.skip(reason="API disabled for now")
    def test_graph_create_nothing_retrieved(self, client, monkeypatch):
        fake_db = AsyncMock()
        fake_db_client = AsyncMock()
        fake_graph_id = ObjectId()
        fake_db.insert_one.return_value.inserted_id = fake_graph_id
        fake_db.graph.find_one.return_value = None

        # Mock the fork_schema function
        fake_fork_schema = AsyncMock()
        fake_fork_schema.return_value = ObjectId()
        monkeypatch.setattr(
            "whyhow_api.services.graph_service.fork_schema", fake_fork_schema
        )

        client.app.dependency_overrides[get_db] = lambda: fake_db
        client.app.dependency_overrides[get_db_client] = lambda: fake_db_client
        client.app.dependency_overrides[get_user] = lambda: ObjectId()
        client.app.dependency_overrides[get_llm_client] = lambda: AsyncMock()
        client.app.dependency_overrides[valid_create_graph] = lambda: True

        request_body = {
            "name": "test graph",
            "workspace": str(ObjectId()),
            "schema": str(ObjectId()),
        }

        result = client.post("/graphs", json=request_body)

        assert result.status_code == 404
        assert result.json() == {"detail": "Failed to retrieve graph."}

    @pytest.mark.skip(reason="API disabled for now")
    def test_graph_create_graph_successful(
        self, client, monkeypatch, graph_object_mock
    ):
        fake_db = AsyncMock()
        fake_existing_graph_id = ObjectId()

        fake_db.graph.find_one.side_effect = [
            None,
            {
                "_id": fake_existing_graph_id,
                "status": "creating",
            },
        ]

        fake_db.graph.insert_one.return_value = AsyncMock(
            inserted_id=fake_existing_graph_id
        )

        client.app.dependency_overrides[get_db] = lambda: fake_db
        client.app.dependency_overrides[get_db_client] = lambda: AsyncMock()
        client.app.dependency_overrides[get_user] = lambda: ObjectId()
        client.app.dependency_overrides[get_llm_client] = lambda: AsyncMock()
        client.app.dependency_overrides[valid_create_graph] = lambda: True

        # Mock the fork_schema function
        fake_fork_schema = AsyncMock()
        fake_fork_schema.return_value = ObjectId()
        monkeypatch.setattr(
            "whyhow_api.services.graph_service.fork_schema", fake_fork_schema
        )

        # Mock the create_or_update_graph function
        fake_create_or_update_graph = AsyncMock()
        monkeypatch.setattr(
            "whyhow_api.services.graph_service.create_or_update_graph",
            fake_create_or_update_graph,
        )

        request_body = {
            "name": "test graph",
            "workspace": str(ObjectId()),
            "schema": str(ObjectId()),
        }

        response = client.post("/graphs", json=request_body)
        data = response.json()

        assert response.status_code == 200
        assert data["status"] == "success"
        assert data["message"] == "Hold tight - your graph is being created!"


class TestGraphsUpdate:

    @pytest.fixture
    def graph_object_mock(self):
        return DetailedGraphDocumentModel(
            _id=ObjectId(),
            name="test graph",
            workspace={"_id": ObjectId(), "name": "workspace"},
            schema_={"_id": ObjectId(), "name": "schema"},
            status="ready",
            public=False,
            created_by=ObjectId(),
        )

    def test_update_graph_successful(
        self, client, monkeypatch, graph_object_mock
    ):

        graph_id_mock = ObjectId()

        fake_update = AsyncMock()
        fake_update.return_value = graph_object_mock.model_dump()
        monkeypatch.setattr(
            "whyhow_api.routers.graphs.update_one", fake_update
        )

        client.app.dependency_overrides[get_db] = lambda: AsyncMock()
        client.app.dependency_overrides[get_user] = lambda: ObjectId()
        client.app.dependency_overrides[valid_graph_id] = (
            lambda: graph_object_mock
        )

        response = client.put(
            f"/graphs/{graph_id_mock}",
            json={"name": graph_object_mock.name},
        )
        assert response.status_code == 200

        data = response.json()
        assert data["message"] == "Graph updated successfully."
        assert data["status"] == "success"
        assert data["count"] == 1
        assert data["graphs"][0]["name"] == graph_object_mock.name
        assert data["graphs"][0]["created_by"] == str(
            graph_object_mock.created_by
        )

    def test_update_graph_not_found(
        self, client, monkeypatch, graph_object_mock
    ):

        monkeypatch.setattr("whyhow_api.dependencies.get_graph", None)

        client.app.dependency_overrides[get_db] = lambda: AsyncMock()
        client.app.dependency_overrides[get_user] = lambda: ObjectId()

        response = client.put(
            f"/graphs/{ObjectId()}",
            json={"name": graph_object_mock.name},
        )
        assert response.status_code == 404

        data = response.json()
        assert "detail" in data
        assert data["detail"] == "Graph not found"


class TestGraphsDelete:

    @pytest.fixture
    def graph_object_mock(self):
        return DetailedGraphDocumentModel(
            _id=ObjectId(),
            name="test graph",
            workspace={"_id": ObjectId(), "name": "workspace"},
            schema_={"_id": ObjectId(), "name": "schema"},
            status="ready",
            public=False,
            created_by=ObjectId(),
        )

    def test_delete_graph_successful(
        self, client, monkeypatch, graph_object_mock
    ):

        graph_id_mock = ObjectId()

        fake_delete = AsyncMock()
        fake_delete.return_value = graph_object_mock
        monkeypatch.setattr(
            "whyhow_api.routers.graphs.delete_graphs", fake_delete
        )

        client.app.dependency_overrides[get_db] = lambda: AsyncMock()
        client.app.dependency_overrides[get_db_client] = lambda: AsyncMock()
        client.app.dependency_overrides[valid_graph_id] = (
            lambda: graph_object_mock
        )
        client.app.dependency_overrides[get_user] = lambda: ObjectId()

        response = client.delete(f"/graphs/{graph_id_mock}")
        assert response.status_code == 200

        data = response.json()
        assert data["message"] == "Graph deleted successfully."
        assert data["status"] == "success"
        assert data["graphs"][0]["name"] == graph_object_mock.name
        assert data["graphs"][0]["created_by"] == str(
            graph_object_mock.created_by
        )


class TestGraphsMergeNodes:

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

    def test_merge_nodes_successful(
        self, client, monkeypatch, graph_object_mock
    ):
        graph_id_mock = ObjectId()
        from_nodes_mock = [ObjectId()]
        to_node_mock = ObjectId()

        fake_merge_nodes = AsyncMock()
        node_id_mock = ObjectId()
        fake_merge_nodes.return_value = NodeWithId(
            _id=node_id_mock,
            name="test node",
            label="test label",
            properties={},
        )
        monkeypatch.setattr(
            "whyhow_api.services.graph_service.merge_nodes", fake_merge_nodes
        )

        client.app.dependency_overrides[valid_graph_id] = (
            lambda: graph_object_mock
        )
        client.app.dependency_overrides[get_db] = lambda: AsyncMock()
        client.app.dependency_overrides[get_user] = lambda: ObjectId()
        response = client.post(
            f"/graphs/{graph_id_mock}/merge_nodes",
            json={
                "from_nodes": [
                    str(from_node) for from_node in from_nodes_mock
                ],
                "to_node": str(to_node_mock),
            },
        )
        assert response.status_code == 200

        data = response.json()
        assert data["message"] == "Nodes merged successfully."
        assert data["status"] == "success"
        assert data["count"] == 1
        assert data["graphs"][0]["name"] == graph_object_mock.name
        assert data["nodes"][0]["_id"] == str(node_id_mock)
        assert data["nodes"][0]["name"] == "test node"
        assert data["nodes"][0]["label"] == "test label"
        assert data["nodes"][0]["properties"] == {}

    def test_merge_nodes_failure(self, client, monkeypatch, graph_object_mock):
        graph_id_mock = ObjectId()
        from_nodes_mock = [ObjectId()]
        to_node_mock = ObjectId()

        fake_merge_nodes = AsyncMock()
        fake_merge_nodes.side_effect = ValueError()
        monkeypatch.setattr(
            "whyhow_api.services.graph_service.merge_nodes", fake_merge_nodes
        )

        client.app.dependency_overrides[valid_graph_id] = (
            lambda: graph_object_mock
        )
        client.app.dependency_overrides[get_db] = lambda: AsyncMock()
        client.app.dependency_overrides[get_user] = lambda: ObjectId()

        response = client.post(
            f"/graphs/{graph_id_mock}/merge_nodes",
            json={
                "from_nodes": [
                    str(from_node) for from_node in from_nodes_mock
                ],
                "to_node": str(to_node_mock),
            },
        )
        assert response.status_code == 400

        data = response.json()
        assert "detail" in data
        assert (
            data["detail"]
            == "Failed to merge nodes. Check that the nodes are existent and have the same type."
        )

    def test_merge_nodes_with_save_as_rule(
        self, client, monkeypatch, graph_object_mock
    ):
        graph_id_mock = ObjectId()
        from_nodes_mock = [ObjectId()]
        to_node_mock = ObjectId()

        fake_merge_nodes = AsyncMock()
        node_id_mock = ObjectId()
        fake_merge_nodes.return_value = NodeWithId(
            _id=node_id_mock,
            name="test node",
            label="test label",
            properties={},
        )
        monkeypatch.setattr(
            "whyhow_api.services.graph_service.merge_nodes", fake_merge_nodes
        )

        fake_create_rule = AsyncMock()
        monkeypatch.setattr(
            "whyhow_api.services.crud.rule.create_one",
            fake_create_rule,
        )

        fake_get_nodes_by_ids = AsyncMock()
        fake_get_nodes_by_ids.side_effect = [
            [
                NodeWithId(
                    _id=ObjectId(),
                    name="test from node",
                    label="test label",
                    properties={},
                )
            ],
            [
                NodeWithId(
                    _id=ObjectId(),
                    name="test to node",
                    label="test label",
                    properties={},
                )
            ],
        ]
        monkeypatch.setattr(
            "whyhow_api.routers.graphs.get_nodes_by_ids",
            fake_get_nodes_by_ids,
        )

        client.app.dependency_overrides[valid_graph_id] = (
            lambda: graph_object_mock
        )
        client.app.dependency_overrides[get_db] = lambda: AsyncMock()
        client.app.dependency_overrides[get_user] = lambda: ObjectId()

        response = client.post(
            f"/graphs/{graph_id_mock}/merge_nodes",
            json={
                "from_nodes": [
                    str(from_node) for from_node in from_nodes_mock
                ],
                "to_node": str(to_node_mock),
                "save_as_rule": True,
            },
        )
        assert response.status_code == 200

        data = response.json()
        assert data["message"] == "Nodes merged successfully."
        assert data["status"] == "success"
        assert data["count"] == 1
        assert data["graphs"][0]["name"] == graph_object_mock.name
        assert data["nodes"][0]["_id"] == str(node_id_mock)
        assert data["nodes"][0]["name"] == "test node"
        assert data["nodes"][0]["label"] == "test label"
        assert data["nodes"][0]["properties"] == {}

        fake_merge_nodes.assert_called_once()
        fake_create_rule.assert_called_once()
        fake_get_nodes_by_ids.assert_called()


class TestGraphsSimilarNodes:

    @pytest.fixture
    def graph_object_mock(self):
        return DetailedGraphDocumentModel(
            _id=ObjectId(),
            name="test graph",
            schema_={"_id": ObjectId(), "name": "schema"},
            status="ready",
            created_by=ObjectId(),
            public=False,
            workspace={"_id": ObjectId(), "name": "workspace"},
        )

    def test_get_similar_nodes_successful(
        self, client, monkeypatch, graph_object_mock
    ):
        similar_nodes_mock = [
            [
                {
                    "_id": "e",
                    "name": "e",
                    "label": "l",
                    "properties": {"k": "v", "k2": "v2"},
                    "chunks": [],
                    "similarity": 0.5,
                }
            ]
        ]

        graph_id_mock = ObjectId()
        fake_get_similar_nodes = AsyncMock()
        fake_get_similar_nodes.return_value = similar_nodes_mock
        monkeypatch.setattr(
            "whyhow_api.services.graph_service.get_similar_nodes",
            fake_get_similar_nodes,
        )

        client.app.dependency_overrides[valid_graph_id] = (
            lambda: graph_object_mock
        )
        client.app.dependency_overrides[get_db] = lambda: AsyncMock()
        client.app.dependency_overrides[get_user] = lambda: ObjectId()
        response = client.get(f"/graphs/{graph_id_mock}/resolve")
        assert response.status_code == 200

        data = response.json()
        assert data["message"] == "Similar nodes retrieved successfully."
        assert data["status"] == "success"
        assert data["count"] == len(similar_nodes_mock)
        assert data["similar_nodes"] == similar_nodes_mock


class TestGraphsNodes:

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
    def nodes_with_id_objects_mock(self):
        user_id_mock = str(ObjectId())
        return [
            NodeWithId(
                _id=user_id_mock,
                name="test node name",
                label="label",
                properties={"test": "test"},
                chunks=[ObjectId(), ObjectId()],
            )
        ]

    def test_graphs_get_nodes_successful(
        self,
        client,
        monkeypatch,
        nodes_with_id_objects_mock,
        graph_object_mock,
    ):
        graph_id_mock = ObjectId()

        # return fake list of node objects
        fake_list_nodes = AsyncMock()
        fake_list_nodes.return_value = (
            nodes_with_id_objects_mock,
            len(nodes_with_id_objects_mock),
        )
        monkeypatch.setattr(
            "whyhow_api.routers.graphs.list_nodes",
            fake_list_nodes,
        )

        client.app.dependency_overrides[get_db] = lambda: AsyncMock()
        client.app.dependency_overrides[get_user] = lambda: ObjectId()
        client.app.dependency_overrides[order_query] = lambda: 1
        client.app.dependency_overrides[valid_graph_id] = (
            lambda: graph_object_mock
        )

        response = client.get(f"/graphs/{graph_id_mock}/nodes")
        assert response.status_code == 200

        data = response.json()
        assert data["message"] == "Graph nodes retrieved successfully."
        assert data["status"] == "success"
        assert data["count"] == len(nodes_with_id_objects_mock)
        node = data["nodes"][0]
        assert node["name"] == nodes_with_id_objects_mock[0].name
        assert node["label"] == nodes_with_id_objects_mock[0].label
        assert node["properties"] == nodes_with_id_objects_mock[0].properties
        assert len(node["chunks"]) == len(nodes_with_id_objects_mock[0].chunks)

    def test_graphs_get_public_nodes_successful(
        self,
        client,
        monkeypatch,
        nodes_with_id_objects_mock,
        graph_object_mock,
    ):
        graph_id_mock = ObjectId()
        # set graph as public
        public_graph = graph_object_mock
        public_graph.public = True

        # return fake list of node objects
        fake_list_nodes = AsyncMock()
        fake_count = 1
        fake_list_nodes.return_value = nodes_with_id_objects_mock, fake_count
        monkeypatch.setattr(
            "whyhow_api.routers.graphs.list_nodes",
            fake_list_nodes,
        )

        client.app.dependency_overrides[get_db] = lambda: AsyncMock()
        client.app.dependency_overrides[order_query] = lambda: 1
        client.app.dependency_overrides[valid_public_graph_id] = (
            lambda: public_graph
        )
        # return fake count of nodes
        fake_get_all_count = AsyncMock()
        fake_get_all_count.return_value = fake_count
        monkeypatch.setattr(
            "whyhow_api.routers.graphs.get_all_count", fake_get_all_count
        )

        response = client.get(f"/graphs/public/{graph_id_mock}/nodes")
        assert response.status_code == 200

        data = response.json()
        assert data["count"] == fake_count
        assert data["message"] == "Graph nodes retrieved successfully."
        assert data["status"] == "success"
        node = data["nodes"][0]
        assert node["name"] == nodes_with_id_objects_mock[0].name
        assert node["label"] == nodes_with_id_objects_mock[0].label
        assert node["properties"] == nodes_with_id_objects_mock[0].properties
        assert data["graphs"][0]["public"] is True


class TestGraphsTriples:

    @pytest.fixture
    def node_with_id_objects_mock(self):
        user_id_mock = str(ObjectId())
        return NodeWithId(
            _id=user_id_mock,
            name="test node with id name",
            label="label",
            properties={"test": "test", "test2": "test2"},
        )

    @pytest.fixture
    def relation_out_mock(self):
        return RelationOut(
            name="example_relation", properties={"key1": "value1", "key2": 2}
        )

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
    def triples_with_id_objects_mock(
        self, node_with_id_objects_mock, relation_out_mock
    ):
        user_id_mock = str(ObjectId())
        return [
            TripleWithId(
                _id=user_id_mock,
                head_node=node_with_id_objects_mock,
                tail_node=node_with_id_objects_mock,
                relation=relation_out_mock,
                chunks=[ObjectId(), ObjectId()],
            )
        ]

    def test_graphs_get_triples_successful(
        self,
        client,
        monkeypatch,
        triples_with_id_objects_mock,
        graph_object_mock,
    ):
        graph_id_mock = ObjectId()
        user_id_mock = ObjectId()

        # return fake list of triples objects
        fake_list_triples = AsyncMock()
        fake_list_triples.return_value = (triples_with_id_objects_mock, 1)
        monkeypatch.setattr(
            "whyhow_api.routers.graphs.list_triples",
            fake_list_triples,
        )

        client.app.dependency_overrides[get_db] = lambda: AsyncMock()
        client.app.dependency_overrides[get_user] = lambda: ObjectId()
        client.app.dependency_overrides[order_query] = lambda: 1
        client.app.dependency_overrides[valid_graph_id] = (
            lambda: graph_object_mock
        )
        # return fake workspace
        workspace_object_mock = WorkspaceDocumentModel(
            name="test workspace", created_by=user_id_mock
        )
        fake_get_one_workspace = AsyncMock()
        fake_get_one_workspace.return_value = (
            workspace_object_mock.model_dump()
        )
        monkeypatch.setattr(
            "whyhow_api.routers.graphs.get_one", fake_get_one_workspace
        )
        # return fake count of triples
        fake_get_all_count = AsyncMock()
        fake_count = 1
        fake_get_all_count.return_value = fake_count
        monkeypatch.setattr(
            "whyhow_api.routers.graphs.get_all_count", fake_get_all_count
        )

        response = client.get(f"/graphs/{graph_id_mock}/triples")
        assert response.status_code == 200

        data = response.json()
        assert "message" in data
        assert data["message"] == "Graph triples retrieved successfully."
        assert "status" in data
        assert data["status"] == "success"
        assert "count" in data
        assert data["count"] == fake_count
        # validate triples data
        assert "triples" in data
        assert len(data["triples"]) == len(triples_with_id_objects_mock)
        for i, triple in enumerate(data["triples"]):
            assert "head_node" in triple
            assert (
                triple["head_node"]["name"]
                == triples_with_id_objects_mock[i].head_node.name
            )
            assert (
                triple["head_node"]["label"]
                == triples_with_id_objects_mock[i].head_node.label
            )
            assert (
                triple["head_node"]["properties"]
                == triples_with_id_objects_mock[i].head_node.properties
            )
            assert "tail_node" in triple
            assert (
                triple["tail_node"]["name"]
                == triples_with_id_objects_mock[i].tail_node.name
            )
            assert (
                triple["tail_node"]["label"]
                == triples_with_id_objects_mock[i].tail_node.label
            )
            assert (
                triple["tail_node"]["properties"]
                == triples_with_id_objects_mock[i].tail_node.properties
            )
            assert "relation" in triple
            assert (
                triple["relation"]["name"]
                == triples_with_id_objects_mock[i].relation.name
            )
            assert (
                triple["relation"]["properties"]
                == triples_with_id_objects_mock[i].relation.properties
            )
            assert "chunks" in triple
            assert len(triple["chunks"]) == len(
                triples_with_id_objects_mock[i].chunks
            )

    def test_graphs_get_public_triples_successful(
        self,
        client,
        monkeypatch,
        triples_with_id_objects_mock,
        graph_object_mock,
    ):
        graph_id_mock = ObjectId()
        # set graph as public
        fake_public_graph = graph_object_mock
        fake_public_graph.public = True
        # return fake list of triples objects
        fake_public_list_triples = AsyncMock()
        fake_public_list_triples.return_value = (
            triples_with_id_objects_mock,
            1,
        )
        monkeypatch.setattr(
            "whyhow_api.routers.graphs.list_triples",
            fake_public_list_triples,
        )

        client.app.dependency_overrides[get_db] = lambda: AsyncMock()
        client.app.dependency_overrides[order_query] = lambda: 1
        client.app.dependency_overrides[valid_public_graph_id] = (
            lambda: fake_public_graph
        )
        # return fake count of triples
        fake_get_all_count = AsyncMock()
        fake_count = 1
        fake_get_all_count.return_value = fake_count
        monkeypatch.setattr(
            "whyhow_api.routers.graphs.get_all_count", fake_get_all_count
        )

        response = client.get(f"/graphs/public/{graph_id_mock}/triples")
        assert response.status_code == 200

        data = response.json()
        assert "message" in data
        assert data["message"] == "Graph triples retrieved successfully."
        assert "status" in data
        assert data["status"] == "success"
        assert "count" in data
        assert data["count"] == fake_count
        # validate triples data
        assert "triples" in data
        assert len(data["triples"]) == len(triples_with_id_objects_mock)
        for i, triple in enumerate(data["triples"]):
            assert "head_node" in triple
            assert (
                triple["head_node"]["name"]
                == triples_with_id_objects_mock[i].head_node.name
            )
            assert (
                triple["head_node"]["label"]
                == triples_with_id_objects_mock[i].head_node.label
            )
            assert (
                triple["head_node"]["properties"]
                == triples_with_id_objects_mock[i].head_node.properties
            )
            assert "tail_node" in triple
            assert (
                triple["tail_node"]["name"]
                == triples_with_id_objects_mock[i].tail_node.name
            )
            assert (
                triple["tail_node"]["label"]
                == triples_with_id_objects_mock[i].tail_node.label
            )
            assert (
                triple["tail_node"]["properties"]
                == triples_with_id_objects_mock[i].tail_node.properties
            )
            assert "relation" in triple
            assert (
                triple["relation"]["name"]
                == triples_with_id_objects_mock[i].relation.name
            )
            assert (
                triple["relation"]["properties"]
                == triples_with_id_objects_mock[i].relation.properties
            )
            assert "chunks" in triple
            assert len(triple["chunks"]) == len(
                triples_with_id_objects_mock[i].chunks
            )
            assert data["graphs"][0]["public"] is True


class TestGraphsRelations:

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
    def relations_object_mock(self):
        relations = ["relation1", "relation2", "relation3"]
        total_count = len(relations)
        return (relations, total_count)

    def test_graphs_get_relations_workspace_not_found(
        self,
        client,
        monkeypatch,
        relations_object_mock,
        graph_object_mock,
    ):
        graph_id_mock = ObjectId()

        # return fake list of node relations
        fake_list_relations = AsyncMock()
        fake_list_relations.return_value = relations_object_mock
        monkeypatch.setattr(
            "whyhow_api.routers.graphs.list_relations",
            fake_list_relations,
        )

        client.app.dependency_overrides[get_db] = lambda: AsyncMock()
        client.app.dependency_overrides[get_user] = lambda: ObjectId()
        client.app.dependency_overrides[order_query] = lambda: 1
        client.app.dependency_overrides[valid_graph_id] = (
            lambda: graph_object_mock
        )
        # return None as Not found for Workspace
        fake_find_workspace = AsyncMock()
        fake_find_workspace.return_value = None
        monkeypatch.setattr(
            "whyhow_api.routers.graphs.get_one", fake_find_workspace
        )

        response = client.get(f"/graphs/{graph_id_mock}/relations")
        assert response.status_code == 404

        data = response.json()
        assert "detail" in data
        assert data["detail"] == "Workspace not found."

    def test_graphs_get_relations_successful(
        self,
        client,
        monkeypatch,
        relations_object_mock,
        graph_object_mock,
    ):
        graph_id_mock = ObjectId()
        user_id_mock = ObjectId()

        # return fake list of node relations
        fake_list_relations = AsyncMock()
        fake_list_relations.return_value = relations_object_mock
        monkeypatch.setattr(
            "whyhow_api.routers.graphs.list_relations",
            fake_list_relations,
        )

        client.app.dependency_overrides[get_db] = lambda: AsyncMock()
        client.app.dependency_overrides[get_user] = lambda: ObjectId()
        client.app.dependency_overrides[order_query] = lambda: 1
        client.app.dependency_overrides[valid_graph_id] = (
            lambda: graph_object_mock
        )
        # return fake workspace as Not found for Workspace
        workspace_object_mock = WorkspaceDocumentModel(
            name="test workspace", created_by=user_id_mock
        )
        fake_get_one_workspace = AsyncMock()
        fake_get_one_workspace.return_value = (
            workspace_object_mock.model_dump()
        )
        monkeypatch.setattr(
            "whyhow_api.routers.graphs.get_one", fake_get_one_workspace
        )

        response = client.get(f"/graphs/{graph_id_mock}/relations")
        assert response.status_code == 200

        data = response.json()
        assert data["message"] == "Graph relations retrieved successfully."
        assert data["status"] == "success"
        assert data["count"] == 3
        graph = data["graphs"][0]
        assert graph["name"] == graph_object_mock.name
        assert graph["created_by"] == str(graph_object_mock.created_by)
        assert data["relations"] == relations_object_mock[0]

    def test_graphs_get_public_relations_successful(
        self,
        client,
        monkeypatch,
        relations_object_mock,
        graph_object_mock,
    ):
        graph_id_mock = ObjectId()
        # get public graph object
        public_graph_object_mock = graph_object_mock
        public_graph_object_mock.public = True

        # return fake list of node relations
        fake_list_relations = AsyncMock()
        fake_list_relations.return_value = relations_object_mock
        monkeypatch.setattr(
            "whyhow_api.routers.graphs.list_relations",
            fake_list_relations,
        )

        client.app.dependency_overrides[get_db] = lambda: AsyncMock()
        client.app.dependency_overrides[order_query] = lambda: 1
        client.app.dependency_overrides[valid_public_graph_id] = (
            lambda: public_graph_object_mock
        )

        response = client.get(f"/graphs/public/{graph_id_mock}/relations")
        assert response.status_code == 200

        data = response.json()
        assert data["message"] == "Graph relations retrieved successfully."
        assert data["status"] == "success"
        assert data["count"] == 3
        graph = data["graphs"][0]
        assert graph["name"] == graph_object_mock.name
        assert graph["created_by"] == str(graph_object_mock.created_by)
        assert data["relations"] == relations_object_mock[0]
        assert data["graphs"][0]["public"] is True

    def test_graphs_export_graph_as_cypher_statements_successful(
        self, client, monkeypatch, graph_object_mock
    ):
        graph_id_mock = ObjectId()
        fake_export_graph = AsyncMock()
        fake_export_graph.return_value = [
            "exported_graph1",
            "exported_graph2",
            "exported_graph3",
        ]
        monkeypatch.setattr(
            "whyhow_api.routers.graphs.graph_service.export_graph_to_cypher",
            fake_export_graph,
        )

        client.app.dependency_overrides[valid_graph_id] = (
            lambda: graph_object_mock
        )
        client.app.dependency_overrides[get_db] = lambda: AsyncMock()
        client.app.dependency_overrides[get_user] = lambda: ObjectId()

        response = client.get(f"/graphs/{graph_id_mock}/export/cypher")
        assert response.status_code == 200

        data = response.json()
        assert data["message"] == "Cypher text successfully generated."

    def test_graphs_export_graph_as_cypher_statements_exception(
        self, client, monkeypatch, graph_object_mock
    ):
        graph_id_mock = ObjectId()
        fake_export_graph = AsyncMock()
        fake_export_graph.side_effect = Exception("Test Exception")
        monkeypatch.setattr(
            "whyhow_api.routers.graphs.graph_service.export_graph_to_cypher",
            fake_export_graph,
        )

        client.app.dependency_overrides[valid_graph_id] = (
            lambda: graph_object_mock
        )
        client.app.dependency_overrides[get_db] = lambda: AsyncMock()
        client.app.dependency_overrides[get_user] = lambda: ObjectId()

        response = client.get(f"/graphs/{graph_id_mock}/export/cypher")
        data = response.json()
        assert response.status_code == 500
        assert (
            data["detail"]
            == "Failed to export graph to Cypher: Test Exception"
        )


class TestGraphRules:

    @pytest.fixture
    def rule_document_mock(self):
        return RuleOut(
            _id=ObjectId(),
            created_by=ObjectId(),
            workspace_id=ObjectId(),
            rule=MergeNodesRule(
                rule_type="merge_nodes",
                from_node_names=["test from node"],
                to_node_name="test to node",
                node_type="test node type",
            ),
        )

    def test_read_graph_rules_successful(
        self, client, rule_document_mock, monkeypatch
    ):

        fake_get_rules = AsyncMock()
        fake_get_rules.return_value = ([rule_document_mock], 1)
        monkeypatch.setattr(
            "whyhow_api.routers.graphs.get_graph_rules", fake_get_rules
        )

        fake_graph = MagicMock()
        fake_graph.id = ObjectId()

        client.app.dependency_overrides[get_db] = lambda: AsyncMock()
        client.app.dependency_overrides[get_user] = lambda: ObjectId()
        client.app.dependency_overrides[valid_graph_id] = lambda: fake_graph

        response = client.get(f"/graphs/{ObjectId()}/rules")
        assert response.status_code == 200

        data = response.json()
        assert data["message"] == "Rules retrieved successfully."
        assert data["status"] == "success"
        assert data["count"] == 1
        assert (
            data["rules"][0]["rule"]["rule_type"]
            == rule_document_mock.rule.rule_type
        )

    def test_read_public_graph_rules_successful(
        self, client, rule_document_mock, monkeypatch
    ):

        fake_get_rules = AsyncMock()
        fake_get_rules.return_value = ([rule_document_mock], 1)
        monkeypatch.setattr(
            "whyhow_api.routers.graphs.get_graph_rules", fake_get_rules
        )

        fake_graph = MagicMock()
        fake_graph.id = ObjectId()

        client.app.dependency_overrides[get_db] = lambda: AsyncMock()
        client.app.dependency_overrides[valid_public_graph_id] = (
            lambda: fake_graph
        )

        response = client.get(f"/graphs/public/{ObjectId()}/rules")
        assert response.status_code == 200

        data = response.json()
        assert data["message"] == "Rules retrieved successfully."
        assert data["status"] == "success"
        assert data["count"] == 1
        assert (
            data["rules"][0]["rule"]["rule_type"]
            == rule_document_mock.rule.rule_type
        )
