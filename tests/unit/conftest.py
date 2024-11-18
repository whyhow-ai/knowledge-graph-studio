import os
from unittest.mock import AsyncMock

import pytest
from bson import ObjectId
from fastapi.testclient import TestClient

from whyhow_api.config import Settings
from whyhow_api.dependencies import get_settings
from whyhow_api.main import app
from whyhow_api.models.common import Triple
from whyhow_api.schemas.graphs import GraphDocumentModel
from whyhow_api.schemas.nodes import NodeDocumentModel
from whyhow_api.schemas.schemas import (
    SchemaDocumentModel,
    SchemaEntity,
    SchemaRelation,
    SchemaTriplePattern,
)
from whyhow_api.schemas.triples import TripleDocumentModel
from whyhow_api.schemas.workspaces import WorkspaceDocumentModel


@pytest.fixture()
def client(monkeypatch):
    """Get client and clear app dependency_overrides."""

    # make sure that before each test the settings is reinstantiated
    # by default the get_settings is cached globally and defined
    # at the startup of the application. This way the tests will be
    # able to dynamically change the settings
    app.dependency_overrides[get_settings] = lambda: Settings()
    client = TestClient(app, raise_server_exceptions=False)

    # Remove RateLimiter middleware due to inability to override `get_settings`
    for middleware in app.user_middleware:
        if middleware.cls.__name__ == "RateLimiter":
            app.user_middleware.remove(middleware)
    yield client

    app.dependency_overrides.clear()


@pytest.fixture(autouse=True)
def disable_aws(monkeypatch):
    """Disable AWS calls."""
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.delenv("AWS_PROFILE", raising=False)


@pytest.fixture(autouse=True, scope="session")
def dont_look_at_env_file():
    """Never look inside of the .env when running unit tests."""
    Settings.model_config["env_file"] = None


@pytest.fixture(autouse=True)
def disable_whyhow(monkeypatch):
    """Disable whyhow env vars"""

    for k, v in os.environ.items():
        if k.startswith("WHYHOW"):
            monkeypatch.delenv(k, raising=False)


class BaseTest:
    workspace_id_mock = ObjectId()
    user_id_mock = ObjectId()
    schema_id_mock = ObjectId()
    graph_id_mock = ObjectId()
    node_id_mock = ObjectId()
    graph_name_mock = "TestGraph"
    schema_name_mock = "TestSchema"

    workspace_db_object_mock = WorkspaceDocumentModel(
        _id=workspace_id_mock, created_by=user_id_mock, name="test_workspace"
    )

    graph_db_object_mock = GraphDocumentModel(
        _id=graph_id_mock,
        created_by=user_id_mock,
        name=graph_name_mock,
        schema=schema_id_mock,
        workspace=workspace_id_mock,
        status="creating",
        type="txt",
    ).model_dump(by_alias=False)

    schema_entities = [
        SchemaEntity(name="character", description=""),
        SchemaEntity(name="object", description=""),
    ]
    schema_relations = [SchemaRelation(name="wears", description="")]
    schema_patterns = [
        SchemaTriplePattern(
            head=schema_entities[0],
            relation=schema_relations[0],
            tail=schema_entities[1],
            description="",
        )
    ]

    schema_db_object_mock = SchemaDocumentModel(
        _id=schema_id_mock,
        created_by=user_id_mock,
        name=schema_name_mock,
        workspace=workspace_id_mock,
        entities=schema_entities,
        relations=schema_relations,
        patterns=schema_patterns,
        type="txt",
    ).model_dump(by_alias=False)

    node_db_object_mock = NodeDocumentModel(
        name="test_node",
        type="test_type",
        created_by=user_id_mock,
        graph=graph_id_mock,
        properties={"key": "value"},
    )

    head_node_id_mock = ObjectId()
    tail_node_id_mock = ObjectId()

    triple_db_object_mock = TripleDocumentModel(
        name="test_triple",
        type="test_type",
        head_node=head_node_id_mock,
        tail_node=tail_node_id_mock,
        created_by=user_id_mock,
        graph=graph_id_mock,
        properties={"key": "value"},
        embedding=[0.1, 0.5],
    )

    triple_create_object_mock = Triple(
        head="test_head", relation="test_relation", tail="test_tail"
    )


@pytest.fixture(scope="class")
def db_mock():
    db = AsyncMock()
    db.get_collection = AsyncMock()
    db.get_collection.return_value.find_one = AsyncMock()
    return db


@pytest.fixture(scope="class")
def db_session_mock():
    return AsyncMock()


@pytest.fixture(scope="class")
def workspace_id_mock():
    return BaseTest.workspace_id_mock


@pytest.fixture(scope="class")
def user_id_mock():
    return BaseTest.user_id_mock


@pytest.fixture(scope="class")
def schema_id_mock():
    return BaseTest.schema_id_mock


@pytest.fixture(scope="class")
def graph_id_mock():
    return BaseTest.graph_id_mock


@pytest.fixture(scope="class")
def node_id_mock():
    return BaseTest.node_id_mock


@pytest.fixture(scope="class")
def graph_name_mock():
    return BaseTest.graph_name_mock


@pytest.fixture(scope="class")
def graph_db_object_mock():
    return BaseTest.graph_db_object_mock


@pytest.fixture(scope="class")
def schema_db_object_mock():
    return BaseTest.schema_db_object_mock


@pytest.fixture(scope="class")
def workspace_db_object_mock():
    return BaseTest.workspace_db_object_mock


@pytest.fixture(scope="class")
def schema_name_mock():
    return BaseTest.schema_name_mock


@pytest.fixture(scope="class")
def api_key_header_mock():
    return "valid_api_key"


@pytest.fixture(scope="class")
def openai_api_key_header_mock():
    return "valid_openai_api_key"


@pytest.fixture(scope="class")
def node_db_object_mock():
    return BaseTest.node_db_object_mock


@pytest.fixture(scope="class")
def triple_db_object_mock():
    return BaseTest.triple_db_object_mock


@pytest.fixture(scope="class")
def triple_create_object_mock():
    return BaseTest.triple_create_object_mock
