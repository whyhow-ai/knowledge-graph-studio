from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from auth0.management import Auth0
from bson import ObjectId
from fastapi import HTTPException, Security, status
from motor.motor_asyncio import AsyncIOMotorDatabase
from openai import AsyncAzureOpenAI, AsyncOpenAI

from whyhow_api.dependencies import (
    api_key_header,
    get_auth0,
    get_db,
    get_db_client,
    get_llm_client,
    get_user,
    valid_create_graph,
    valid_document_id,
    valid_graph_id,
    valid_node_id,
    valid_workspace_id,
)
from whyhow_api.main import app
from whyhow_api.schemas.documents import (
    DocumentMetadata,
    DocumentOutWithWorkspaceDetails,
)
from whyhow_api.schemas.graphs import (
    CreateGraphBody,
    DetailedGraphDocumentModel,
    GraphDocumentModel,
)
from whyhow_api.schemas.nodes import NodeDocumentModel
from whyhow_api.schemas.schemas import SchemaDocumentModel
from whyhow_api.schemas.workspaces import WorkspaceDocumentModel


@pytest.fixture
def mock_client():
    """Fixture to mock AsyncIOMotorClient."""
    client = MagicMock()
    database = MagicMock(spec=AsyncIOMotorDatabase)
    database.name = "test_db"
    client.get_default_database.return_value = database
    return client


@pytest.fixture
def mock_settings():
    """Fixture to mock settings access."""

    class MockSettings:
        mongodb = MagicMock()

    return MockSettings()


def test_get_settings_placeholder():
    pass


@pytest.mark.asyncio
@patch("whyhow_api.dependencies.get_client")
@patch("whyhow_api.dependencies.get_settings")
async def test_get_db_success(mock_settings, mock_get_client, mock_client):
    """Test successful database connection."""
    # Setting up the mock return values
    mock_settings.mongodb.database_name = "test_db"
    mock_get_client.return_value = mock_client

    async for db in get_db(mock_settings):
        assert db.name == "test_db"


@pytest.mark.asyncio
class TestDatabaseConnection:
    @patch("whyhow_api.dependencies.get_client")
    @patch("whyhow_api.dependencies.get_settings")
    async def test_get_db_success(
        self, mock_settings, mock_get_client, mock_client
    ):
        """Test successful database connection."""
        mock_settings.mongodb.database_name = "test_db"
        mock_get_client.return_value = mock_client

        async for db in get_db(mock_settings):
            assert db.name == "test_db"

    @patch("whyhow_api.dependencies.get_client")
    @patch("whyhow_api.dependencies.get_settings")
    async def test_get_db_client_failure(self, mock_settings, mock_get_client):
        """Test failure due to get_client returning None."""
        mock_settings.mongodb.database_name = "test_db"
        mock_get_client.return_value = None  # Simulate failure to get a client

        with pytest.raises(ConnectionError) as excinfo:
            async for db in get_db():
                pass
        assert "Failed to retrieve MongoDB client" in str(excinfo.value)

    @pytest.mark.skip("Test not implemented")
    async def test_get_db_settings_error(self):
        """Test failure due to misconfigured settings."""
        pass


@pytest.mark.asyncio
class TestGetDbClient:
    @patch("whyhow_api.dependencies.get_client")
    async def test_get_db_client_success(self, mock_get_client, mock_client):
        """Test successful retrieval of MongoDB client."""
        mock_get_client.return_value = mock_client

        async for client in get_db_client():
            assert client is mock_client

    @patch("whyhow_api.dependencies.get_client")
    async def test_get_db_client_failure(self, mock_get_client):
        """Test failure to retrieve MongoDB client."""
        mock_get_client.return_value = None

        with pytest.raises(ConnectionError) as excinfo:
            async for client in get_db_client():
                pass
        assert "Failed to retrieve MongoDB client" in str(excinfo.value)

    @patch("whyhow_api.dependencies.get_client")
    async def test_client_release(self, mock_get_client, mock_client):
        """Test that the MongoDB client is properly released."""
        mock_get_client.return_value = mock_client

        async for client in get_db_client():
            pass


@app.get("/api-key-required")
async def api_key_required(api_key: str = Security(api_key_header)):
    if api_key is None:
        raise HTTPException(status_code=403, detail="API Key Required")
    return {"message": "Success", "api_key": api_key}


@pytest.mark.skip("Not working. Need to handle environment variables.")
def test_api_key_required_success(client):
    response = client.get(
        "/api-key-required", headers={"x-api-key": "testapikey123"}
    )
    assert response.status_code == 200
    assert response.json() == {
        "message": "Success",
        "api_key": "testapikey123",
    }


@pytest.mark.skip("Not working. Need to handle environment variables.")
def test_api_key_required_failure(client):
    response = client.get("/api-key-required")
    assert response.status_code == 403
    assert response.json() == {"detail": "API Key Required"}


@pytest.mark.skip("Not working. Need to handle environment variables.")
def test_openai_key_required_success(client):
    response = client.get(
        "/openai-key-required", headers={"x-openai-key": "openaikey123"}
    )
    assert response.status_code == 200
    assert response.json() == {
        "message": "Success",
        "openai_key": "openaikey123",
    }


@pytest.mark.skip("Not working. Need to handle environment variables.")
def test_openai_key_required_failure(client):
    response = client.get("/openai-key-required")
    assert response.status_code == 403
    assert response.json() == {"detail": "OpenAI API Key Required"}


@pytest.mark.asyncio
class TestGetUser:
    async def test_valid_api_key(
        self, db_mock, api_key_header_mock, user_id_mock
    ):
        # Setup the mock to return a specific user document
        db_mock.user.find_one.return_value = {"_id": user_id_mock}

        mock_request = MagicMock()
        mock_request.app.state.jwks_client = MagicMock()

        # Call the function with a valid API key
        user_id = await get_user(
            request=mock_request, api_key=api_key_header_mock, db=db_mock
        )

        # Check that the user ID is returned correctly
        assert user_id == user_id_mock

        # Ensure that find_one was called with the correct filter
        db_mock.user.find_one.assert_awaited_once_with(
            {"api_key": api_key_header_mock}
        )

    async def test_missing_api_key(self, db_mock):
        # Attempt to call the function without providing an API key
        mock_request = MagicMock()
        mock_request.app.state.jwks_client = MagicMock()

        with pytest.raises(HTTPException) as exc_info:
            await get_user(
                request=mock_request, api_key=None, token=None, db=db_mock
            )

        # Check that the appropriate HTTPException is raised
        assert exc_info.value.status_code == 401
        assert exc_info.value.detail == "API key or token is required"

    async def test_invalid_api_key(self, db_mock):
        db_mock.user.find_one.return_value = None  # Simulate invalid API key
        mock_request = MagicMock()
        mock_request.app.state.jwks_client = MagicMock()

        # Call the function with an invalid API key
        with pytest.raises(HTTPException) as exc_info:
            await get_user(
                request=mock_request, api_key="invalid_api_key", db=db_mock
            )

        # Check that the appropriate HTTPException is raised
        assert exc_info.value.status_code == 401
        assert exc_info.value.detail == "Invalid API key"


@pytest.mark.asyncio
class TestValidWorkspaceId:
    @patch("whyhow_api.dependencies.get_one", new_callable=AsyncMock)
    async def test_valid_workspace_id(self, get_one_mock):
        user_id_mock = ObjectId()
        db_mock = AsyncMock()
        workspace_id_mock = ObjectId()

        # Setup the mock to return a specific workspace document
        get_one_mock.return_value = {
            "_id": workspace_id_mock,
            "created_by": user_id_mock,
            "name": "test_workspace",
        }

        # Setup the mock for the collection
        workspace_collection_mock = AsyncMock()
        db_mock.__getitem__.return_value = workspace_collection_mock

        # Call the function with a valid workspace ID
        workspace = await valid_workspace_id(
            workspace_id=str(workspace_id_mock),
            user_id=user_id_mock,
            db=db_mock,
        )

        # Check that the workspace is returned correctly
        assert isinstance(workspace, WorkspaceDocumentModel)

        # Ensure that get_one was called with the correct arguments
        get_one_mock.assert_awaited_once_with(
            collection=workspace_collection_mock,
            document_model=WorkspaceDocumentModel,
            id=ObjectId(workspace_id_mock),
            user_id=user_id_mock,
        )

    @patch("whyhow_api.dependencies.get_one", new_callable=AsyncMock)
    async def test_invalid_workspace_id(self, get_one_mock):
        user_id_mock = ObjectId()
        get_one_mock.return_value = None  # Simulate invalid workspace

        # Call the function with an invalid workspace_id
        with pytest.raises(HTTPException) as exc_info:
            await valid_workspace_id(
                workspace_id="invalid_workspace_id", user_id=user_id_mock
            )

        # Check that the appropriate HTTPException is raised
        assert exc_info.value.status_code == 404
        assert exc_info.value.detail == "Workspace not found"

    @patch("whyhow_api.dependencies.get_user", autospec=True)
    @patch("whyhow_api.dependencies.get_db", autospec=True)
    async def test_missing_user_id(
        self,
        get_db_mock,
        get_user_mock,
    ):
        db_mock = AsyncMock()
        user_id_mock = ObjectId()
        workspace_id_mock = ObjectId()
        # Setup the mocks
        get_user_mock.return_value = user_id_mock
        get_db_mock.return_value = db_mock
        # Attempt to call the function without providing a user ID
        with pytest.raises(HTTPException) as exc_info:
            await valid_workspace_id(workspace_id=workspace_id_mock)

        assert exc_info.value.status_code == status.HTTP_404_NOT_FOUND
        assert "Workspace not found" in exc_info.value.detail

    async def test_missing_workspace_id(self):
        user_id_mock = ObjectId()
        db_mock = AsyncMock()
        # Attempt to call the function without providing a workspace_id
        with pytest.raises(TypeError) as exc_info:
            await valid_workspace_id(user_id=user_id_mock, db=db_mock)

        # Check that the appropriate TypeError is raised
        assert "missing 1 required positional argument" in str(exc_info.value)


@pytest.mark.asyncio
class TestValidNodeId:
    @patch("whyhow_api.dependencies.get_one", new_callable=AsyncMock)
    async def test_valid_node_id(self, get_one_mock):
        user_id_mock = ObjectId()
        db_mock = AsyncMock()
        node_id_mock = ObjectId()
        graph_id_mock = ObjectId()

        # Setup the mock to return a specific node document
        get_one_mock.return_value = {
            "_id": node_id_mock,
            "created_by": user_id_mock,
            "name": "test_node",
            "graph": graph_id_mock,
        }

        # Setup the mock for the collection
        node_collection_mock = AsyncMock()
        db_mock.__getitem__.return_value = node_collection_mock

        # Call the function with a valid node ID
        node = await valid_node_id(
            node_id=str(node_id_mock), user_id=user_id_mock, db=db_mock
        )

        # Check that the node is returned correctly
        assert isinstance(node, NodeDocumentModel)

        # Ensure that get_one was called with the correct arguments
        get_one_mock.assert_awaited_once_with(
            collection=node_collection_mock,
            document_model=NodeDocumentModel,
            id=ObjectId(node_id_mock),
            user_id=user_id_mock,
        )

    @patch("whyhow_api.dependencies.get_one", new_callable=AsyncMock)
    async def test_invalid_node_id(self, get_one_mock):
        user_id_mock = ObjectId()
        get_one_mock.return_value = None  # Simulate invalid node

        # Call the function with an invalid node ID
        with pytest.raises(HTTPException) as exc_info:
            await valid_node_id(
                node_id="invalid_node_id", user_id=user_id_mock
            )

        # Check that the appropriate HTTPException is raised
        assert exc_info.value.status_code == 404
        assert exc_info.value.detail == "Node not found"

    @patch("whyhow_api.dependencies.get_user", autospec=True)
    @patch("whyhow_api.dependencies.get_db", autospec=True)
    async def test_missing_user_id(
        self,
        get_db_mock,
        get_user_mock,
    ):
        db_mock = AsyncMock()
        user_id_mock = ObjectId()
        node_id_mock = ObjectId()
        # Setup the mocks
        get_user_mock.return_value = user_id_mock
        get_db_mock.return_value = db_mock
        # Attempt to call the function without providing a user ID
        with pytest.raises(HTTPException) as exc_info:
            await valid_node_id(node_id=str(node_id_mock))

        assert exc_info.value.status_code == status.HTTP_404_NOT_FOUND
        assert "Node not found" in exc_info.value.detail

    async def test_missing_node_id(self):
        user_id_mock = ObjectId()
        db_mock = AsyncMock()
        # Attempt to call the function without providing a node ID
        with pytest.raises(TypeError) as exc_info:
            await valid_node_id(user_id=user_id_mock, db=db_mock)

        # Check that the appropriate TypeError is raised
        assert "missing 1 required positional argument" in str(exc_info.value)


@pytest.mark.asyncio
class TestValidGraphId:

    @patch("whyhow_api.dependencies.get_graph", new_callable=AsyncMock)
    async def test_valid_graph_id(self, get_graph_mock):
        user_id_mock = ObjectId()
        db_mock = AsyncMock()
        graph_id_mock = ObjectId()
        workspace_id_mock = ObjectId()
        # Setup the mock to return a specific graph document
        get_graph_mock.return_value = DetailedGraphDocumentModel(
            _id=graph_id_mock,
            created_by=user_id_mock,
            name="test_graph",
            status="ready",
            workspace={"_id": workspace_id_mock, "name": "test_workspace"},
            schema_={"_id": ObjectId(), "name": "test_schema"},
            public=False,
        ).model_dump(by_alias=True)

        graph_collection_mock = AsyncMock()
        db_mock.__getitem__.return_value = graph_collection_mock

        # Call the function with a valid graph ID
        graph = await valid_graph_id(
            graph_id=str(graph_id_mock), user_id=user_id_mock, db=db_mock
        )

        # Check that the graph is returned correctly
        assert isinstance(graph, DetailedGraphDocumentModel)

        # Ensure that get was called with the correct arguments
        get_graph_mock.assert_awaited_once_with(
            collection=graph_collection_mock,
            graph_id=ObjectId(graph_id_mock),
            user_id=user_id_mock,
        )

    @patch("whyhow_api.dependencies.get_one", new_callable=AsyncMock)
    async def test_invalid_graph_id(self, get_one_mock):
        user_id_mock = ObjectId()
        get_one_mock.get.return_value = None  # Simulate invalid graph

        # Call the function with an invalid graph ID
        with pytest.raises(HTTPException) as exc_info:
            await valid_graph_id(
                graph_id="invalid_graph_id", user_id=user_id_mock
            )

        # Check that the appropriate HTTPException is raised
        assert exc_info.value.status_code == 404
        assert exc_info.value.detail == "Graph not found"

    @patch("whyhow_api.dependencies.get_user", autospec=True)
    @patch("whyhow_api.dependencies.get_db", autospec=True)
    async def test_missing_user_id(
        self,
        get_db_mock,
        get_user_mock,
    ):
        db_mock = AsyncMock()
        user_id_mock = ObjectId()
        graph_id_mock = ObjectId()

        # Setup the mocks
        get_user_mock.return_value = user_id_mock
        get_db_mock.return_value = db_mock
        # Attempt to call the function without providing a user ID
        with pytest.raises(HTTPException) as exc_info:
            await valid_graph_id(graph_id=graph_id_mock)

        assert exc_info.value.status_code == status.HTTP_404_NOT_FOUND
        assert "Graph not found" in exc_info.value.detail

    async def test_missing_graph_id(self):
        user_id_mock = ObjectId()
        db_mock = AsyncMock()
        # Attempt to call the function without providing a graph ID
        with pytest.raises(TypeError) as exc_info:
            await valid_graph_id(user_id=user_id_mock, db=db_mock)

        # Check that the appropriate TypeError is raised
        assert "missing 1 required positional argument" in str(exc_info.value)


@pytest.mark.asyncio
class TestValidDocumentId:

    @patch("whyhow_api.dependencies.get_document", new_callable=AsyncMock)
    async def test_valid_document_id(self, get_document_mock):
        user_id_mock = ObjectId()
        db_mock = AsyncMock()
        document_id_mock = ObjectId()

        # Setup the mock to return a specific document document
        workspace_id = ObjectId()
        mock_document = DocumentOutWithWorkspaceDetails(
            _id=document_id_mock,
            created_by=user_id_mock,
            workspaces=[{"_id": workspace_id, "name": "test workspace"}],
            status="processed",
            metadata=DocumentMetadata(
                size=1234, format="txt", filename="test_file.txt"
            ),
            tags={str(workspace_id): ["tag1", "tag2"]},
        )

        get_document_mock.return_value = mock_document.model_dump(
            by_alias=True, exclude_unset=True
        )

        # Setup the mock for the collection
        document_collection_mock = AsyncMock()
        db_mock.__getitem__.return_value = document_collection_mock

        # Call the function with a valid document ID
        document = await valid_document_id(
            document_id=str(document_id_mock), user_id=user_id_mock, db=db_mock
        )

        # Check that the document is returned correctly
        assert isinstance(document, DocumentOutWithWorkspaceDetails)

        # Ensure that get_one was called with the correct arguments
        get_document_mock.assert_awaited_once_with(
            collection=document_collection_mock,
            id=ObjectId(document_id_mock),
            user_id=user_id_mock,
        )

    @patch("whyhow_api.dependencies.get_one", new_callable=AsyncMock)
    async def test_invalid_document_id(self, get_one_mock):
        user_id_mock = ObjectId()
        get_one_mock.return_value = None  # Simulate invalid document

        # Call the function with an invalid document ID
        with pytest.raises(HTTPException) as exc_info:
            await valid_document_id(
                document_id="invalid_document_id", user_id=user_id_mock
            )

        # Check that the appropriate HTTPException is raised
        assert exc_info.value.status_code == 404
        assert exc_info.value.detail == "Document not found"

    @patch("whyhow_api.dependencies.get_user", autospec=True)
    @patch("whyhow_api.dependencies.get_db", autospec=True)
    async def test_missing_user_id(
        self,
        get_db_mock,
        get_user_mock,
    ):
        db_mock = AsyncMock()
        user_id_mock = ObjectId()
        document_id_mock = ObjectId()
        # Setup the mocks
        get_user_mock.return_value = user_id_mock
        get_db_mock.return_value = db_mock
        # Attempt to call the function without providing a user ID
        with pytest.raises(HTTPException) as exc_info:
            await valid_document_id(document_id=str(document_id_mock))

        assert exc_info.value.status_code == status.HTTP_404_NOT_FOUND
        assert "Document not found" in exc_info.value.detail

    async def test_missing_document_id(self):
        user_id_mock = ObjectId()
        db_mock = AsyncMock()
        # Attempt to call the function without providing a document ID
        with pytest.raises(TypeError) as exc_info:
            await valid_document_id(user_id=user_id_mock, db=db_mock)

        # Check that the appropriate TypeError is raised
        assert "missing 1 required positional argument" in str(exc_info.value)


@pytest.mark.asyncio
class TestValidCreateGraph:
    @patch("whyhow_api.dependencies.get_one", new_callable=AsyncMock)
    async def test_valid_create_graph(
        self,
        get_one_mock,
    ):
        user_id_mock = ObjectId()
        db_mock = AsyncMock()
        workspace_id_mock = ObjectId()
        schema_id_mock = ObjectId()

        body = CreateGraphBody(
            name="test_graph",
            workspace=str(workspace_id_mock),
            schema=str(schema_id_mock),
            filters={"ids": [ObjectId(), ObjectId()]},
        )

        # Mock the get_one function for each collection call
        get_one_mock.side_effect = [
            {"_id": workspace_id_mock, "created_by": user_id_mock},
            {"_id": schema_id_mock, "created_by": user_id_mock},
            None,  # For the graph check, indicating that the graph does not exist
        ]

        # Call the function with a valid body
        result = await valid_create_graph(
            body=body, user_id=user_id_mock, db=db_mock
        )

        # Check that the result is True
        assert result is True

        # Ensure that get_one was called with the correct arguments for workspace
        get_one_mock.assert_any_call(
            collection=db_mock["workspace"],
            document_model=WorkspaceDocumentModel,
            id=ObjectId(body.workspace),
            user_id=user_id_mock,
        )

        # Ensure that get_one was called with the correct arguments for schema
        get_one_mock.assert_any_call(
            collection=db_mock["schema"],
            document_model=SchemaDocumentModel,
            id=ObjectId(body.schema_),
            user_id=user_id_mock,
        )

        # Ensure that get_one was called with the correct arguments for graph
        filters = {
            "name": body.name,
            "workspace": body.workspace,
            "schema": body.schema_,
        }
        get_one_mock.assert_any_call(
            collection=db_mock["graph"],
            document_model=GraphDocumentModel,
            user_id=user_id_mock,
            filters=filters,
        )

    @patch("whyhow_api.dependencies.get_one", new_callable=AsyncMock)
    async def test_workspace_not_found(self, get_one_mock):
        user_id_mock = ObjectId()
        db_mock = AsyncMock()

        body = CreateGraphBody(
            name="test_graph",
            workspace=str(ObjectId()),
            schema_=str(ObjectId()),
            filters={"ids": [ObjectId(), ObjectId()]},
        )

        # Mock the get_one function for workspace call to return None
        get_one_mock.side_effect = [
            None,  # Workspace not found
        ]

        # Call the function and expect an HTTPException
        with pytest.raises(HTTPException) as exc_info:
            await valid_create_graph(
                body=body, user_id=user_id_mock, db=db_mock
            )

        assert exc_info.value.status_code == 404
        assert exc_info.value.detail == "Workspace not found."

    @patch("whyhow_api.dependencies.get_one", new_callable=AsyncMock)
    async def test_schema_not_found(self, get_one_mock):
        user_id_mock = ObjectId()
        db_mock = AsyncMock()

        body = CreateGraphBody(
            name="test_graph",
            workspace=str(ObjectId()),
            schema_=str(ObjectId()),
            filters={"ids": [ObjectId(), ObjectId()]},
        )

        # Mock the get_one function for workspace call to return a valid workspace
        # Mock the get_one function for schema call to return None
        get_one_mock.side_effect = [
            {
                "_id": ObjectId(body.workspace),
                "created_by": user_id_mock,
                "name": "test_workspace",
            },
            None,  # Schema not found
        ]

        # Call the function and expect an HTTPException
        with pytest.raises(HTTPException) as exc_info:
            await valid_create_graph(
                body=body, user_id=user_id_mock, db=db_mock
            )

        assert exc_info.value.status_code == 404
        assert exc_info.value.detail == "Schema not found."

    @patch("whyhow_api.dependencies.get_one", new_callable=AsyncMock)
    async def test_graph_already_exists(self, get_one_mock):
        user_id_mock = ObjectId()
        db_mock = AsyncMock()

        body = CreateGraphBody(
            name="test_graph",
            workspace=str(ObjectId()),
            schema_=str(ObjectId()),
            filters={"ids": [ObjectId(), ObjectId()]},
        )

        # Mock the get_one function for workspace call to return a valid workspace
        # Mock the get_one function for schema call to return a valid schema
        # Mock the get_one function for graph call to return an existing graph
        get_one_mock.side_effect = [
            {
                "_id": ObjectId(body.workspace),
                "created_by": user_id_mock,
                "name": "test_workspace",
            },
            {
                "_id": ObjectId(body.schema_),
                "created_by": user_id_mock,
                "name": "test_schema",
            },
            {
                "_id": ObjectId(),
                "name": body.name,
                "workspace": body.workspace,
                "schema": body.schema_,
            },
        ]

        # Call the function and expect an HTTPException
        with pytest.raises(HTTPException) as exc_info:
            await valid_create_graph(
                body=body, user_id=user_id_mock, db=db_mock
            )

        assert exc_info.value.status_code == 409
        assert (
            exc_info.value.detail
            == "Graph already exists or is being created."
        )


@pytest.mark.asyncio
class TestGetLLMClient:
    async def test_get_llm_client_with_valid_openai_provider(
        self, monkeypatch
    ):
        settings_mock = MagicMock()
        settings_mock.generative.openai.api_key.get_secret_value.return_value = (
            "test-api-key"
        )

        user_id_mock = ObjectId()
        db_mock = AsyncMock()

        # Mock the user document
        user_document = {
            "_id": ObjectId(),
            "providers": [
                {
                    "type": "llm",
                    "value": "byo-openai",
                    "api_key": "api key 1",
                    "metadata": {
                        "byo-openai": {
                            "language_model_name": None,
                            "embedding_name": None,
                        },
                        "byo-azure-openai": {
                            "api_version": None,
                            "azure_endpoint": None,
                            "language_model_name": None,
                            "embedding_name": None,
                        },
                    },
                }
            ],
        }
        db_mock.user.find_one.return_value = user_document

        # Call the function
        llm_client = await get_llm_client(
            user_id=user_id_mock, db=db_mock, settings=settings_mock
        )

        # Check that the correct client is returned
        assert isinstance(llm_client.client, AsyncOpenAI)

    async def test_get_llm_client_with_valid_azure_provider(self, monkeypatch):
        settings_mock = MagicMock()
        settings_mock.generative.openai.api_key.get_secret_value.return_value = (
            "test-api-key"
        )

        user_id_mock = ObjectId()
        db_mock = AsyncMock()

        # Mock the user document
        user_document = {
            "_id": ObjectId(),
            "providers": [
                {
                    "type": "llm",
                    "value": "byo-azure-openai",
                    "api_key": "api key 1",
                    "metadata": {
                        "byo-openai": {
                            "language_model_name": None,
                            "embedding_name": None,
                        },
                        "byo-azure-openai": {
                            "api_version": "test-version",
                            "azure_endpoint": "test-endpoint",
                            "language_model_name": "test-model",
                            "embedding_name": "test-embedding",
                        },
                    },
                }
            ],
        }
        db_mock.user.find_one.return_value = user_document

        # Call the function
        llm_client = await get_llm_client(
            user_id=user_id_mock, db=db_mock, settings=settings_mock
        )

        # Check that the correct client is returned
        assert isinstance(llm_client.client, AsyncAzureOpenAI)

        await llm_client.client.close()

    async def test_get_llm_client_with_valid_byo_openai_provider(
        self, monkeypatch
    ):
        settings_mock = MagicMock()
        settings_mock.generative.openai.api_key.get_secret_value.return_value = (
            "test-api-key"
        )

        user_id_mock = ObjectId()
        db_mock = AsyncMock()

        # Mock the user document
        user_document = {
            "_id": ObjectId(),
            "providers": [
                {
                    "type": "llm",
                    "value": "byo-openai",
                    "api_key": "api key 1",
                    "metadata": {
                        "byo-openai": {
                            "language_model_name": "test-model",
                            "embedding_name": "test-embedding",
                        },
                        "byo-azure-openai": {
                            "api_version": None,
                            "azure_endpoint": None,
                            "language_model_name": None,
                            "embedding_name": None,
                        },
                    },
                }
            ],
        }
        db_mock.user.find_one.return_value = user_document

        # Call the function
        llm_client = await get_llm_client(
            user_id=user_id_mock, db=db_mock, settings=settings_mock
        )

        # Check that the correct client is returned
        assert isinstance(llm_client.client, AsyncOpenAI)

    async def test_get_llm_client_with_invalid_provider(self):
        settings_mock = MagicMock()
        settings_mock.generative.openai.api_key.get_secret_value.return_value = (
            "test-api-key"
        )
        user_id_mock = ObjectId()
        db_mock = AsyncMock()

        # Mock the user document with an invalid provider
        user_document = {
            "_id": ObjectId(),
            "providers": [
                {
                    "type": "llm",
                    "value": "invalid-provider",
                    "api_key": None,
                    "metadata": {
                        "byo-openai": {
                            "language_model_name": None,
                            "embedding_name": None,
                        },
                        "byo-azure-openai": {
                            "api_version": None,
                            "azure_endpoint": None,
                            "language_model_name": None,
                            "embedding_name": None,
                        },
                    },
                }
            ],
        }
        db_mock.user.find_one.return_value = user_document

        # Call the function and check that it raises an HTTPException
        with pytest.raises(HTTPException):
            await get_llm_client(
                user_id=user_id_mock, db=db_mock, settings=settings_mock
            )

    async def test_get_llm_client_with_missing_user_id(self):
        settings_mock = MagicMock()
        settings_mock.generative.openai.api_key.get_secret_value.return_value = (
            "test-api-key"
        )
        user_id_mock = ObjectId()
        db_mock = AsyncMock()

        # Mock the user document as None
        db_mock.user.find_one.return_value = None

        # Call the function and check that it raises an HTTPException
        with pytest.raises(HTTPException):
            await get_llm_client(
                user_id=user_id_mock, db=db_mock, settings=settings_mock
            )


@pytest.mark.asyncio
class TestGetAuth0:

    @pytest.fixture
    def mock_settings(self):
        """Fixture to mock settings access."""
        settings = MagicMock()
        settings.api.auth0.client_domain = MagicMock()
        settings.api.auth0.client_id = MagicMock()
        settings.api.auth0.client_secret = MagicMock()
        return settings

    @patch("whyhow_api.dependencies.GetToken")
    async def test_get_auth0_success(self, mock_get_token, mock_settings):
        """Test successful Auth0 client retrieval."""
        # Mocking the behavior of GetToken and token retrieval
        mock_token_instance = MagicMock()
        mock_get_token.return_value = mock_token_instance
        mock_token_instance.client_credentials.return_value = {
            "access_token": "mocked_token"
        }

        auth0_client = await get_auth0(settings=mock_settings)

        assert isinstance(auth0_client, Auth0)

    async def test_get_auth0_missing_domain(self, mock_settings):
        """Test failure due to missing Auth0 client domain."""
        mock_settings.api.auth0.client_domain = None

        with pytest.raises(HTTPException) as excinfo:
            await get_auth0(settings=mock_settings)
        assert excinfo.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert "Auth0 domain is missing" in str(excinfo.value.detail)

    async def test_get_auth0_missing_client_id(self, mock_settings):
        """Test failure due to missing Auth0 client ID."""
        mock_settings.api.auth0.client_id = None

        with pytest.raises(HTTPException) as excinfo:
            await get_auth0(settings=mock_settings)
        assert excinfo.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert "Auth0 client ID is missing" in str(excinfo.value.detail)

    async def test_get_auth0_missing_client_secret(self, mock_settings):
        """Test failure due to missing Auth0 client secret."""
        mock_settings.api.auth0.client_secret = None

        with pytest.raises(HTTPException) as excinfo:
            await get_auth0(settings=mock_settings)
        assert excinfo.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert "Auth0 client secret is missing" in str(excinfo.value.detail)
