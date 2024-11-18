import sys
from unittest.mock import AsyncMock, MagicMock

import pandas
import pytest
from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorClientSession, AsyncIOMotorDatabase
from pymongo import InsertOne

from whyhow_api.schemas.chunks import (
    AddChunkModel,
    ChunkDocumentModel,
    ChunkMetadata,
    ChunkOut,
    UpdateChunkModel,
)
from whyhow_api.services.crud.chunks import (
    add_chunks,
    assign_chunks_to_workspace,
    create_structured_chunks,
    create_unstructured_chunks,
    delete_chunk,
    perform_node_chunk_unassignment,
    perform_triple_chunk_unassignment,
    prepare_chunks,
    process_structured_chunks,
    split_text_into_chunks,
    update_chunk,
    validate_and_convert,
)


@pytest.fixture
def document_id():
    return ObjectId()


@pytest.fixture
def workspace_id():
    return ObjectId()


@pytest.fixture
def user_id():
    return ObjectId()


@pytest.mark.asyncio
async def test_delete_chunk_success():
    fake_chunk_id = ObjectId()
    user_id = ObjectId()
    fake_chunk = ChunkOut(
        id=fake_chunk_id,
        created_by=user_id,
        document=ObjectId(),
        workspaces=[ObjectId()],
        data_type="object",
        content="test content",
        tags={str(user_id): ["tag1"]},
        user_metadata={str(user_id): {"hello": "world"}},
        metadata={
            "language": "en",
            "size": 10,
            "data_source_type": "manual",
        },
    )

    node_ids = [ObjectId(), ObjectId()]
    triple_ids = [ObjectId(), ObjectId()]

    db = MagicMock()
    db.node.aggregate.return_value.to_list = AsyncMock(
        return_value=[{"_id": node_id} for node_id in node_ids]
    )
    db.triple.aggregate.return_value.to_list = AsyncMock(
        return_value=[{"_id": triple_id} for triple_id in triple_ids]
    )
    db.chunk.find_one = AsyncMock(return_value=fake_chunk.model_dump())
    db.chunk.delete_one = AsyncMock(return_value=None)
    db.node.update_many = AsyncMock(return_value=None)
    db.triple.update_many = AsyncMock(return_value=None)

    session = MagicMock()
    session.start_transaction.return_value = AsyncMock()
    session.commit_transaction = AsyncMock()

    db_client = AsyncMock()
    db_client.start_session.return_value.__aenter__.return_value = session

    result = await delete_chunk(fake_chunk_id, db_client, db, user_id)

    assert result is not None
    assert isinstance(result, ChunkOut)
    db.chunk.find_one.assert_awaited_once_with(
        {"_id": fake_chunk_id, "created_by": user_id},
        {"embedding": 0},
        session=session,
    )
    db.chunk.delete_one.assert_awaited_once_with(
        {"_id": fake_chunk_id, "created_by": user_id}, session=session
    )
    session.commit_transaction.assert_awaited_once()


@pytest.mark.asyncio
async def test_delete_chunk_not_found():
    fake_chunk_id = ObjectId()
    user_id = ObjectId()

    db = MagicMock()
    db.node.aggregate.return_value.to_list = AsyncMock(return_value=[])
    db.triple.aggregate.return_value.to_list = AsyncMock(return_value=[])
    db.chunk.find_one = AsyncMock(return_value=None)
    db.chunk.delete_one = AsyncMock(return_value=None)
    db.node.update_many = AsyncMock(return_value=None)
    db.triple.update_many = AsyncMock(return_value=None)

    session = MagicMock()
    session.start_transaction.return_value = AsyncMock()
    session.commit_transaction = AsyncMock()

    db_client = AsyncMock()
    db_client.start_session.return_value.__aenter__.return_value = session

    result = await delete_chunk(fake_chunk_id, db_client, db, user_id)

    assert result is None


@pytest.mark.parametrize(
    "value,expected",
    [
        ("person", "person"),
        (10, 10),
        (0.1, 0.1),
        (True, True),
        (False, False),
        (None, None),
        ({"key": "value"}, "{'key': 'value'}"),
        (["list"], "['list']"),
        (b"binary", "b'binary'"),
    ],
)
def test_validate_and_convert(value, expected):
    """Test the validate_and_convert function with various types."""
    result = validate_and_convert(value)
    assert result == expected


def test_validate_and_convert_object_id():
    """Test the validate_and_convert function for ObjectId."""
    oid = ObjectId()
    assert validate_and_convert(oid) == str(oid)


class TestSplitTextIntoChunks:

    @pytest.fixture
    def text(self):
        return "This is some text content to split into chunks."

    @pytest.fixture
    def page_number(self):
        return 1

    @pytest.fixture
    def chunk_size(self):
        return 1

    @pytest.fixture
    def chunk_overlap(self):
        return 0

    def test_split_text_into_chunks_no_page_number(
        self, monkeypatch, text, chunk_size, chunk_overlap
    ):

        mock_settings = MagicMock()
        mock_settings.api.max_chars_per_chunk = chunk_size
        mock_settings.api.chunk_overlap = chunk_overlap

        monkeypatch.setattr(
            "whyhow_api.services.crud.chunks.settings", mock_settings
        )

        chunks = split_text_into_chunks(text=text)
        assert len(chunks) == len(text)
        assert chunks[0]["content"] == text[:1]
        assert chunks[0]["metadata"]["start"] == 0
        assert chunks[0]["metadata"]["end"] == 1

    def test_split_text_into_chunks_page_number(
        self, monkeypatch, text, chunk_size, chunk_overlap, page_number
    ):

        mock_settings = MagicMock()
        mock_settings.api.max_chars_per_chunk = chunk_size
        mock_settings.api.chunk_overlap = chunk_overlap

        monkeypatch.setattr(
            "whyhow_api.services.crud.chunks.settings", mock_settings
        )

        chunks = split_text_into_chunks(text=text, page_number=page_number)
        assert len(chunks) == len(text)
        assert chunks[0]["content"] == text[:1]
        assert chunks[0]["metadata"]["start"] == 0
        assert chunks[0]["metadata"]["end"] == 1
        assert chunks[0]["metadata"]["page"] == page_number

    def test_split_text_into_chunks_emtpy_text(self):
        chunks = split_text_into_chunks(text="")
        assert len(chunks) == 0


class TestChunkCreation:
    @pytest.fixture
    def basic_csv_content(self):
        """Fixture to provide basic CSV content."""
        return b"name,age\nJohn,30\nDoe,25"

    @pytest.fixture
    def complex_csv_content(self):
        """Fixture to provide CSV content with non-standard data types."""
        return b"name,age\nJohn,thirty\nDoe,25"

    @pytest.fixture
    def missing_csv_content(self):
        """Fixture to provide CSV content with missing data."""
        return b"name,age\nJohn\nDoe,25"

    @pytest.fixture
    def missing_header_csv_content(self):
        """Fixture to provide CSV content with missing header."""
        return b"John,30\nDoe,25"

    @pytest.fixture
    def empty_csv_content(self):
        """Fixture to provide empty CSV content."""
        return b""  # empty CSV content

    @pytest.fixture
    def document_id(self):
        """Fixture to provide a mock document ID."""
        return ObjectId()

    @pytest.fixture
    def workspace_id(self):
        """Fixture to provide a mock workspace ID."""
        return ObjectId()

    @pytest.fixture
    def user_id(self):
        """Fixture to provide a mock user ID."""
        return ObjectId()

    @pytest.fixture
    def basic_json_content(self):
        """Fixture to provide basic JSON content."""
        return b'[{"name": "John", "age": 30}, {"name": "Doe", "age": 25}]'

    @pytest.fixture
    def complex_json_content(self):
        """Fixture to provide JSON content with non-standard data types."""
        return (
            b'[{"name": "John", "age": "thirty"}, {"name": "Doe", "age": 25}]'
        )

    @pytest.fixture
    def nested_json_content(self):
        """Fixture to provide nested JSON content."""
        return b'[{"name": "John", "age": 30, "contact": {"email": "john@example.com"}}, {"name": "Doe", "age": 25}]'

    @pytest.fixture
    def empty_json_content(self):
        """Fixture to provide empty JSON content."""
        return b"[]"  # Empty JSON array

    def test_csv_processing(
        self, basic_csv_content, document_id, workspace_id, user_id
    ):
        """Test processing of basic CSV content."""
        chunks = create_structured_chunks(
            basic_csv_content, document_id, workspace_id, user_id, "csv"
        )
        assert len(chunks) == 2
        assert chunks[0].content == {"name": "John", "age": 30}
        assert chunks[1].content == {"name": "Doe", "age": 25}

    def test_csv_invalid_data_type_conversion(
        self, complex_csv_content, document_id, workspace_id, user_id
    ):
        """Test the conversion of non-standard data types in CSV."""
        chunks = create_structured_chunks(
            complex_csv_content, document_id, workspace_id, user_id, "csv"
        )
        assert (
            chunks[0].content["age"] == "thirty"
        )  # assuming your processing converts 'thirty' to a string

    def test_unsupported_file_type(
        self, basic_csv_content, document_id, workspace_id, user_id
    ):
        """Test handling of unsupported file types."""
        with pytest.raises(ValueError):
            create_structured_chunks(
                basic_csv_content, document_id, workspace_id, user_id, "xml"
            )

    def test_csv_missing_data(
        self, missing_csv_content, document_id, workspace_id, user_id
    ):
        """Test handling of missing data in CSV. This defaults to None."""
        chunks = create_structured_chunks(
            missing_csv_content, document_id, workspace_id, user_id, "csv"
        )
        assert len(chunks) == 2
        assert chunks[0].content == {"name": "John", "age": None}
        assert chunks[1].content == {"name": "Doe", "age": 25}

    def test_csv_missing_header(
        self, missing_header_csv_content, document_id, workspace_id, user_id
    ):
        """Test handling of missing header in CSV."""
        chunks = create_structured_chunks(
            missing_header_csv_content,
            document_id,
            workspace_id,
            user_id,
            "csv",
        )
        assert len(chunks) == 1
        # assuming the first row is treated as the header
        assert chunks[0].content == {"30": 25, "John": "Doe"}

    def test_empty_csv(
        self, empty_csv_content, document_id, workspace_id, user_id
    ):
        """Test handling of empty CSV."""
        with pytest.raises(pandas.errors.EmptyDataError):
            create_structured_chunks(
                empty_csv_content, document_id, workspace_id, user_id, "csv"
            )

    def test_json_processing(
        self, basic_json_content, document_id, workspace_id, user_id
    ):
        """Test processing of basic JSON content."""
        chunks = create_structured_chunks(
            basic_json_content, document_id, workspace_id, user_id, "json"
        )
        assert len(chunks) == 2
        assert chunks[0].content == {"name": "John", "age": 30}
        assert chunks[1].content == {"name": "Doe", "age": 25}

    def test_json_invalid_data_type_conversion(
        self, complex_json_content, document_id, workspace_id, user_id
    ):
        """Test the conversion of non-standard data types in JSON."""
        chunks = create_structured_chunks(
            complex_json_content, document_id, workspace_id, user_id, "json"
        )
        assert (
            chunks[0].content["age"] == "thirty"
        )  # Assuming your processing converts 'thirty' to a string

    def test_nested_json(
        self, nested_json_content, document_id, workspace_id, user_id
    ):
        """Test handling of nested JSON content."""
        chunks = create_structured_chunks(
            nested_json_content, document_id, workspace_id, user_id, "json"
        )
        assert len(chunks) == 2
        assert chunks[0].content == {
            "name": "John",
            "age": 30,
            "contact": "{'email': 'john@example.com'}",
        }
        assert chunks[1].content == {"name": "Doe", "contact": None, "age": 25}

    def test_empty_json(
        self, empty_json_content, document_id, workspace_id, user_id
    ):
        """Test handling of empty JSON."""
        chunks = create_structured_chunks(
            empty_json_content, document_id, workspace_id, user_id, "json"
        )
        assert len(chunks) == 0  # Expect no chunks from an empty JSON array


class TestPrepareChunks:
    @pytest.fixture
    def workspace_id(self):
        return ObjectId()

    @pytest.fixture
    def user_id(self):
        return ObjectId()

    @pytest.fixture
    def string_chunk(self):
        return AddChunkModel(
            content="This is a string",
            tags=["tag1"],
            user_metadata={"user": "data"},
        )

    @pytest.fixture
    def object_chunk(self):
        return AddChunkModel(
            content={"key": "value"},
            tags=["tag2"],
            user_metadata={"user": "info"},
        )

    @pytest.fixture
    def empty_tags_and_metadata_chunk(self):
        return AddChunkModel(
            content="Empty tags and metadata", tags=[], user_metadata={}
        )

    def test_prepare_chunks_with_string(
        self, workspace_id, user_id, string_chunk
    ):
        chunks = prepare_chunks([string_chunk], workspace_id, user_id)
        assert len(chunks) == 1
        chunk = chunks[0]
        assert chunk.content == "This is a string"
        assert chunk.data_type == "string"
        assert chunk.tags == {str(workspace_id): ["tag1"]}
        assert chunk.metadata.length == len("This is a string")
        assert chunk.metadata.size == sys.getsizeof("This is a string")
        assert chunk.metadata.data_source_type == "manual"
        assert chunk.user_metadata == {str(workspace_id): {"user": "data"}}
        assert chunk.created_by == user_id

    def test_prepare_chunks_with_object(
        self, workspace_id, user_id, object_chunk
    ):
        chunks = prepare_chunks([object_chunk], workspace_id, user_id)
        assert len(chunks) == 1
        chunk = chunks[0]
        assert chunk.content == {"key": "value"}
        assert chunk.data_type == "object"
        assert chunk.tags == {str(workspace_id): ["tag2"]}
        assert chunk.metadata.length == len({"key": "value"}.keys())
        assert chunk.metadata.size == sys.getsizeof({"key": "value"})
        assert chunk.user_metadata == {str(workspace_id): {"user": "info"}}

    def test_prepare_chunks_with_empty_tags_and_metadata(
        self, workspace_id, user_id, empty_tags_and_metadata_chunk
    ):
        chunks = prepare_chunks(
            [empty_tags_and_metadata_chunk], workspace_id, user_id
        )
        assert len(chunks) == 1
        chunk = chunks[0]
        assert chunk.tags == {}
        assert chunk.user_metadata == {}


@pytest.mark.asyncio
class TestProcessStructuredChunks:
    @pytest.fixture
    def content(self):
        return b"some content"

    @pytest.fixture
    def document_id(self):
        return ObjectId()

    @pytest.fixture
    def workspace_id(self):
        return ObjectId()

    @pytest.fixture
    def user_id(self):
        return ObjectId()

    @pytest.fixture
    def db(self):
        mock_db = AsyncMock()
        mock_collection = AsyncMock()
        mock_collection.update_one.return_value = AsyncMock()
        mock_db.__getitem__.return_value = mock_collection
        return mock_db

    @pytest.fixture
    def llm_client(self):
        return AsyncMock()

    async def test_process_invalid_file_type(
        self,
        monkeypatch,
        content,
        document_id,
        workspace_id,
        user_id,
        db,
        llm_client,
    ):
        # Create mocks
        mock_create = MagicMock(side_effect=ValueError("Invalid file type"))
        mock_add = AsyncMock()
        mock_update_one = AsyncMock()

        # Apply monkeypatch
        monkeypatch.setattr(
            "whyhow_api.services.crud.chunks.create_structured_chunks",
            mock_create,
        )
        monkeypatch.setattr(
            "whyhow_api.services.crud.chunks.add_chunks", mock_add
        )
        monkeypatch.setattr(
            "whyhow_api.services.crud.chunks.update_one", mock_update_one
        )

        # Execute the function and check for Exception
        with pytest.raises(Exception) as exc_info:
            await process_structured_chunks(
                content,
                document_id,
                db,
                llm_client,
                workspace_id,
                user_id,
                "xlsx",
            )

            assert "Unsupported file type selected" in str(exc_info.value)

        # Verify mocks
        mock_create.assert_called_once_with(
            content, document_id, workspace_id, user_id, "xlsx"
        )
        mock_add.assert_not_called()
        mock_update_one.assert_awaited_once()

    async def test_process_successful(
        self,
        monkeypatch,
        content,
        document_id,
        workspace_id,
        user_id,
        db,
        llm_client,
    ):
        # Create mocks
        mock_create = MagicMock(return_value=[{"data": "value"}])
        mock_add = AsyncMock(return_value=None)

        # Apply monkeypatch
        monkeypatch.setattr(
            "whyhow_api.services.crud.chunks.create_structured_chunks",
            mock_create,
        )
        monkeypatch.setattr(
            "whyhow_api.services.crud.chunks.add_chunks", mock_add
        )

        # Execute the function
        await process_structured_chunks(
            content, document_id, db, llm_client, workspace_id, user_id, "csv"
        )

        # Assertions to verify the mocks were called as expected
        mock_create.assert_called_once_with(
            content, document_id, workspace_id, user_id, "csv"
        )
        mock_add.assert_called_once_with(
            db, llm_client, mock_create.return_value
        )


class TestCreateUnstructuredChunks:
    @pytest.fixture
    def pdf_content(self):
        # Create a mock PDF file content
        return b"%PDF-1.4 example content"

    @pytest.fixture
    def txt_content(self):
        return b"Hello, this is some text content for testing."

    def test_txt_file_processing(
        self, txt_content, document_id, workspace_id, user_id
    ):
        chunks = create_unstructured_chunks(
            txt_content, document_id, workspace_id, user_id, "txt"
        )

        assert len(chunks) == len(
            split_text_into_chunks(
                "Hello, this is some text content for testing."
            )
        )
        assert isinstance(chunks[0], ChunkDocumentModel)
        assert (
            chunks[0].content
            == "Hello, this is some text content for testing."[:100]
        )


@pytest.mark.asyncio
class TestAssignChunksToWorkspace:
    @pytest.fixture
    def db(self, monkeypatch):
        # Mock the database and collection
        mock_db = AsyncMock()
        mock_collection = AsyncMock()
        mock_cursor = AsyncMock()

        # Prepare the data to be returned by to_list
        chunks = [
            {"_id": ObjectId(), "workspaces": []},
            {"_id": ObjectId(), "workspaces": [ObjectId()]},
            {"_id": ObjectId(), "workspaces": []},
        ]
        mock_cursor.to_list = AsyncMock(return_value=chunks)

        # Monkeypatch the find method to return the mock cursor
        monkeypatch.setattr(
            mock_collection, "find", lambda *args, **kwargs: mock_cursor
        )
        monkeypatch.setattr(mock_db, "chunk", mock_collection)

        return mock_db

    @pytest.fixture
    def chunk_ids(self):
        return [ObjectId() for _ in range(3)]

    @pytest.fixture
    def workspace_id(self):
        return ObjectId()

    @pytest.fixture
    def user_id(self):
        return ObjectId()

    async def test_all_chunks_assigned(
        self, monkeypatch, chunk_ids, workspace_id, user_id
    ):
        mock_db = AsyncMock()
        mock_collection = AsyncMock()
        mock_cursor = AsyncMock()
        additional_chunk_id = ObjectId()
        chunks = [{"_id": oid, "workspaces": []} for oid in chunk_ids] + [
            {"_id": additional_chunk_id, "workspaces": [workspace_id]}
        ]
        mock_cursor.to_list = AsyncMock(return_value=chunks)

        monkeypatch.setattr(
            mock_collection, "find", lambda *args, **kwargs: mock_cursor
        )
        monkeypatch.setattr(mock_db, "chunk", mock_collection)

        result = await assign_chunks_to_workspace(
            db=mock_db,
            chunk_ids=chunk_ids + [additional_chunk_id],
            workspace_id=workspace_id,
            user_id=user_id,
        )

        assert len(result.assigned) == 3
        assert len(result.already_assigned) == 1
        assert len(result.not_found) == 0

    async def test_chunks_not_found(
        self, monkeypatch, chunk_ids, workspace_id, user_id
    ):
        mock_db = AsyncMock()
        mock_collection = AsyncMock()
        mock_cursor = AsyncMock()

        chunks = []
        mock_cursor.to_list = AsyncMock(return_value=chunks)

        monkeypatch.setattr(
            mock_collection, "find", lambda *args, **kwargs: mock_cursor
        )
        monkeypatch.setattr(mock_db, "chunk", mock_collection)

        result = await assign_chunks_to_workspace(
            db=mock_db,
            chunk_ids=chunk_ids,
            workspace_id=workspace_id,
            user_id=user_id,
        )

        assert len(result.assigned) == 0
        assert len(result.already_assigned) == 0
        assert len(result.not_found) == 3

    async def test_all_already_assigned(
        self, monkeypatch, chunk_ids, workspace_id, user_id
    ):
        mock_db = AsyncMock()
        mock_collection = AsyncMock()
        mock_cursor = AsyncMock()

        chunks = [
            {"_id": oid, "workspaces": [workspace_id]} for oid in chunk_ids
        ]
        mock_cursor.to_list = AsyncMock(return_value=chunks)

        monkeypatch.setattr(
            mock_collection, "find", lambda *args, **kwargs: mock_cursor
        )
        monkeypatch.setattr(mock_db, "chunk", mock_collection)

        result = await assign_chunks_to_workspace(
            db=mock_db,
            chunk_ids=chunk_ids,
            workspace_id=workspace_id,
            user_id=user_id,
        )

        assert len(result.assigned) == 0
        assert len(result.already_assigned) == 3
        assert len(result.not_found) == 0


@pytest.mark.asyncio
async def test_perform_node_chunk_unassignment_success():
    mock_chunk_ids_to_delete = [ObjectId(), ObjectId(), ObjectId()]
    mock_node_response = [{"_id": 1}, {"_id": 2}, {"_id": 3}]
    mock_user_id = ObjectId()

    # Create a mock database
    mock_db = MagicMock(spec=AsyncIOMotorDatabase)
    mock_session = MagicMock(spec=AsyncIOMotorClientSession)

    # Create a mock collection
    mock_collection = MagicMock()

    # Create a mock cursor
    mock_cursor = AsyncMock()
    mock_cursor.to_list.return_value = mock_node_response

    # Set up the mock collection to return the mock cursor
    mock_collection.find.return_value = mock_cursor

    # Assign the mock collection to the 'node' attribute of the mock database
    mock_db.node = mock_collection
    mock_db.node.update_many = AsyncMock(return_value=None)

    # Call the function
    result = await perform_node_chunk_unassignment(
        mock_db, mock_session, mock_chunk_ids_to_delete, mock_user_id
    )

    # Assert that the function returned the expected result
    assert result is None

    # Assert that the database methods were called correctly
    mock_db.node.update_many.assert_called_once_with(
        {
            "chunks": {"$in": mock_chunk_ids_to_delete},
            "created_by": mock_user_id,
        },
        {"$pull": {"chunks": {"$in": mock_chunk_ids_to_delete}}},
        session=mock_session,
    )


@pytest.mark.asyncio
async def test_perform_node_triple_unassignment_success():
    mock_chunk_ids_to_delete = [ObjectId(), ObjectId(), ObjectId()]
    mock_user_id = ObjectId()

    # Create a mock database
    mock_db = MagicMock(spec=AsyncIOMotorDatabase)
    mock_session = MagicMock(spec=AsyncIOMotorClientSession)

    mock_collection = MagicMock()
    mock_db.triple = mock_collection
    mock_db.triple.update_many = AsyncMock(return_value=None)

    # Call the function
    await perform_triple_chunk_unassignment(
        mock_db, mock_session, mock_chunk_ids_to_delete, mock_user_id
    )

    mock_db.triple.update_many.assert_called_once_with(
        {
            "chunks": {"$in": mock_chunk_ids_to_delete},
            "created_by": mock_user_id,
        },
        {"$pull": {"chunks": {"$in": mock_chunk_ids_to_delete}}},
        session=mock_session,
    )


@pytest.mark.asyncio
async def test_update_chunk_success(monkeypatch):
    mock_chunk_id = ObjectId()
    mock_workspace_id = ObjectId()
    mock_user_id = ObjectId()

    # Mock database and its methods
    mock_db = MagicMock()
    mock_db.chunk = AsyncMock()
    mock_db.chunk.update_one = AsyncMock()

    # Mock UpdateChunkModel
    mock_body = MagicMock(spec=UpdateChunkModel)
    mock_body.model_dump.return_value = {
        "field1": "value1",
        "field2": "value2",
    }

    # Mock ObjectId
    mock_object_id = MagicMock(spec=ObjectId)

    # Mock get_chunks function
    async def mock_get_chunks(*args, **kwargs):
        return [{"_id": mock_object_id, "content": "mocked chunk"}]

    monkeypatch.setattr(
        "whyhow_api.services.crud.chunks.get_chunks", mock_get_chunks
    )

    mock_db.chunk.update_one.return_value = AsyncMock(
        matched_count=1, modified_count=1
    )

    message, chunks = await update_chunk(
        chunk_id=mock_chunk_id,
        workspace_id=mock_workspace_id,
        body=mock_body,
        user_id=mock_user_id,
        db=mock_db,
    )

    assert message == "Chunk updated successfully"
    assert len(chunks) == 1
    assert chunks[0]["content"] == "mocked chunk"


@pytest.mark.asyncio
async def test_update_chunk_not_found(monkeypatch):
    mock_chunk_id = ObjectId()
    mock_workspace_id = ObjectId()
    mock_user_id = ObjectId()

    # Mock database and its methods
    mock_db = MagicMock()
    mock_db.chunk = AsyncMock()
    mock_db.chunk.update_one = AsyncMock()

    # Mock UpdateChunkModel
    mock_body = MagicMock(spec=UpdateChunkModel)
    mock_body.model_dump.return_value = {
        "field1": "value1",
        "field2": "value2",
    }

    # Mock ObjectId
    mock_object_id = MagicMock(spec=ObjectId)

    # Mock get_chunks function
    async def mock_get_chunks(*args, **kwargs):
        return [{"_id": mock_object_id, "content": "mocked chunk"}]

    monkeypatch.setattr(
        "whyhow_api.services.crud.chunks.get_chunks", mock_get_chunks
    )

    # Test case where no document is found
    mock_db.chunk.update_one.return_value = AsyncMock(
        matched_count=0, modified_count=0
    )

    message, chunks = await update_chunk(
        chunk_id=mock_chunk_id,
        workspace_id=mock_workspace_id,
        body=mock_body,
        user_id=mock_user_id,
        db=mock_db,
    )

    assert message == "No chunk found to update"


@pytest.mark.asyncio
async def test_update_chunk_no_change(monkeypatch):
    mock_chunk_id = ObjectId()
    mock_workspace_id = ObjectId()
    mock_user_id = ObjectId()

    # Mock database and its methods
    mock_db = MagicMock()
    mock_db.chunk = AsyncMock()
    mock_db.chunk.update_one = AsyncMock()

    # Mock UpdateChunkModel
    mock_body = MagicMock(spec=UpdateChunkModel)
    mock_body.model_dump.return_value = {
        "field1": "value1",
        "field2": "value2",
    }

    # Mock ObjectId
    mock_object_id = MagicMock(spec=ObjectId)

    # Mock get_chunks function
    async def mock_get_chunks(*args, **kwargs):
        return [{"_id": mock_object_id, "content": "mocked chunk"}]

    monkeypatch.setattr(
        "whyhow_api.services.crud.chunks.get_chunks", mock_get_chunks
    )

    # Test case where document is found but not modified
    mock_db.chunk.update_one.return_value = AsyncMock(
        matched_count=1, modified_count=0
    )

    message, chunks = await update_chunk(
        chunk_id=mock_chunk_id,
        workspace_id=mock_workspace_id,
        body=mock_body,
        user_id=mock_user_id,
        db=mock_db,
    )

    assert message == "No changes made to the chunk"


@pytest.mark.asyncio
async def test_add_chunks(monkeypatch):

    chunk_id = ObjectId
    monkeypatch.setattr(
        "whyhow_api.services.crud.chunks.embed_texts",
        AsyncMock(return_value=[[0.1, 0.1]]),
    )
    monkeypatch.setattr(
        "whyhow_api.services.crud.chunks.ObjectId",
        MagicMock(return_value=chunk_id),
    )

    chunks_in = [
        ChunkDocumentModel(
            content="Test content",
            data_type="string",
            created_by=ObjectId(),
            metadata=ChunkMetadata(
                language="en",
                size=10,
                data_source_type="manual",
            ),
            workspaces=[ObjectId()],
            tags={},
            user_metadata={},
        )
    ]
    chunks_out = []
    for c in chunks_in:
        c.embedding = [0.1, 0.1]
        chunks_out.append(c.model_dump(by_alias=True, exclude_none=True))

    # Mock database and its methods
    mock_db = MagicMock()
    mock_db.chunk = AsyncMock()
    mock_db.chunk.bulk_write = AsyncMock()
    mock_db.chunk.find = MagicMock()
    mock_db.chunk.find.return_value.to_list = AsyncMock(return_value=[])

    await add_chunks(
        db=mock_db,
        llm_client=AsyncMock(),
        chunks=chunks_in,
    )
    # Assert that bulk_write was called correctly
    chunks_out_updated = []
    for chunk in chunks_out:
        chunk["_id"] = chunk_id
        chunks_out_updated.append(chunk)
    expected_operations = [InsertOne(chunk) for chunk in chunks_out_updated]
    mock_db.chunk.bulk_write.assert_called_once_with(expected_operations)


@pytest.mark.asyncio
async def test_add_chunks_bulk_write_error():
    pass
