import json

import pytest
from bson import ObjectId
from pydantic import ValidationError

from whyhow_api.config import Settings
from whyhow_api.schemas.chunks import (
    AddChunkModel,
    ChunkDocumentModel,
    ChunkMetadata,
    ChunkOut,
    UpdateChunkModel,
)


class TestChunkMetadata:
    def test_default_values(self):
        """Test that default values are correctly applied."""
        chunk_metadata = ChunkMetadata()
        assert chunk_metadata.language == "en"
        assert chunk_metadata.length is None
        assert chunk_metadata.size is None
        assert chunk_metadata.data_source_type is None
        assert chunk_metadata.index is None
        assert chunk_metadata.page is None
        assert chunk_metadata.start is None
        assert chunk_metadata.end is None

    def test_field_types(self):
        """Test that fields accept correct types."""
        chunk_metadata = ChunkMetadata(
            language="fr",
            length=100,
            size=200,
            data_source_type="manual",
            index=1,
            page=10,
            start=0,
            end=100,
        )
        assert chunk_metadata.language == "fr"
        assert chunk_metadata.length == 100
        assert chunk_metadata.size == 200
        assert chunk_metadata.data_source_type == "manual"
        assert chunk_metadata.index == 1
        assert chunk_metadata.page == 10
        assert chunk_metadata.start == 0
        assert chunk_metadata.end == 100

    def test_optional_fields(self):
        """Test that optional fields can be omitted."""
        chunk_metadata = ChunkMetadata(language="es")
        assert chunk_metadata.language == "es"
        # Optional fields should default to None
        assert chunk_metadata.length is None
        assert chunk_metadata.size is None

    def test_invalid_data_source_type(self):
        """Test that an invalid data_source_type raises a ValidationError."""
        with pytest.raises(ValidationError):
            ChunkMetadata(data_source_type="invalid_type")

    def test_serialization_deserialization(self):
        """Test model serialization and deserialization."""
        chunk_metadata = ChunkMetadata(
            language="de",
            length=150,
            size=250,
            data_source_type="external",
            index=2,
            page=20,
            start=100,
            end=200,
        )
        serialized = chunk_metadata.model_dump_json()
        deserialized = ChunkMetadata.model_validate_json(serialized)
        assert deserialized == chunk_metadata


class TestChunkDocumentModel:
    def test_field_types_and_defaults(self):
        """Test that fields accept correct types and have correct default values."""

        workspace_id_mock = ObjectId()

        chunk_document = ChunkDocumentModel(
            created_by=ObjectId(),
            workspaces=[workspace_id_mock],
            data_type="string",
            content="This is a test content",
            metadata=ChunkMetadata(),
        )
        assert chunk_document.workspaces[0] == workspace_id_mock
        assert chunk_document.document is None  # Optional field
        assert chunk_document.data_type == "string"
        assert chunk_document.content == "This is a test content"
        assert chunk_document.tags == {}  # Default value
        assert chunk_document.embedding is None  # Optional field
        assert chunk_document.user_metadata == {}

    def test_optional_fields(self):
        """Test that optional fields can be omitted."""
        workspace_id_mock = ObjectId()

        chunk_document = ChunkDocumentModel(
            created_by=ObjectId(),
            workspaces=[workspace_id_mock],
            data_type="object",
            content={"key": "value"},
            metadata=ChunkMetadata(),
        )
        # Optional fields should default to None or their default values
        assert chunk_document.document is None
        assert chunk_document.embedding is None
        assert isinstance(chunk_document.user_metadata, dict)

    def test_invalid_data_type(self):
        """Test that an invalid data_type raises a ValidationError."""
        workspace_id_mock = ObjectId()
        with pytest.raises(ValidationError):
            ChunkDocumentModel(
                created_by=ObjectId(),
                workspace=workspace_id_mock,
                data_type="invalid_type",  # Invalid data_type
                content="This is a test content",
                metadata=ChunkMetadata(),
            )


class TestChunkOut:
    def test_field_aliases(self):
        """Test that field aliases are correctly applied."""
        workspace_id = ObjectId()
        chunk_data = {
            "_id": ObjectId(),
            "created_by": ObjectId(),
            "workspaces": [workspace_id],
            "document": ObjectId(),
            "data_type": "string",
            "content": "Test",
            "metadata": {},
            "tags": {str(workspace_id): []},
            "user_metadata": {str(workspace_id): {}},
        }
        chunk = ChunkOut(**chunk_data)
        assert chunk.id == str(chunk_data["_id"])

    def test_field_types(self):
        """Test that fields accept correct types."""
        workspace_id = ObjectId()
        chunk_data = {
            "_id": ObjectId(),
            "created_by": ObjectId(),
            "workspaces": [workspace_id],
            "document": None,  # Document can be None
            "data_type": "string",
            "content": "Test",
            "metadata": {},
            "tags": {str(workspace_id): []},
            "user_metadata": {str(workspace_id): {}},
        }
        chunk = ChunkOut(**chunk_data)
        assert isinstance(chunk.created_by, str)
        assert isinstance(chunk.workspaces, list)
        assert isinstance(chunk.workspaces[0], str)
        assert isinstance(chunk.id, str)
        assert chunk.document is None

    def test_invalid_field_type(self):
        """Test that an invalid field type raises a ValidationError."""
        with pytest.raises(ValidationError):
            ChunkOut(
                _id="not_an_object_id",  # This should cause a validation error
                created_by=ObjectId(),
                workspaces=[ObjectId()],
                document=ObjectId(),
            )

    def test_serialization(self):
        """Test model serialization."""
        workspace_id = ObjectId()
        chunk_data = {
            "_id": ObjectId(),
            "created_by": ObjectId(),
            "workspaces": [workspace_id],
            "document": None,  # Document can be None
            "data_type": "string",
            "content": "Test",
            "metadata": {},
            "tags": {str(workspace_id): []},
            "user_metadata": {str(workspace_id): {}},
        }
        chunk = ChunkOut(**chunk_data)
        serialized_chunk = chunk.model_dump_json(by_alias=True)
        for key in ["_id", "created_by", "workspaces", "document"]:
            assert key in serialized_chunk

    def test_serialization_deserialization(self):
        """Test model serialization and deserialization."""
        workspace_id = ObjectId()
        chunk_data = {
            "_id": ObjectId(),
            "created_by": ObjectId(),
            "workspaces": [workspace_id],
            "document": None,  # Document can be None
            "data_type": "string",
            "content": "Test",
            "metadata": {},
            "tags": {str(workspace_id): []},
            "user_metadata": {str(workspace_id): {}},
        }
        chunk_document = ChunkOut(**chunk_data)
        serialized = chunk_document.model_dump_json()
        deserialized = ChunkOut.model_validate_json(serialized)
        assert deserialized == chunk_document


class TestAddChunkModel:
    def test_content_as_string(self):
        """Test model with content as a string."""
        model = AddChunkModel(content="This is a test string.")
        assert model.content == "This is a test string."

    def test_content_as_dict(self):
        """Test model with content as a dictionary."""
        content_dict = {"key": "value"}
        model = AddChunkModel(content=content_dict)
        assert model.content == content_dict

    def test_content_as_dict_with_invalid_value_dict(self):
        """Test model with content as a dictionary with an invalid value."""
        content_dict = {"key": {"hello": "world"}}
        with pytest.raises(ValidationError):
            AddChunkModel(content=content_dict)

    def test_content_as_dict_with_invalid_value_list_with_dict(self):
        """Test model with content as a list with an invalid dictionary value."""
        content_list_with_dict = {"key": [{"hello": "world"}]}
        with pytest.raises(ValidationError):
            AddChunkModel(content=content_list_with_dict)

    def test_content_as_integer(self):
        """Test model with content as an integer."""
        with pytest.raises(ValidationError):
            AddChunkModel(content=123)

    def test_user_metadata_and_tags_defaults(self):
        """Test default values for user_metadata and tags."""
        model = AddChunkModel(content="Test")
        assert model.user_metadata is None
        assert model.tags is None

    def test_user_metadata_and_tags_assignment(self):
        """Test assignment of user_metadata and tags."""
        user_metadata = {"meta_key": "meta_value"}
        tags = ["tag1", "tag2"]
        model = AddChunkModel(
            content="Test", user_metadata=user_metadata, tags=tags
        )
        assert model.user_metadata == user_metadata
        assert model.tags == tags

    def test_content_length_validation_for_string(self):
        """Test content length validation for string."""

        settings = Settings()

        long_string = "a" * (
            settings.api.max_chars_per_chunk + 1
        )  # One character too long
        with pytest.raises(ValueError):
            AddChunkModel(content=long_string)

    def test_content_length_validation_for_dict(self):
        """Test content length validation for dictionary."""

        settings = Settings()

        long_dict = {
            "key": "a" * (settings.api.max_chars_per_chunk - 9)
        }  # Adjusted for JSON formatting overhead
        with pytest.raises(ValueError):
            AddChunkModel(content=long_dict)

    def test_model_allows_maximum_length_content(self):
        """Test that the model allows content up to the maximum length."""

        settings = Settings()

        max_length_string = "a" * settings.api.max_chars_per_chunk
        max_length_dict = json.loads(
            '{"key": "' + "a" * (settings.api.max_chars_per_chunk - 11) + '"}'
        )  # Adjust for JSON overhead
        string_model = AddChunkModel(content=max_length_string)
        dict_model = AddChunkModel(content=max_length_dict)
        assert len(string_model.content) == settings.api.max_chars_per_chunk
        assert (
            len(json.dumps(dict_model.content))
            <= settings.api.max_chars_per_chunk
        )


class TestUpdateChunkModel:
    def test_model_creation_with_valid_data(self):
        """Test model creation with valid data."""
        model_data = {
            "user_metadata": {"key": "value"},
            "tags": ["tag1", "tag2"],
        }
        model = UpdateChunkModel(**model_data)
        assert model.user_metadata == model_data["user_metadata"]
        assert model.tags == model_data["tags"]

    def test_model_default_values(self):
        """Test that default values are correctly applied."""
        model = UpdateChunkModel()
        assert model.user_metadata is None
        assert model.tags is None

    def test_model_with_none_values(self):
        """Test model instantiation with None values for optional fields."""
        model = UpdateChunkModel(user_metadata=None, tags=None)
        assert model.user_metadata is None
        assert model.tags is None

    def test_model_type_enforcement_for_user_metadata(self):
        """Test type enforcement for user_metadata."""
        with pytest.raises(ValidationError):
            UpdateChunkModel(
                user_metadata="not a dict"
            )  # Should raise an error

    def test_model_type_enforcement_for_tags(self):
        """Test type enforcement for tags."""
        with pytest.raises(ValidationError):
            UpdateChunkModel(tags="not a list")  # Should raise an error

    def test_model_accepts_empty_dict_and_list(self):
        """Test that the model accepts an empty dict for user_metadata and an empty list for tags."""
        model = UpdateChunkModel(user_metadata={}, tags=[])
        assert model.user_metadata == {}
        assert model.tags == []
