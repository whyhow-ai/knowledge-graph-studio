"""Tests for the graph service."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from bson import ObjectId

from whyhow_api.models.common import (
    EntityField,
    SchemaEntity,
    SchemaRelation,
    SchemaTriplePattern,
    Triple,
)
from whyhow_api.schemas.chunks import ChunkDocumentModel, ChunkMetadata
from whyhow_api.schemas.nodes import NodeWithIdAndSimilarity
from whyhow_api.services.crud.triple import embed_triples
from whyhow_api.services.graph_service import (
    MixedQueryProcessor,
    apply_rules,
    clusters_pipeline,
    convert_pattern_to_text,
    convert_triple_to_text,
    create_node_id_map,
    create_structured_patterns,
    extract_properties_from_fields,
    extract_structured_graph_triples,
    get_and_separate_chunks_on_data_type,
    get_similar_nodes,
    merge_dicts,
    node_keys,
    triple_key,
)


class TestMergeDicts:

    def test_merge_simple_dicts(self):
        # Arrange
        d1 = {"a": 1, "b": 2}
        d2 = {"b": 3, "c": 4}
        expected = {"a": 1, "b": [2, 3], "c": 4}

        # Act
        result = merge_dicts(d1, d2)

        # Assert
        assert result == expected

    def test_merge_dicts_with_lists(self):
        # Arrange
        d1 = {"a": [1, 2], "b": [3]}
        d2 = {"a": [4], "b": [5], "c": [6]}
        expected = {"a": [1, 2, 4], "b": [3, 5], "c": [6]}

        # Act
        result = merge_dicts(d1, d2)

        # Assert
        assert result == expected

    def test_merge_dicts_multiple_sequential_operations(self):
        # Arrange
        d1 = {"a": 1, "b": 2}
        d2 = {"a": 2, "b": 3, "c": 4}
        d3 = {"a": 5, "b": 6, "c": 7}
        expected = {"a": [1, 2, 5], "b": [2, 3, 6], "c": [4, 7]}

        # Act
        result = merge_dicts(d1, d2)
        result = merge_dicts(result, d3)

        # Assert
        assert result == expected

    def test_merge_nested_dicts(self):
        # Arrange
        d1 = {"a": {"x": 1}, "b": {"y": 2}}
        d2 = {"a": {"z": 3}, "b": {"y": 4}, "c": {"w": 5}}
        expected = {"a": {"x": 1, "z": 3}, "b": {"y": [2, 4]}, "c": {"w": 5}}

        # Act
        result = merge_dicts(d1, d2)

        # Assert
        assert result == expected

    def test_merge_dicts_with_conflicting_types(self):
        # Arrange
        d1 = {"a": [1, 2], "b": {"x": 3}}
        d2 = {"a": {"y": 4}, "b": [5]}
        expected = {"a": [1, 2, {"y": 4}], "b": [{"x": 3}, 5]}

        # Act
        result = merge_dicts(d1, d2)

        # Assert
        assert result == expected

    def test_merge_dicts_with_empty_dicts(self):
        # Arrange
        d1 = {}
        d2 = {"a": 1}
        expected = {"a": 1}

        # Act
        result = merge_dicts(d1, d2)

        # Assert
        assert result == expected

    def test_merge_dicts_with_lists_and_conflicting_types(self):
        # Arrange
        d1 = {"a": [1, 2], "b": 3}
        d2 = {"a": 4, "b": [5, 6]}
        expected = {"a": [1, 2, 4], "b": [3, 5, 6]}

        # Act
        result = merge_dicts(d1, d2)

        # Assert
        assert result == expected


class TestTripleKey:

    def test_triple_key(self):
        # Arrange
        triple = Triple(
            head="subject",
            head_type="type1",
            relation="relates_to",
            tail="object",
            tail_type="type2",
        )
        expected = ("subject", "type1", "relates_to", "object", "type2")

        # Act
        result = triple_key(triple)

        # Assert
        assert result == expected

    def test_triple_key_with_special_characters(self):
        # Arrange
        triple = Triple(
            head="sub@ject",
            head_type="ty#pe1",
            relation="re!lates_to",
            tail="obj*ect",
            tail_type="ty(pe2",
        )
        expected = ("sub@ject", "ty#pe1", "re!lates_to", "obj*ect", "ty(pe2")

        # Act
        result = triple_key(triple)

        # Assert
        assert result == expected

    def test_triple_key_with_numbers(self):
        # Arrange
        triple = Triple(
            head="123",
            head_type="type1",
            relation="456",
            tail="789",
            tail_type="type2",
        )
        expected = ("123", "type1", "456", "789", "type2")

        # Act
        result = triple_key(triple)

        # Assert
        assert result == expected


class TestNodeKeys:

    def test_node_keys(self):
        # Arrange
        triple = Triple(
            head="subject",
            head_type="type1",
            relation="relates_to",
            tail="object",
            tail_type="type2",
        )
        expected = (("subject", "type1"), ("object", "type2"))

        # Act
        result = node_keys(triple)

        # Assert
        assert result == expected

    def test_node_keys_with_special_characters(self):
        # Arrange
        triple = Triple(
            head="sub@ject",
            head_type="ty#pe1",
            relation="re!lates_to",
            tail="obj*ect",
            tail_type="ty(pe2",
        )
        expected = (("sub@ject", "ty#pe1"), ("obj*ect", "ty(pe2"))

        # Act
        result = node_keys(triple)

        # Assert
        assert result == expected

    def test_node_keys_with_numbers(self):
        # Arrange
        triple = Triple(
            head="123",
            head_type="type1",
            relation="456",
            tail="789",
            tail_type="type2",
        )
        expected = (("123", "type1"), ("789", "type2"))

        # Act
        result = node_keys(triple)

        # Assert
        assert result == expected


@pytest.mark.parametrize(
    "name, type, user_id, graph_id, expected_pipeline",
    [
        (
            "test_name",
            "test_type",
            ObjectId("60dbf4a206b94b214fded12a"),
            ObjectId("60dbf4a206b94b214fded12b"),
            [
                {
                    "$search": {
                        "index": "node_index",
                        "text": {
                            "query": "test_name",
                            "path": "name",
                            "fuzzy": {"maxEdits": 1},
                        },
                    }
                },
                {
                    "$match": {
                        "type": "test_type",
                        "graph": ObjectId("60dbf4a206b94b214fded12b"),
                        "created_by": ObjectId("60dbf4a206b94b214fded12a"),
                    }
                },
                {
                    "$project": {
                        "_id": 1,
                        "name": 1,
                        "label": "$type",
                        "properties": 1,
                        "similarity": {"$meta": "searchScore"},
                    }
                },
                {"$match": {"similarity": {"$gt": len("test_name") / 5}}},
                {"$group": {"_id": None, "nodes": {"$push": "$$ROOT"}}},
                {"$project": {"_id": 0, "nodes": 1}},
            ],
        )
    ],
)
def test_clusters_pipeline(name, type, user_id, graph_id, expected_pipeline):
    pipeline = clusters_pipeline(name, type, user_id, graph_id)
    assert pipeline == expected_pipeline


@pytest.mark.asyncio
async def test_get_similar_nodes():
    graph_id = ObjectId("60dbf4a206b94b214fded12a")
    user_id = ObjectId("60dbf4a206b94b214fded12b")

    db = MagicMock()
    db.node.aggregate.return_value = AsyncMock()
    db.node.aggregate.return_value.to_list.return_value = [
        {"name": "node1", "type": "type1"},
        {"name": "node2", "type": "type2"},
        {"name": "node3", "type": "type3"},
    ]

    cluster_cursor_mock = AsyncMock()
    cluster_cursor_mock.to_list.side_effect = [
        [
            {
                "nodes": [
                    {
                        "_id": ObjectId(),
                        "name": "node1",
                        "label": "type1",
                        "properties": {},
                        "similarity": 1.3,
                    }
                ]
            }
        ],
        [
            {
                "nodes": [
                    {
                        "_id": ObjectId(),
                        "name": "node2",
                        "label": "type2",
                        "properties": {},
                        "similarity": 1.4,
                    },
                    {
                        "_id": ObjectId(),
                        "name": "node3",
                        "label": "type3",
                        "properties": {},
                        "similarity": 1.5,
                    },
                ]
            }
        ],
        [
            {
                "nodes": [
                    {
                        "_id": ObjectId(),
                        "name": "node4",
                        "label": "type4",
                        "properties": {},
                        "similarity": 1.6,
                    },
                    {
                        "_id": ObjectId(),
                        "name": "node5",
                        "label": "type5",
                        "properties": {},
                        "similarity": 1.7,
                    },
                ]
            }
        ],
    ]
    db.node.aggregate.side_effect = [
        db.node.aggregate.return_value,
        cluster_cursor_mock,
        cluster_cursor_mock,
        cluster_cursor_mock,
    ]

    limit = 1
    result = await get_similar_nodes(db, graph_id, user_id, limit)

    assert isinstance(result, list)
    assert len(result) == limit
    assert isinstance(result[0], list)
    assert isinstance(result[0][0], NodeWithIdAndSimilarity)


@pytest.mark.asyncio
async def test_get_similar_nodes_empty():
    graph_id = ObjectId("60dbf4a206b94b214fded12a")
    user_id = ObjectId("60dbf4a206b94b214fded12b")

    db = MagicMock()
    db.node.aggregate.return_value = AsyncMock()
    db.node.aggregate.return_value.to_list.return_value = []

    result = await get_similar_nodes(db, graph_id, user_id)

    assert isinstance(result, list)
    assert len(result) == 0


class TestTripleToText:
    def test_basic_triple(self):
        triple = {
            "head_type": "Person",
            "head": "Alice",
            "relation": "is friends with",
            "tail_type": "Person",
            "tail": "Bob",
        }
        expected_output = (
            "Alice which is a Person is friends with Bob, a Person"
        )
        assert (
            convert_triple_to_text(triple, include_chunks=False)
            == expected_output
        )

    def test_triple_with_properties(self):
        triple = {
            "head_type": "Person",
            "head": "Alice",
            "relation": "is friends with",
            "relation_properties": {"since": "2020"},
            "tail_type": "Person",
            "tail": "Bob",
            "head_properties": {
                "age_years": 25,
                "occupation_role": "Software_Engineer",
            },
            "tail_properties": {"age_years": 28},
        }
        expected_output = (
            "Alice which is a Person with age years of 25, occupation role of Software Engineer "
            "is friends with Bob, a Person with age years of 28 due to since of 2020"
        )
        assert (
            convert_triple_to_text(triple, include_chunks=False)
            == expected_output
        )

    def test_triple_with_special_characters_in_keys(self):
        triple = {
            "head_type": "Person",
            "head": "Charlie",
            "relation": "works with",
            "tail_type": "Person",
            "tail": "Dana",
            "head_properties": {
                "age-years": 30,
                "occupation-role": "Dev-Engineer",
            },
            "tail_properties": {"project_period": "2020-2021"},
        }
        expected_output = (
            "Charlie which is a Person with age years of 30, occupation role of Dev Engineer "
            "works with Dana, a Person with project period of 2020 2021"
        )
        assert (
            convert_triple_to_text(triple, include_chunks=False)
            == expected_output
        )

    def test_triple_with_missing_properties(self):
        triple = {
            "head_type": "Person",
            "head": "Eve",
            "relation": "is related to",
            "tail_type": "Person",
            "tail": "Frank",
        }
        expected_output = "Eve which is a Person is related to Frank, a Person"
        assert (
            convert_triple_to_text(triple, include_chunks=False)
            == expected_output
        )

    def test_triple_with_text_chunks(self):
        triple = {
            "head_type": "Person",
            "head": "Grace",
            "relation": "collaborates with",
            "tail_type": "Person",
            "tail": "Heidi",
            "chunks_content": [
                {"content": "They worked on project X together."}
            ],
        }
        expected_output = "Grace which is a Person collaborates with Heidi, a Person. This is further explained by the chunks: They worked on project X together."
        assert (
            convert_triple_to_text(triple, include_chunks=True)
            == expected_output
        )

    def test_triple_with_structured_chunks(self):
        triple = {
            "head_type": "Person",
            "head": "Ivan",
            "relation": "mentors",
            "tail_type": "Person",
            "tail": "Judy",
            "chunks_content": [
                {
                    "content": {
                        "context": "Mentorship program",
                        "duration": "1 year",
                    }
                }
            ],
        }
        expected_output = "Ivan which is a Person mentors Judy, a Person. This is further explained by the chunks: context: Mentorship program, duration: 1 year"
        assert (
            convert_triple_to_text(triple, include_chunks=True)
            == expected_output
        )


class TestExtractPropertiesFromFields:
    def test_empty_fields_list(self):
        """Test that an empty list of fields returns an empty string."""
        result = extract_properties_from_fields([])
        assert result == "", "Expected an empty string for empty fields list"

    def test_fields_with_no_properties(self):
        """Test that fields with no properties are skipped."""
        fields = [EntityField(name="field 1"), EntityField(name="field 2")]
        result = extract_properties_from_fields(fields)
        assert result == "", "Fields with no properties should be skipped"

    def test_single_property_field(self):
        """Test correct handling of a field with a single property."""
        fields = [EntityField(name="field 1", properties=["color"])]
        result = extract_properties_from_fields(fields)
        assert (
            result == "color"
        ), "Single property should be returned correctly"

    def test_multiple_properties_field(self):
        """Test formatting for a field with multiple properties."""
        fields = [
            EntityField(name="field 1", properties=["color", "size", "shape"])
        ]
        result = extract_properties_from_fields(fields)
        assert (
            result == "color, size, and shape"
        ), "Multiple properties should be formatted correctly"

    def test_combination_of_various_fields(self):
        """Test with a mix of fields with different numbers of properties."""
        fields = [
            EntityField(name="field 1", properties=["color"]),
            EntityField(name="field 2", properties=["height", "width"]),
            EntityField(name="field 3", properties=[]),
            EntityField(
                name="field 4", properties=["texture", "material", "type"]
            ),
        ]
        result = extract_properties_from_fields(fields)
        assert (
            result == "color, height, and width, texture, material, and type"
        ), "Combination of fields should be formatted correctly"


class TestPatternToText:
    def test_basic_pattern(self):
        pattern = SchemaTriplePattern(
            head=SchemaEntity(
                name="person",
                description="An individual who plays a pivotal role in leading an organization.",
            ),
            relation=SchemaRelation(
                name="runs",
                description="A relationship where an individual leads or manages an organization.",
            ),
            tail=SchemaEntity(
                name="company",
                description="An organizational entity that engages in commercial, industrial, or professional activities.",
            ),
            description="Indicates that a person holds a leadership or managerial role within a company.",
        )
        expected_output = "The person (an individual who plays a pivotal role in leading an organization.) runs the company (an organizational entity that engages in commercial, industrial, or professional activities.)"
        assert convert_pattern_to_text(pattern) == expected_output

    def test_pattern_with_properties(self):
        pattern = SchemaTriplePattern(
            head=SchemaEntity(
                name="person",
                description="An individual who plays a pivotal role in leading an organization.",
                fields=[
                    EntityField(
                        name="CEO",
                        properties=["CEO Age", "CEO Length with Business"],
                    )
                ],
            ),
            relation=SchemaRelation(
                name="runs",
                description="A relationship where an individual leads or manages an organization.",
            ),
            tail=SchemaEntity(
                name="company",
                description="An organizational entity that engages in commercial, industrial, or professional activities.",
                fields=[
                    EntityField(
                        name="Company",
                        properties=["Location", "Number of Employees"],
                    )
                ],
            ),
            description="Indicates that a person holds a leadership or managerial role within a company.",
        )
        expected_output = "The person (an individual who plays a pivotal role in leading an organization.) with properties CEO Age, and CEO Length with Business, runs the company (an organizational entity that engages in commercial, industrial, or professional activities.) with properties Location, and Number of Employees"
        assert convert_pattern_to_text(pattern) == expected_output


@pytest.mark.asyncio
async def test_get_and_separate_chunks_on_data_type_success():
    mock_chunk_ids = [ObjectId(), ObjectId()]
    mock_node_response = [
        ChunkDocumentModel(
            _id=ObjectId(),
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
        ).model_dump(),
        ChunkDocumentModel(
            _id=ObjectId(),
            content={"key": "value"},
            data_type="object",
            created_by=ObjectId(),
            metadata=ChunkMetadata(
                language="en",
                size=10,
                data_source_type="manual",
            ),
            workspaces=[ObjectId()],
            tags={},
            user_metadata={},
        ).model_dump(),
    ]

    mock_cursor = AsyncMock()
    mock_cursor.to_list.return_value = mock_node_response

    # Create a mock database
    mock_collection = MagicMock()
    mock_collection.find.return_value = mock_cursor

    # Call the function
    chunks_dict = await get_and_separate_chunks_on_data_type(
        collection=mock_collection, chunk_ids=mock_chunk_ids
    )

    # Assert
    assert set(chunks_dict.keys()) - {"string", "object"} == set()
    assert len(chunks_dict["string"]) == 1
    assert len(chunks_dict["object"]) == 1

    mock_collection.find.assert_called_once_with(
        {"_id": {"$in": mock_chunk_ids}}, {"embedding": 0}
    )


@pytest.mark.asyncio
async def test_get_and_separate_chunks_on_data_type_no_chunk_ids():
    mock_chunk_ids = []
    mock_node_response = []

    mock_cursor = AsyncMock()
    mock_cursor.to_list.return_value = mock_node_response

    # Create a mock database
    mock_collection = MagicMock()
    mock_collection.find.return_value = mock_cursor

    # Call the function
    chunks_dict = await get_and_separate_chunks_on_data_type(
        collection=mock_collection, chunk_ids=mock_chunk_ids
    )

    # Assert
    assert chunks_dict == {}

    mock_collection.find.assert_called_once_with(
        {"_id": {"$in": mock_chunk_ids}}, {"embedding": 0}
    )


@pytest.mark.asyncio
async def test_create_node_id_map():
    # Mock database and cursor
    db = MagicMock()
    mock_cursor = AsyncMock()

    example_nodes = [
        {"name": "Node1", "type": "TypeA", "_id": ObjectId()},
        {"name": "Node2", "type": "TypeB", "_id": ObjectId()},
    ]

    # Set up async iteration on the mock cursor
    mock_cursor.__aiter__.return_value = (node for node in example_nodes)

    # Patch the db.node.find to return the mock cursor
    db.node.find = MagicMock(return_value=mock_cursor)

    # Call the function with mock data
    node_names = {"Node1", "Node2"}
    node_types = {"TypeA", "TypeB"}
    graph_id = ObjectId()
    user_id = ObjectId()
    node_id_map = await create_node_id_map(
        db, node_names, node_types, graph_id, user_id
    )

    # Check the results
    assert len(node_id_map) == 2
    assert (example_nodes[0]["name"], example_nodes[0]["type"]) in node_id_map
    assert (
        node_id_map[(example_nodes[0]["name"], example_nodes[0]["type"])]
        == example_nodes[0]["_id"]
    )


@pytest.mark.asyncio
async def test_embed_triples(monkeypatch):
    # Mock the LLMClient
    mock_llm_client = MagicMock()
    mock_llm_client.metadata = MagicMock(
        embedding_name="text-embedding-3-small"
    )
    mock_response = MagicMock()
    mock_response.data = [
        MagicMock(embedding=[0.1, 0.2, 0.3]) for _ in range(5)
    ]  # Amount of embeddings returned _each_ call

    # Set the async method return value
    mock_llm_client.client.embeddings.create = AsyncMock(
        return_value=mock_response
    )

    # Mock the triple to text conversion function if needed
    triples = [{"subject": "S1", "predicate": "P1", "object": "O1"}] * 10
    with monkeypatch.context() as m:
        m.setattr(
            "whyhow_api.services.crud.triple.convert_triple_to_text",
            lambda triple, include_chunks: "natural language variation of triple",
        )

        # Execute the function
        embeddings = await embed_triples(
            mock_llm_client, triples, batch_size=5
        )

        # Assertions
        assert len(embeddings) == 10, "Should have embeddings for each triple"
        assert all(
            len(embed) == 3 for embed in embeddings
        ), "Each embedding should have three dimensions"
        mock_llm_client.client.embeddings.create.assert_called()  # Ensure the API was called
        assert (
            mock_llm_client.client.embeddings.create.call_count == 2
        ), "Should be called twice for two batches"


@pytest.mark.asyncio
async def test_embed_triples_no_triples():

    embeddings = await embed_triples(
        llm_client=AsyncMock(), triples=[], batch_size=1
    )

    assert embeddings == []


def test_create_structured_patterns_no_fields():

    mock_patterns = [
        SchemaTriplePattern(
            head=SchemaEntity(
                name="person",
                description="An individual who plays a pivotal role in leading an organization.",
            ),
            relation=SchemaRelation(
                name="runs",
                description="A relationship where an individual leads or manages an organization.",
            ),
            tail=SchemaEntity(
                name="company",
                description="An organizational entity that engages in commercial, industrial, or professional activities.",
            ),
            description="Indicates that a person holds a leadership or managerial role within a company.",
        )
    ]

    structured_patterns = create_structured_patterns(patterns=mock_patterns)

    assert len(structured_patterns) == 0


def test_create_structured_patterns_fields():
    mock_patterns = [
        SchemaTriplePattern(
            head=SchemaEntity(
                name="person",
                description="An individual who plays a pivotal role in leading an organization.",
                fields=[
                    EntityField(
                        name="CEO",
                        properties=["CEO Age", "CEO Length with Business"],
                    ),
                    EntityField(name="CFO", properties=["CFO Age"]),
                ],
            ),
            relation=SchemaRelation(
                name="runs",
                description="A relationship where an individual leads or manages an organization.",
            ),
            tail=SchemaEntity(
                name="company",
                description="An organizational entity that engages in commercial, industrial, or professional activities.",
                fields=[
                    EntityField(
                        name="Company",
                        properties=["Location", "Number of Employees"],
                    ),
                    EntityField(name="Business", properties=[]),
                ],
            ),
            description="Indicates that a person holds a leadership or managerial role within a company.",
        )
    ]

    structured_patterns = create_structured_patterns(patterns=mock_patterns)

    assert len(structured_patterns) == 4


def test_extract_structured_graph_triples_success():
    mock_patterns = [
        SchemaTriplePattern(
            head=SchemaEntity(
                name="person",
                description="An individual who plays a pivotal role in leading an organization.",
                fields=[
                    EntityField(
                        name="CEO",
                        properties=["CEO Age", "CEO Length with Business"],
                    ),
                    EntityField(name="CFO", properties=["CFO Age"]),
                ],
            ),
            relation=SchemaRelation(
                name="runs",
                description="A relationship where an individual leads or manages an organization.",
            ),
            tail=SchemaEntity(
                name="company",
                description="An organizational entity that engages in commercial, industrial, or professional activities.",
                fields=[
                    EntityField(
                        name="Company",
                        properties=["Location", "Number of Employees"],
                    ),
                    EntityField(name="Business", properties=[]),
                ],
            ),
            description="Indicates that a person holds a leadership or managerial role within a company.",
        )
    ]
    mock_chunks = [
        ChunkDocumentModel(
            _id=ObjectId(),
            content={
                "CEO": "Alice",
                "CEO Age": 30,
                "CEO Length with Business": 5,
                "Company": "Company A",
                "Location": "Location A",
                "Number of Employees": 100,
                "Business": "Business A",
                "CFO": "Bob",
                "CFO Age": 35,
            },
            data_type="object",
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

    structured_triples = extract_structured_graph_triples(
        patterns=mock_patterns, chunks=mock_chunks
    )

    assert len(structured_triples) == 4


def test_extract_structured_graph_triples_success_missing_data():
    mock_patterns = [
        SchemaTriplePattern(
            head=SchemaEntity(
                name="person",
                description="An individual who plays a pivotal role in leading an organization.",
                fields=[
                    EntityField(
                        name="CEO",
                        properties=["CEO Age", "CEO Length with Business"],
                    ),
                    EntityField(name="CFO", properties=["CFO Age"]),
                ],
            ),
            relation=SchemaRelation(
                name="runs",
                description="A relationship where an individual leads or manages an organization.",
            ),
            tail=SchemaEntity(
                name="company",
                description="An organizational entity that engages in commercial, industrial, or professional activities.",
                fields=[
                    EntityField(
                        name="Company",
                        properties=["Location", "Number of Employees"],
                    ),
                    EntityField(name="Business", properties=[]),
                ],
            ),
            description="Indicates that a person holds a leadership or managerial role within a company.",
        )
    ]
    mock_chunks = [
        ChunkDocumentModel(
            _id=ObjectId(),
            content={
                "boss": "Alice",
                "company_name": "Company A",
                "Location": "Location A",
                "Number of Employees": 100,
                "Business": "Business A",
                "CFO": "Bob",
                "CFO Age": 35,
            },
            data_type="object",
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

    structured_triples = extract_structured_graph_triples(
        patterns=mock_patterns, chunks=mock_chunks
    )

    assert len(structured_triples) == 1


def test_extract_structured_graph_triples_string_chunk():
    mock_patterns = [
        SchemaTriplePattern(
            head=SchemaEntity(
                name="person",
                description="An individual who plays a pivotal role in leading an organization.",
                fields=[
                    EntityField(
                        name="CEO",
                        properties=["CEO Age", "CEO Length with Business"],
                    ),
                    EntityField(name="CFO", properties=["CFO Age"]),
                ],
            ),
            relation=SchemaRelation(
                name="runs",
                description="A relationship where an individual leads or manages an organization.",
            ),
            tail=SchemaEntity(
                name="company",
                description="An organizational entity that engages in commercial, industrial, or professional activities.",
                fields=[
                    EntityField(
                        name="Company",
                        properties=["Location", "Number of Employees"],
                    ),
                    EntityField(name="Business", properties=[]),
                ],
            ),
            description="Indicates that a person holds a leadership or managerial role within a company.",
        )
    ]
    mock_chunks = [
        ChunkDocumentModel(
            _id=ObjectId(),
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

    structured_triples = extract_structured_graph_triples(
        patterns=mock_patterns, chunks=mock_chunks
    )

    assert len(structured_triples) == 0


@pytest.mark.asyncio
class TestMixedQuery:

    @pytest.fixture
    def llm_client(self):
        client_mock = MagicMock()
        client_mock.metadata = MagicMock(
            embedding_name="text-embedding-3-small"
        )
        client_mock.client.embeddings.create = AsyncMock(
            return_value=MagicMock(data=[MagicMock(embedding=[0.1] * 1024)])
        )
        return client_mock

    @pytest.fixture
    def db(self):
        db_mock = MagicMock()
        db_mock.triple = MagicMock()
        db_mock.triple.aggregate = AsyncMock(
            return_value=MagicMock(to_list=AsyncMock(return_value=[]))
        )
        return db_mock

    @pytest.fixture
    def graph_id(self):
        return ObjectId()

    @pytest.fixture
    def user_id(self):
        return ObjectId()

    @pytest.fixture
    def workspace_id(self):
        return ObjectId()

    @pytest.fixture
    def schema_id(self):
        return ObjectId()

    async def test_init(
        self, db, llm_client, graph_id, user_id, workspace_id, schema_id
    ):

        settings_mock = MagicMock()
        settings_mock.api.query_sim_triple_limit.return_value = 128
        settings_mock.api.query_sim_triple_candidates.return_value = 256

        nl_query = MixedQueryProcessor(
            db=db,
            graph_id=graph_id,
            user_id=user_id,
            workspace_id=workspace_id,
            llm_client=llm_client,
            settings=settings_mock,
            schema_id=schema_id,
        )

        assert nl_query.db is not None
        assert nl_query.user_id is not None
        assert nl_query.graph_id is not None
        assert nl_query.llm_client is not None
        assert nl_query.settings is not None
        assert nl_query.schema_id is not None


@pytest.mark.asyncio
async def test_apply_rules(monkeypatch):
    db = MagicMock()
    extracted_triples = [
        Triple(
            head="test_head",
            tail="test_tail",
            head_type="test_type",
            tail_type="test_type",
            relation="test_relation",
        )
    ]
    workspace_id = ObjectId()
    graph_id = ObjectId()
    user_id = ObjectId()
    errors = []

    rules = [
        {
            "_id": ObjectId(),
            "workspace": workspace_id,
            "created_by": user_id,
            "rule": {
                "rule_type": "merge_nodes",
                "from_node_names": ["test_head"],
                "to_node_name": "test_to_node",
                "node_type": "test_type",
            },
        }
    ]

    fake_find_to_list = AsyncMock(return_value=rules)
    db.rule.find = MagicMock()
    db.rule.find.return_value.to_list = fake_find_to_list

    db.graph.find_one = AsyncMock(
        return_value={"_id": graph_id, "rules": rules}
    )

    fake_apply_rules_to_triples = MagicMock(return_value=extracted_triples)
    monkeypatch.setattr(
        "whyhow_api.services.graph_service.apply_rules_to_triples",
        fake_apply_rules_to_triples,
    )

    result = await apply_rules(
        db, extracted_triples, workspace_id, graph_id, user_id, errors
    )

    assert result == extracted_triples
    assert len(errors) == 0

    db.rule.find.assert_called_once_with(
        {"workspace": workspace_id, "created_by": user_id}
    )
    fake_find_to_list.assert_awaited_once()
    fake_apply_rules_to_triples.assert_called_once()
    db.graph.update_one.assert_not_called()
