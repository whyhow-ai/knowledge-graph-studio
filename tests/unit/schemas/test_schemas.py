import pytest
from bson import ObjectId
from pydantic import ValidationError

from whyhow_api.models.common import (
    SchemaEntity,
    SchemaRelation,
    TriplePattern,
)
from whyhow_api.schemas.schemas import (
    GeneratedSchema,
    GenerateSchemaBody,
    SchemaCreate,
    SchemaUpdate,
)


class TestSchemaUpdate:
    def test_default_value(self):
        """Test that the default value of the name field is None."""
        schema_update = SchemaUpdate()
        assert schema_update.name is None

    def test_type_acceptance(self):
        """Test that the name field accepts both str and None types."""
        schema_update_str = SchemaUpdate(name="TestName")
        assert schema_update_str.name == "TestName"

        schema_update_none = SchemaUpdate(name=None)
        assert schema_update_none.name is None

    def test_serialization_deserialization(self):
        """Test model serialization and deserialization."""
        schema_update = SchemaUpdate(name="TestName")
        serialized = schema_update.model_dump_json()
        deserialized = SchemaUpdate.model_validate_json(serialized)
        assert deserialized == schema_update


class TestSchemaCreate:
    """Test SchemaCreate schema."""

    def test_default_values(self):
        schema = SchemaCreate(
            name="Test Schema",
            workspace=str(ObjectId()),
            entities=[
                SchemaEntity(
                    name="Entity1", description="Entity1 description"
                ),
                SchemaEntity(
                    name="Entity2", description="Entity2 description"
                ),
            ],
            relations=[
                SchemaRelation(name="Relation", description="Description")
            ],
            patterns=[
                TriplePattern(
                    head="Entity1",
                    relation="Relation",
                    tail="Entity2",
                    description="Pattern1",
                )
            ],
        )

        assert schema.name == "Test Schema"
        assert isinstance(schema.workspace, ObjectId)
        assert isinstance(schema.entities, list)
        assert isinstance(schema.relations, list)
        assert isinstance(schema.patterns, list)

    def test_with_fields(self):
        schema = SchemaCreate(
            name="Test Schema",
            workspace=str(ObjectId()),
            entities=[
                SchemaEntity(
                    name="Entity1",
                    description="Entity1 description",
                    fields=[
                        {"name": "Field1", "properties": ["prop1", "prop2"]},
                        {"name": "Field2", "properties": ["prop3", "prop4"]},
                    ],
                ),
                SchemaEntity(
                    name="Entity2",
                    description="Entity2 description",
                    fields=[
                        {"name": "Field3", "properties": ["prop5", "prop6"]},
                        {"name": "Field4", "properties": ["prop7", "prop8"]},
                    ],
                ),
            ],
            relations=[
                SchemaRelation(name="Relation", description="Description")
            ],
            patterns=[
                TriplePattern(
                    head="Entity1",
                    relation="Relation",
                    tail="Entity2",
                    description="Pattern1",
                )
            ],
        )

        assert schema.name == "Test Schema"
        assert isinstance(schema.workspace, ObjectId)
        assert isinstance(schema.entities, list)
        assert schema.entities[0].fields[0].name == "Field1"
        assert schema.entities[0].fields[0].properties == ["prop1", "prop2"]
        assert schema.entities[0].fields[1].name == "Field2"
        assert schema.entities[0].fields[1].properties == ["prop3", "prop4"]
        assert schema.entities[1].fields[0].name == "Field3"
        assert schema.entities[1].fields[0].properties == ["prop5", "prop6"]
        assert schema.entities[1].fields[1].name == "Field4"
        assert schema.entities[1].fields[1].properties == ["prop7", "prop8"]
        assert isinstance(schema.relations, list)
        assert isinstance(schema.patterns, list)

    def test_sizes_validation(self):
        """Test the size validation for entities, relations, and patterns."""
        with pytest.raises(
            ValueError, match="At least one entity must be supplied."
        ):
            SchemaCreate(
                name="Test Schema",
                workspace=ObjectId(),
                entities=[],
                relations=[
                    SchemaRelation(name="Relation", description="Description")
                ],
                patterns=[
                    TriplePattern(
                        head="Entity1",
                        relation="Relation",
                        tail="Entity1",
                        description="Pattern1",
                    )
                ],
            )

        with pytest.raises(
            ValueError, match="At least one relation must be supplied."
        ):
            SchemaCreate(
                name="Test Schema",
                workspace=ObjectId(),
                entities=[
                    SchemaEntity(
                        name="Entity1", description="Entity1 description"
                    ),
                    SchemaEntity(
                        name="Entity2", description="Entity2 description"
                    ),
                ],
                relations=[],  # No relations
                patterns=[
                    TriplePattern(
                        head="Entity1",
                        relation="Relation",
                        tail="Entity2",
                        description="Pattern1",
                    )
                ],
            )

        with pytest.raises(
            ValueError, match="At least one pattern must be supplied."
        ):
            SchemaCreate(
                name="Test Schema",
                workspace=ObjectId(),
                entities=[
                    SchemaEntity(
                        name="Entity1", description="Entity1 description"
                    ),
                    SchemaEntity(
                        name="Entity2", description="Entity2 description"
                    ),
                ],
                relations=[
                    SchemaRelation(name="Relation", description="Description")
                ],
                patterns=[],  # No patterns
            )

    def test_update_patterns_error(self):
        """Test that patterns are correctly updated."""
        with pytest.raises(ValidationError):
            SchemaCreate(
                name="Test Schema",
                workspace=ObjectId(),
                entities=[
                    SchemaEntity(
                        name="Entity1", description="Entity1 description"
                    ),
                    SchemaEntity(
                        name="Entity2", description="Entity2 description"
                    ),
                ],
                relations=[
                    SchemaRelation(name="Relation", description="Description")
                ],
                patterns=[
                    TriplePattern(
                        head="Entity1",
                        relation="Relation",
                        tail="Entity3",
                        description="Pattern1",
                    )
                ],
            )

        with pytest.raises(ValidationError):
            SchemaCreate(
                name="Test Schema",
                workspace=ObjectId(),
                entities=[
                    SchemaEntity(
                        name="Entity1", description="Entity1 description"
                    ),
                    SchemaEntity(
                        name="Entity2", description="Entity2 description"
                    ),
                ],
                relations=[
                    SchemaRelation(name="Relation", description="Description")
                ],
                patterns=[
                    TriplePattern(
                        head="Entity1",
                        relation="Relation2",
                        tail="Entity2",
                        description="Pattern1",
                    )
                ],
            )

    @pytest.mark.parametrize(
        "name, workspace, entities, relations, patterns",
        [
            (None, None, None, None, None),  # All fields are None
            (
                1234,
                ObjectId(),
                [
                    SchemaEntity(
                        name="Entity1", description="Entity1 description"
                    ),
                    SchemaEntity(
                        name="Entity2", description="Entity2 description"
                    ),
                ],
                [SchemaRelation(name="Relation", description="Description")],
                [
                    TriplePattern(
                        head="Entity1",
                        relation="Relation",
                        tail="Entity2",
                        description="Pattern1",
                    )
                ],
            ),  # Invalid name
            (
                "tests",
                1234,
                [
                    SchemaEntity(
                        name="Entity1", description="Entity1 description"
                    ),
                    SchemaEntity(
                        name="Entity2", description="Entity2 description"
                    ),
                ],
                [SchemaRelation(name="Relation", description="Description")],
                [
                    TriplePattern(
                        head="Entity1",
                        relation="Relation",
                        tail="Entity2",
                        description="Pattern1",
                    )
                ],
            ),  # Invalid workspace
            (
                "tests",
                ObjectId(),
                [
                    SchemaEntity(
                        name="Entity2", description="Entity2 description"
                    ),
                ],
                [SchemaRelation(name="Relation", description="Description")],
                [
                    TriplePattern(
                        head="Entity1",
                        relation="Relation",
                        tail="Entity2",
                        description="Pattern1",
                    )
                ],
            ),  # Missing entity
            (
                "tests",
                ObjectId(),
                [
                    SchemaEntity(
                        name="Entity1", description="Entity1 description"
                    ),
                    SchemaEntity(
                        name="Entity2", description="Entity2 description"
                    ),
                ],
                [],
                [
                    TriplePattern(
                        head="Entity1",
                        relation="Relation",
                        tail="Entity2",
                        description="Pattern1",
                    )
                ],
            ),  # Missing relation
            (
                "tests",
                ObjectId(),
                [
                    SchemaEntity(
                        name="Entity1", description="Entity1 description"
                    ),
                    SchemaEntity(
                        name="Entity2", description="Entity2 description"
                    ),
                ],
                [SchemaRelation(name="Relation", description="Description")],
                [],
            ),  # Missing triple
        ],
    )
    def test_validation_errors(
        self, name, workspace, entities, relations, patterns
    ):
        with pytest.raises(ValidationError):
            SchemaCreate(
                name=name,
                workspace=workspace,
                entities=entities,
                relations=relations,
                patterns=patterns,
            )


class TestGenerateSchemaBody:
    def test_field_requirements_and_defaults(self):
        """Test field requirements and default values."""
        # Assuming AfterAnnotatedObjectId can be instantiated with a string for testing
        workspace_id = ObjectId()
        questions = ["What is the meaning of life?"]

        schema_body = GenerateSchemaBody(
            workspace=workspace_id, questions=questions
        )

        assert schema_body.workspace == workspace_id
        assert schema_body.questions == questions

    def test_type_acceptance(self):
        """Test that fields accept correct types."""
        workspace_id = ObjectId()
        questions = ["What is the meaning of life?"]

        schema_body = GenerateSchemaBody(
            workspace=workspace_id, questions=questions
        )

        assert isinstance(schema_body.workspace, ObjectId)
        assert isinstance(schema_body.questions, list)
        assert all(isinstance(q, str) for q in schema_body.questions)

    def test_custom_validation(self):
        """Test the custom validation method."""
        workspace_id = ObjectId()
        questions = ["", " ", "What is the meaning of life?"]

        schema_body = GenerateSchemaBody(
            workspace=workspace_id, questions=questions
        )
        assert schema_body.questions == [
            "What is the meaning of life?"
        ]  # Filtered questions

        with pytest.raises(
            ValueError, match="At least one question must be provided."
        ):
            GenerateSchemaBody(workspace=workspace_id, questions=["", " "])

    def test_arbitrary_types_allowed(self):
        """Test that arbitrary types are allowed by the model configuration."""
        workspace_id = ObjectId()
        questions = ["What is the meaning of life?"]
        # Directly testing model configuration is not typical, but we can ensure it behaves as expected
        try:
            GenerateSchemaBody(workspace=workspace_id, questions=questions)
            passed = True
        except ValidationError:
            passed = False
        assert passed


class TestGeneratedSchema:
    def test_validate_patterns_success(self):
        schema = GeneratedSchema(
            entities=[
                SchemaEntity(
                    name="Entity1", description="Entity1 description"
                ),
                SchemaEntity(
                    name="Entity2", description="Entity2 description"
                ),
            ],
            relations=[
                SchemaRelation(
                    name="Relation1", description="Relation1 description"
                ),
            ],
            patterns=[
                TriplePattern(
                    head="Entity1",
                    relation="Relation1",
                    tail="Entity2",
                    description="Pattern1",
                ),
            ],
        )
        assert schema.validate_patterns() == schema

    def test_validate_patterns_fail_head_not_found(self):
        with pytest.raises(ValueError) as exc_info:
            GeneratedSchema(
                entities=[
                    SchemaEntity(
                        name="Entity2", description="Entity2 description"
                    ),
                ],
                relations=[
                    SchemaRelation(
                        name="Relation1", description="Relation1 description"
                    ),
                ],
                patterns=[
                    TriplePattern(
                        head="Entity1",
                        relation="Relation1",
                        tail="Entity2",
                        description="Pattern1",
                    ),
                ],
            ).validate_patterns()
        assert "Pattern head 'Entity1' not found in entities." in str(
            exc_info.value
        )

    def test_validate_patterns_fail_tail_not_found(self):
        with pytest.raises(ValueError) as exc_info:
            GeneratedSchema(
                entities=[
                    SchemaEntity(
                        name="Entity1", description="Entity1 description"
                    ),
                ],
                relations=[
                    SchemaRelation(
                        name="Relation1", description="Relation1 description"
                    ),
                ],
                patterns=[
                    TriplePattern(
                        head="Entity1",
                        relation="Relation1",
                        tail="Entity2",
                        description="Pattern1",
                    ),
                ],
            ).validate_patterns()
        assert "Pattern tail 'Entity2' not found in entities." in str(
            exc_info.value
        )

    def test_validate_patterns_fail_relation_not_found(self):
        with pytest.raises(ValueError) as exc_info:
            GeneratedSchema(
                entities=[
                    SchemaEntity(
                        name="Entity1", description="Entity1 description"
                    ),
                    SchemaEntity(
                        name="Entity2", description="Entity2 description"
                    ),
                ],
                relations=[
                    SchemaRelation(name="Relation2", description="Relation2"),
                ],
                patterns=[
                    TriplePattern(
                        head="Entity1",
                        relation="Relation1",
                        tail="Entity2",
                        description="Pattern1",
                    ),
                ],
            ).validate_patterns()
        assert "Pattern relation 'Relation1' not found in relations." in str(
            exc_info.value
        )
