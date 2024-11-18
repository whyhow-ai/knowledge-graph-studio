import pytest
from pydantic import ValidationError

from whyhow_api.models.common import (
    DatasetModel,
    Entity,
    EntityField,
    Node,
    PDFProcessorConfig,
    Relation,
    Schema,
    SchemaEntity,
    SchemaRelation,
    SchemaTriplePattern,
    StructuredSchemaEntity,
    StructuredSchemaTriplePattern,
    Triple,
    TriplePattern,
)


class TestNode:
    def test_create_node(self):
        node = Node(
            name="Python",
            label="Programming Language",
            properties={"version": "3.8"},
        )
        assert node.name == "Python"
        assert node.label == "Programming Language"
        assert node.properties == {"version": "3.8"}

    def test_create_node_without_label(self):
        node = Node(name="Python", properties={"version": "3.8"})
        assert node.name == "Python"
        assert node.label is None
        assert node.properties == {"version": "3.8"}

    def test_create_node_without_properties(self):
        node = Node(name="Python", label="Programming Language")
        assert node.name == "Python"
        assert node.label == "Programming Language"
        assert node.properties == {}

    def test_create_node_with_invalid_name(self):
        with pytest.raises(ValidationError):
            Node(
                name=123,
                label="Programming Language",
                properties={"version": "3.8"},
            )


class TestRelation:
    def test_create_relation(self):
        start_node = Node(
            name="Python",
            label="Programming Language",
            properties={"version": "3.8"},
        )
        end_node = Node(
            name="Java",
            label="Programming Language",
            properties={"version": "11"},
        )
        relation = Relation(
            label="similar_to",
            start_node=start_node,
            end_node=end_node,
            properties={"reason": "Both are programming languages"},
        )
        assert relation.label == "similar_to"
        assert relation.start_node == start_node
        assert relation.end_node == end_node
        assert relation.properties == {
            "reason": "Both are programming languages"
        }

    def test_create_relation_without_properties(self):
        start_node = Node(
            name="Python",
            label="Programming Language",
            properties={"version": "3.8"},
        )
        end_node = Node(
            name="Java",
            label="Programming Language",
            properties={"version": "11"},
        )
        relation = Relation(
            label="similar_to", start_node=start_node, end_node=end_node
        )
        assert relation.label == "similar_to"
        assert relation.start_node == start_node
        assert relation.end_node == end_node
        assert relation.properties == {}

    def test_create_relation_with_invalid_label(self):
        start_node = Node(
            name="Python",
            label="Programming Language",
            properties={"version": "3.8"},
        )
        end_node = Node(
            name="Java",
            label="Programming Language",
            properties={"version": "11"},
        )
        with pytest.raises(ValidationError):
            Relation(
                label=123,
                start_node=start_node,
                end_node=end_node,
                properties={"reason": "Both are programming languages"},
            )


class TestEntity:
    def test_create_entity(self):
        entity = Entity(
            text="Python",
            label="Programming Language",
            properties={"version": "3.8"},
        )
        assert entity.text == "Python"
        assert entity.label == "Programming Language"
        assert entity.properties == {"version": "3.8"}

    def test_create_entity_without_label(self):
        entity = Entity(text="Python", properties={"version": "3.8"})
        assert entity.text == "Python"
        assert entity.label is None
        assert entity.properties == {"version": "3.8"}

    def test_create_entity_without_properties(self):
        entity = Entity(text="Python", label="Programming Language")
        assert entity.text == "Python"
        assert entity.label == "Programming Language"
        assert entity.properties == {}

    def test_create_entity_with_invalid_text(self):
        with pytest.raises(ValidationError):
            Entity(
                text=123,
                label="Programming Language",
                properties={"version": "3.8"},
            )


class TestTriple:
    def test_create_triple(self):
        triple = Triple(
            head="Python",
            head_type="Entity",
            relation="is a",
            tail="Programming Language",
            tail_type="Entity",
            head_properties={"version": "3.8"},
            relation_properties={},
            tail_properties={},
        )
        assert triple.head == "Python"
        assert triple.head_type == "Entity"
        assert triple.relation == "is a"
        assert triple.tail == "Programming Language"
        assert triple.tail_type == "Entity"
        assert triple.head_properties == {"version": "3.8"}
        assert triple.relation_properties == {}
        assert triple.tail_properties == {}

    def test_triple_str(self):
        triple = Triple(
            head="Python",
            head_type="Entity",
            relation="is a",
            tail="Programming Language",
            tail_type="Entity",
            head_properties={"version": "3.8"},
            relation_properties={},
            tail_properties={},
        )
        assert isinstance(triple.__str__(), str)

    def test_create_triple_with_invalid_head(self):
        with pytest.raises(ValidationError):
            Triple(
                head=123,
                head_type="Entity",
                relation="is a",
                tail="Programming Language",
                tail_type="Entity",
                head_properties={"version": "3.8"},
                relation_properties={},
                tail_properties={},
            )

    def test_empty_head_raises_validation_error(self):
        with pytest.raises(ValidationError):
            Triple(
                head="",
                head_type="Entity",
                relation="is a",
                tail="Programming Language",
                tail_type="Entity",
                head_properties={"version": "3.8"},
                relation_properties={},
                tail_properties={},
            )

    def test_empty_tail_raises_validation_error(self):
        with pytest.raises(ValidationError):
            Triple(
                head="Python",
                head_type="Entity",
                relation="is a",
                tail="",
                tail_type="Entity",
                head_properties={"version": "3.8"},
                relation_properties={},
                tail_properties={},
            )


class TestEntityField:
    def test_entity_field_initialization(self):
        """Test the initialization of EntityField."""
        field = EntityField(name="TestField", properties=["prop1", "prop2"])
        assert field.name == "TestField"
        assert field.properties == ["prop1", "prop2"]

    def test_entity_field_default_properties(self):
        """Test the default properties list of EntityField."""
        field = EntityField(name="TestField")
        assert field.properties == []

    def test_entity_field_properties_append(self):
        """Test appending to the properties list of EntityField."""
        field = EntityField(name="TestField")
        field.properties.append("prop1")
        assert "prop1" in field.properties


class TestSchemaEntity:
    def test_create_schema_entity(self):
        schema_entity = SchemaEntity(
            name="EntityName", description="EntityDescription"
        )
        assert schema_entity.name == "EntityName"
        assert schema_entity.description == "EntityDescription"

    def test_create_schema_entity_with_invalid_name(self):
        with pytest.raises(ValidationError):
            SchemaEntity(name=123, description="EntityDescription")

    def test_create_schema_entity_with_invalid_description(self):
        with pytest.raises(ValidationError):
            SchemaEntity(name="EntityName", description=123)


class TestSchemaRelation:
    def test_create_schema_relation(self):
        schema_relation = SchemaRelation(
            name="RelationName", description="RelationDescription"
        )
        assert schema_relation.name == "RelationName"
        assert schema_relation.description == "RelationDescription"

    def test_create_schema_relation_with_invalid_name(self):
        with pytest.raises(ValidationError):
            SchemaRelation(name=123, description="RelationDescription")

    def test_create_schema_relation_with_invalid_description(self):
        with pytest.raises(ValidationError):
            SchemaRelation(name="RelationName", description=123)


class TestTriplePattern:
    def test_create_triple_pattern(self):
        triple_pattern = TriplePattern(
            head="Entity1",
            relation="Relation",
            tail="Entity2",
            description="Description",
        )
        assert triple_pattern.head == "Entity1"
        assert triple_pattern.relation == "Relation"
        assert triple_pattern.tail == "Entity2"
        assert triple_pattern.description == "Description"

    def test_create_triple_pattern_with_invalid_head(self):
        with pytest.raises(ValidationError):
            TriplePattern(
                head=123,
                relation="Relation",
                tail="Entity2",
                description="Description",
            )

    def test_create_triple_pattern_with_invalid_relation(self):
        with pytest.raises(ValidationError):
            TriplePattern(
                head="Entity1",
                relation=123,
                tail="Entity2",
                description="Description",
            )

    def test_create_triple_pattern_with_invalid_tail(self):
        with pytest.raises(ValidationError):
            TriplePattern(
                head="Entity1",
                relation="Relation",
                tail=123,
                description="Description",
            )

    def test_create_triple_pattern_with_invalid_description(self):
        with pytest.raises(ValidationError):
            TriplePattern(
                head="Entity1",
                relation="Relation",
                tail="Entity2",
                description=123,
            )


class TestSchemaTriplePattern:
    def test_create_schema_triple_pattern(self):
        head = SchemaEntity(name="Entity1", description="Entity1 Description")
        relation = SchemaRelation(
            name="Relation", description="Relation Description"
        )
        tail = SchemaEntity(name="Entity2", description="Entity2 Description")
        schema_triple_pattern = SchemaTriplePattern(
            head=head,
            relation=relation,
            tail=tail,
            description="Pattern Description",
        )
        assert schema_triple_pattern.head == head
        assert schema_triple_pattern.relation == relation
        assert schema_triple_pattern.tail == tail
        assert schema_triple_pattern.description == "Pattern Description"


class TestSchema:
    def test_create_schema(self):
        entity1 = SchemaEntity(
            name="Entity1", description="Entity1 Description"
        )
        entity2 = SchemaEntity(
            name="Entity2", description="Entity2 Description"
        )
        relation = SchemaRelation(
            name="Relation", description="Relation Description"
        )
        triple_pattern = TriplePattern(
            head="Entity1",
            relation="Relation",
            tail="Entity2",
            description="Pattern Description",
        )
        schema_triple_pattern = SchemaTriplePattern(
            head=entity1,
            relation=relation,
            tail=entity2,
            description="Pattern Description",
        )
        schema = Schema(
            entities=[entity1, entity2],
            relations=[relation],
            patterns=[triple_pattern, schema_triple_pattern],
        )
        assert schema.entities == [entity1, entity2]
        assert schema.relations == [relation]
        assert schema.patterns == [triple_pattern, schema_triple_pattern]

    def test_get_entity(self):
        entity1 = SchemaEntity(
            name="Entity1", description="Entity1 Description"
        )
        entity2 = SchemaEntity(
            name="Entity2", description="Entity2 Description"
        )
        schema = Schema(entities=[entity1, entity2])
        assert schema.get_entity("Entity1") == entity1
        assert schema.get_entity("Entity2") == entity2
        assert schema.get_entity("Entity3") is None

    def test_get_relation(self):
        relation1 = SchemaRelation(
            name="Relation1", description="Relation1 Description"
        )
        relation2 = SchemaRelation(
            name="Relation2", description="Relation2 Description"
        )
        schema = Schema(relations=[relation1, relation2])
        assert schema.get_relation("Relation1") == relation1
        assert schema.get_relation("Relation2") == relation2
        assert schema.get_relation("Relation3") is None


class TestDatasetModel:
    def test_create_dataset_model_with_list(self):
        dataset_model = DatasetModel(dataset=["item1", "item2", "item3"])
        assert dataset_model.dataset == ["item1", "item2", "item3"]

    def test_create_dataset_model_with_dict(self):
        dataset_model = DatasetModel(
            dataset={"key1": ["item1", "item2"], "key2": ["item3", "item4"]}
        )
        assert dataset_model.dataset == {
            "key1": ["item1", "item2"],
            "key2": ["item3", "item4"],
        }

    def test_create_dataset_model_with_invalid_data(self):
        with pytest.raises(ValidationError):
            DatasetModel(dataset="invalid data")


class TestPDFProcessorConfig:
    def test_create_pdf_processor_config(self):
        pdf_processor_config = PDFProcessorConfig(
            file_path="path/to/document.pdf", chunk_size=512, chunk_overlap=0
        )
        assert pdf_processor_config.file_path == "path/to/document.pdf"
        assert pdf_processor_config.chunk_size == 512
        assert pdf_processor_config.chunk_overlap == 0

    def test_create_pdf_processor_config_with_invalid_file_path(self):
        with pytest.raises(ValidationError):
            PDFProcessorConfig(file_path=123, chunk_size=512, chunk_overlap=0)

    def test_create_pdf_processor_config_with_invalid_chunk_size(self):
        with pytest.raises(ValidationError):
            PDFProcessorConfig(
                file_path="path/to/document.pdf",
                chunk_size="invalid",
                chunk_overlap=0,
            )

    def test_create_pdf_processor_config_with_invalid_chunk_overlap(self):
        with pytest.raises(ValidationError):
            PDFProcessorConfig(
                file_path="path/to/document.pdf",
                chunk_size=512,
                chunk_overlap="invalid",
            )


class TestStructuredSchemaEntity:
    def test_create_structured_schema_entity(self):
        structured_entity = StructuredSchemaEntity(
            name="EntityName",
            field=EntityField(name="Field1", properties=["prop1"]),
        )
        assert structured_entity.name == "EntityName"
        assert structured_entity.field.name == "Field1"
        assert structured_entity.field.properties == ["prop1"]

    def test_create_structured_schema_entity_defaults(self):
        structured_entity = StructuredSchemaEntity(
            name="EntityName",
            field=EntityField(name="Field1"),
        )
        assert structured_entity.name == "EntityName"
        assert structured_entity.field.name == "Field1"
        assert structured_entity.field.properties == []

    def test_structured_schema_entity_no_field_error(self):
        with pytest.raises(ValidationError):
            StructuredSchemaEntity(name="EntityName")


class TestStructuredSchemaTriplePattern:
    def test_create_structured_schema_triple_pattern(self):

        structured_pattern = StructuredSchemaTriplePattern(
            head=StructuredSchemaEntity(
                name="Entity1",
                field=EntityField(name="Field1", properties=["prop1"]),
            ),
            relation="Relation",
            tail=StructuredSchemaEntity(
                name="Entity2",
                field=EntityField(name="Field2", properties=["prop2"]),
            ),
        )

        assert structured_pattern.head.name == "Entity1"
        assert structured_pattern.head.field.name == "Field1"
        assert structured_pattern.head.field.properties == ["prop1"]
        assert structured_pattern.relation == "Relation"
        assert structured_pattern.tail.name == "Entity2"
        assert structured_pattern.tail.field.name == "Field2"
        assert structured_pattern.tail.field.properties == ["prop2"]

    def test_create_structured_schema_triple_pattern_defaults(self):
        structured_pattern = StructuredSchemaTriplePattern(
            head=StructuredSchemaEntity(
                name="Entity1", field=EntityField(name="Field1")
            ),
            relation="Relation",
            tail=StructuredSchemaEntity(
                name="Entity2", field=EntityField(name="Field2")
            ),
        )

        assert structured_pattern.head.name == "Entity1"
        assert structured_pattern.head.field.name == "Field1"
        assert structured_pattern.head.field.properties == []
        assert structured_pattern.relation == "Relation"
        assert structured_pattern.tail.name == "Entity2"
        assert structured_pattern.tail.field.name == "Field2"
        assert structured_pattern.tail.field.properties == []

    def test_structured_schema_triple_pattern_no_field_error(self):
        with pytest.raises(ValidationError):
            StructuredSchemaTriplePattern(
                head=StructuredSchemaEntity(
                    name="Entity1", field=EntityField(name="Field1")
                ),
                relation="Relation",
                tail=StructuredSchemaEntity(name="Entity2"),
            )
        with pytest.raises(ValidationError):
            StructuredSchemaTriplePattern(
                head=StructuredSchemaEntity(name="Entity1"),
                relation="Relation",
                tail=StructuredSchemaEntity(name="Entity2"),
            )
