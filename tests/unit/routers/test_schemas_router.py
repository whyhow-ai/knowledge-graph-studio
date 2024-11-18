from unittest.mock import AsyncMock

import pytest
from bson import ObjectId

from whyhow_api.dependencies import (
    get_db,
    get_db_client,
    get_user,
    valid_schema_id,
)
from whyhow_api.models.common import (
    SchemaEntity,
    SchemaRelation,
    SchemaTriplePattern,
)
from whyhow_api.schemas.schemas import (
    SchemaDocumentModel,
    SchemaOutWithWorkspaceDetails,
)


class TestSchemasGetOne:

    @pytest.fixture
    def schema_object_mock(self):
        return SchemaDocumentModel(
            _id=ObjectId(),
            name="test schema",
            workspace=ObjectId(),
            entities=[
                SchemaEntity(name="entity", description="entity description"),
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
                        name="entity", description="entity description"
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

    @pytest.mark.skip("Skipping for now, TODO: Fix test")
    def test_get_schema_successful(
        self, client, monkeypatch, schema_object_mock
    ):
        schema_id_mock = ObjectId()
        fake_get_one = AsyncMock()
        fake_get_one.return_value = [
            schema_object_mock.model_dump(by_alias=True)
        ]
        monkeypatch.setattr("whyhow_api.routers.schemas.get_all", fake_get_one)

        client.app.dependency_overrides[get_db] = lambda: AsyncMock()
        client.app.dependency_overrides[get_user] = lambda: ObjectId()

        response = client.get(f"/schemas/{schema_id_mock}")
        assert response.status_code == 200

        data = response.json()
        assert data["message"] == "Schema retrieved successfully."
        assert data["status"] == "success"
        assert data["count"] == 1
        assert data["schemas"][0]["name"] == "test schema"
        assert len(data["schemas"][0]["entities"]) == len(
            schema_object_mock.entities
        )
        assert len(data["schemas"][0]["relations"]) == len(
            schema_object_mock.relations
        )
        assert len(data["schemas"][0]["patterns"]) == len(
            schema_object_mock.patterns
        )

    @pytest.mark.skip("Skipping for now, TODO: Fix test")
    def test_get_schema_failure(self, client, monkeypatch):

        schema_id_mock = ObjectId()

        fake_get_one = AsyncMock()
        fake_get_one.return_value = None
        monkeypatch.setattr("whyhow_api.routers.schemas.get_one", fake_get_one)

        client.app.dependency_overrides[get_db] = lambda: AsyncMock()
        client.app.dependency_overrides[get_user] = lambda: ObjectId()

        response = client.get(f"/schemas/{schema_id_mock}")
        assert response.status_code == 404

        data = response.json()
        assert data["detail"] == "Schema not found"


class TestSchemasGetAll:

    @pytest.fixture
    def schema_object_mock(self):
        return SchemaOutWithWorkspaceDetails(
            _id=ObjectId(),
            name="test schema",
            workspace={"_id": ObjectId(), "name": "test workspace"},
            entities=[
                SchemaEntity(name="entity", description="entity description"),
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
                        name="entity", description="entity description"
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
            type="txt",
        )

    def test_get_schemas_successful(
        self, client, monkeypatch, schema_object_mock
    ):

        fake_get_all = AsyncMock()
        fake_get_all.return_value = [
            schema_object_mock.model_dump(by_alias=True)
        ]
        monkeypatch.setattr("whyhow_api.routers.schemas.get_all", fake_get_all)

        fake_get_all_count = AsyncMock()
        fake_get_all_count.return_value = 1
        monkeypatch.setattr(
            "whyhow_api.routers.schemas.get_all_count", fake_get_all_count
        )

        client.app.dependency_overrides[get_db] = lambda: AsyncMock()
        client.app.dependency_overrides[get_user] = lambda: ObjectId()

        response = client.get("/schemas", params={"skip": 0, "limit": 1})
        assert response.status_code == 200

        data = response.json()
        assert data["message"] == "Schemas retrieved successfully."
        assert data["status"] == "success"
        assert data["count"] == 1
        assert data["schemas"][0]["name"] == "test schema"
        assert len(data["schemas"][0]["entities"]) == len(
            schema_object_mock.entities
        )
        assert len(data["schemas"][0]["relations"]) == len(
            schema_object_mock.relations
        )
        assert len(data["schemas"][0]["patterns"]) == len(
            schema_object_mock.patterns
        )


class TestSchemasCreate:
    pass

    # @pytest.fixture
    # def text_schema_object_mock(self):
    #     return SchemaDocumentModel(
    #         _id=ObjectId(),
    #         name="test schema",
    #         workspace=ObjectId(),
    #         entities=[
    #             SchemaEntity(name="entity", description="entity description"),
    #             SchemaEntity(
    #                 name="entity2", description="entity2 description"
    #             ),
    #         ],
    #         relations=[
    #             SchemaRelation(
    #                 name="relation", description="relation description"
    #             ),
    #         ],
    #         patterns=[
    #             SchemaTriplePattern(
    #                 head=SchemaEntity(
    #                     name="entity", description="entity description"
    #                 ),
    #                 tail=SchemaEntity(
    #                     name="entity2", description="entity2 description"
    #                 ),
    #                 relation=SchemaRelation(
    #                     name="relation", description="relation description"
    #                 ),
    #                 description="pattern description",
    #             )
    #         ],
    #         created_by=ObjectId(),
    #         type="txt",
    #     )

    # def test_create_schema_successful(
    #     self, client, monkeypatch, text_schema_object_mock
    # ):

    #     fake_get_one = AsyncMock()
    #     fake_get_one.return_value = text_schema_object_mock.workspace
    #     monkeypatch.setattr("whyhow_api.routers.schemas.get_one", fake_get_one)

    #     fake_create = AsyncMock()
    #     fake_create.return_value = text_schema_object_mock.model_dump()
    #     monkeypatch.setattr(
    #         "whyhow_api.routers.schemas.create_one", fake_create
    #     )

    #     client.app.dependency_overrides[get_db] = lambda: AsyncMock()
    #     client.app.dependency_overrides[get_user] = (
    #         lambda: text_schema_object_mock.created_by
    #     )

    #     create_schema_body = {
    #         "name": text_schema_object_mock.name,
    #         "workspace": str(text_schema_object_mock.workspace),
    #         "entities": [
    #             e.model_dump() for e in text_schema_object_mock.entities
    #         ],
    #         "relations": [
    #             r.model_dump() for r in text_schema_object_mock.relations
    #         ],
    #         "patterns": [
    #             {
    #                 "head": p.head.name,
    #                 "tail": p.tail.name,
    #                 "relation": p.relation.name,
    #                 "description": p.description,
    #             }
    #             for p in text_schema_object_mock.patterns
    #         ],
    #         "type": "txt",
    #     }

    #     response = client.post("/schemas/txt", json=create_schema_body)
    #     assert response.status_code == 200

    #     data = response.json()
    #     assert data["message"] == "Schema created successfully."
    #     assert data["status"] == "success"
    #     assert data["count"] == 1
    #     assert data["schemas"][0]["name"] == "test schema"
    #     assert data["schemas"][0]["type"] == "txt"
    #     assert len(data["schemas"][0]["entities"]) == len(
    #         text_schema_object_mock.entities
    #     )
    #     assert len(data["schemas"][0]["relations"]) == len(
    #         text_schema_object_mock.relations
    #     )
    #     assert len(data["schemas"][0]["patterns"]) == len(
    #         text_schema_object_mock.patterns
    #     )

    # def test_create_schema_workspace_not_found(
    #     self, client, monkeypatch, text_schema_object_mock
    # ):

    #     fake_get_one = AsyncMock()
    #     fake_get_one.return_value = None
    #     monkeypatch.setattr("whyhow_api.routers.schemas.get_one", fake_get_one)

    #     client.app.dependency_overrides[get_db] = lambda: AsyncMock()
    #     client.app.dependency_overrides[get_user] = lambda: ObjectId()

    #     create_schema_body = {
    #         "name": text_schema_object_mock.name,
    #         "workspace": str(text_schema_object_mock.workspace),
    #         "entities": [
    #             e.model_dump() for e in text_schema_object_mock.entities
    #         ],
    #         "relations": [
    #             r.model_dump() for r in text_schema_object_mock.relations
    #         ],
    #         "patterns": [
    #             {
    #                 "head": p.head.name,
    #                 "tail": p.tail.name,
    #                 "relation": p.relation.name,
    #                 "description": p.description,
    #             }
    #             for p in text_schema_object_mock.patterns
    #         ],
    #         "type": "txt",
    #     }

    #     response = client.post("/schemas/txt", json=create_schema_body)
    #     assert response.status_code == 404
    #     assert response.json()["detail"] == "Workspace not found"


class TestSchemasUpdate:

    @pytest.fixture
    def schema_object_mock(self):
        return SchemaDocumentModel(
            _id=ObjectId(),
            name="test schema",
            workspace=ObjectId(),
            entities=[
                SchemaEntity(name="entity", description="entity description"),
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
                        name="entity", description="entity description"
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
            type="txt",
        )

    def test_update_schema_successful(
        self, client, monkeypatch, schema_object_mock
    ):

        schema_id_mock = ObjectId()

        fake_update = AsyncMock()
        fake_update.return_value = schema_object_mock.model_dump()
        monkeypatch.setattr(
            "whyhow_api.routers.schemas.update_one", fake_update
        )

        client.app.dependency_overrides[get_db] = lambda: AsyncMock()
        client.app.dependency_overrides[get_user] = lambda: ObjectId()

        response = client.put(
            f"/schemas/{schema_id_mock}",
            json={"name": schema_object_mock.name},
        )
        assert response.status_code == 200

        data = response.json()
        assert data["message"] == "Schema updated successfully."
        assert data["status"] == "success"
        assert data["count"] == 1
        assert data["schemas"][0]["name"] == schema_object_mock.name

        assert data["schemas"][0]["created_by"] == str(
            schema_object_mock.created_by
        )

    def test_update_schema_not_found(
        self, client, monkeypatch, schema_object_mock
    ):

        fake_update = AsyncMock()
        fake_update.return_value = None
        monkeypatch.setattr(
            "whyhow_api.routers.schemas.update_one", fake_update
        )

        client.app.dependency_overrides[get_db] = lambda: AsyncMock()
        client.app.dependency_overrides[get_user] = lambda: ObjectId()

        response = client.put(
            f"/schemas/{ObjectId()}",
            json={"name": schema_object_mock.name},
        )
        assert response.status_code == 404

        data = response.json()
        assert data["detail"] == "Schema not found"


class TestSchemasDelete:

    @pytest.fixture
    def schema_object_mock(self):
        return SchemaDocumentModel(
            _id=ObjectId(),
            name="test schema",
            workspace=ObjectId(),
            entities=[
                SchemaEntity(name="entity", description="entity description"),
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
                        name="entity", description="entity description"
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
            type="txt",
        )

    def test_delete_schema_successful(
        self, client, monkeypatch, schema_object_mock
    ):

        schema_id_mock = ObjectId()

        fake_delete = AsyncMock()
        monkeypatch.setattr(
            "whyhow_api.routers.schemas.delete_schema", fake_delete
        )

        client.app.dependency_overrides[get_db] = lambda: AsyncMock()
        client.app.dependency_overrides[get_db_client] = lambda: AsyncMock()
        client.app.dependency_overrides[get_user] = lambda: ObjectId()
        client.app.dependency_overrides[valid_schema_id] = (
            lambda: schema_object_mock
        )

        response = client.delete(f"/schemas/{schema_id_mock}")
        assert response.status_code == 200

        data = response.json()
        assert data["message"] == "Schema deleted successfully."
        assert data["status"] == "success"
        assert data["count"] == 1
        assert data["schemas"][0]["name"] == "test schema"
