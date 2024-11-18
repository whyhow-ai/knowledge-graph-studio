from unittest.mock import AsyncMock

import pytest
from bson import ObjectId

from whyhow_api.dependencies import get_db, get_user
from whyhow_api.schemas.rules import MergeNodesRule, RuleCreate, RuleOut


class TestRules:

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

    def test_create_rule_successful(
        self, client, rule_document_mock, monkeypatch
    ):
        rule_create_mock = RuleCreate(
            workspace=ObjectId(), rule=rule_document_mock.rule
        )
        rule_create_mock.workspace = str(rule_create_mock.workspace)
        fake_create_rule = AsyncMock()
        fake_create_rule.return_value = rule_document_mock
        monkeypatch.setattr(
            "whyhow_api.routers.rules.create_rule",
            fake_create_rule,
        )

        client.app.dependency_overrides[get_db] = lambda: AsyncMock()
        client.app.dependency_overrides[get_user] = lambda: ObjectId()

        response = client.post(
            "/rules",
            json=rule_create_mock.model_dump(),
        )
        assert response.status_code == 200

        data = response.json()
        assert data["message"] == "Rule created successfully."
        assert data["status"] == "success"
        assert data["count"] == 1
        assert (
            data["rules"][0]["rule"]["rule_type"]
            == rule_document_mock.rule.rule_type
        )

    def test_read_workspace_rules_successful(
        self, client, rule_document_mock, monkeypatch
    ):

        fake_get_rules = AsyncMock()
        fake_get_rules.return_value = ([rule_document_mock], 1)
        monkeypatch.setattr(
            "whyhow_api.routers.rules.get_workspace_rules", fake_get_rules
        )

        client.app.dependency_overrides[get_db] = lambda: AsyncMock()
        client.app.dependency_overrides[get_user] = lambda: ObjectId()

        response = client.get(
            "/rules", params={"workspace_id": rule_document_mock.workspace}
        )
        assert response.status_code == 200

        data = response.json()
        assert data["message"] == "Rules retrieved successfully."
        assert data["status"] == "success"
        assert data["count"] == 1
        assert (
            data["rules"][0]["rule"]["rule_type"]
            == rule_document_mock.rule.rule_type
        )

    def test_delete_rule_successful(
        self, client, rule_document_mock, monkeypatch
    ):

        fake_delete_rule = AsyncMock()
        fake_delete_rule.return_value = rule_document_mock
        monkeypatch.setattr(
            "whyhow_api.routers.rules.delete_rule",
            fake_delete_rule,
        )

        client.app.dependency_overrides[get_db] = lambda: AsyncMock()
        client.app.dependency_overrides[get_user] = lambda: ObjectId()

        response = client.delete(f"/rules/{ObjectId()}")
        assert response.status_code == 200

        data = response.json()
        assert data["message"] == "Rule deleted successfully."
        assert data["status"] == "success"
        assert data["count"] == 1
        assert (
            data["rules"][0]["rule"]["rule_type"]
            == rule_document_mock.rule.rule_type
        )

    def test_delete_workspace_rule_not_found(self, client, monkeypatch):

        fake_delete_rule = AsyncMock()
        fake_delete_rule.return_value = None
        monkeypatch.setattr(
            "whyhow_api.routers.rules.delete_rule",
            fake_delete_rule,
        )

        client.app.dependency_overrides[get_db] = lambda: AsyncMock()
        client.app.dependency_overrides[get_user] = lambda: ObjectId()

        response = client.delete(f"/rules/{ObjectId()}")
        assert response.status_code == 404

        data = response.json()
        assert data["detail"] == "Rule not found."
