from unittest.mock import AsyncMock, MagicMock

import pytest
from bson import ObjectId

from whyhow_api.models.common import Triple
from whyhow_api.schemas.rules import (
    MergeNodesRule,
    RuleCreate,
    RuleDocumentModel,
    RuleOut,
)
from whyhow_api.services.crud.rule import (
    apply_rules_to_triples,
    create_rule,
    delete_rule,
    get_graph_rules,
    get_workspace_rules,
    merge_nodes_transform,
)


@pytest.mark.asyncio
async def test_create_rule(monkeypatch):
    db = MagicMock()
    user_id = ObjectId()
    rule = RuleCreate(
        workspace=ObjectId(),
        rule=MergeNodesRule(
            rule_type="merge_nodes",
            from_node_names=["test from node"],
            to_node_name="test to node",
            node_type="test node type",
        ),
    )

    fake_create_rule = AsyncMock()
    fake_create_rule.return_value = rule
    monkeypatch.setattr(
        "whyhow_api.services.crud.rule.create_one",
        fake_create_rule,
    )

    result = await create_rule(db, rule, user_id)

    assert result.workspace == rule.workspace
    assert result.rule.rule_type == rule.rule.rule_type

    fake_create_rule.assert_called_once_with(
        collection=db.rule,
        document_model=RuleDocumentModel,
        user_id=user_id,
        document=rule,
    )


@pytest.mark.asyncio
async def test_get_workspace_rules(monkeypatch):
    db = MagicMock()
    workspace_id = ObjectId()
    user_id = ObjectId()
    skip = 0
    limit = 10
    order = 1

    fake_get_all_count = AsyncMock(return_value=5)
    monkeypatch.setattr(
        "whyhow_api.services.crud.rule.get_all_count",
        fake_get_all_count,
    )

    fake_get_all = AsyncMock(return_value=[MagicMock(spec=RuleDocumentModel)])
    monkeypatch.setattr(
        "whyhow_api.services.crud.rule.get_all",
        fake_get_all,
    )

    result, total_count = await get_workspace_rules(
        db, user_id, skip, limit, order, workspace_id
    )

    assert len(result) == 1
    assert total_count == 5

    fake_get_all_count.assert_called_once()
    fake_get_all.assert_called_once()


@pytest.mark.asyncio
async def test_get_graph_rules(monkeypatch):
    db = MagicMock()
    graph_id = ObjectId()
    user_id = ObjectId()
    skip = 0
    limit = 10
    order = 1

    fake_get_all_count = AsyncMock(return_value=5)
    monkeypatch.setattr(
        "whyhow_api.services.crud.rule.get_all_count",
        fake_get_all_count,
    )

    fake_aggregate = AsyncMock(return_value=[{"rules": MagicMock()}])
    db.graph.aggregate.return_value.to_list = fake_aggregate

    result, total_count = await get_graph_rules(
        db, graph_id, skip, limit, order, user_id
    )

    assert len(result) == 1
    assert total_count == 5

    fake_get_all_count.assert_called_once()
    fake_aggregate.assert_called_once()


@pytest.mark.asyncio
async def test_delete_rule():
    db = MagicMock()
    rule_id = ObjectId()
    user_id = ObjectId()

    fake_find_one = AsyncMock()
    fake_find_one.return_value = {"_id": rule_id, "created_by": user_id}
    db.rule.find_one = fake_find_one

    fake_delete_one = AsyncMock()
    fake_delete_one.return_value = MagicMock(deleted_count=1)
    db.rule.delete_one = fake_delete_one

    result = await delete_rule(db, rule_id, user_id)

    assert result["_id"] == rule_id

    fake_find_one.assert_called_once()
    fake_delete_one.assert_called_once()


def test_merge_nodes_transform():
    triples = [
        Triple(
            head="A",
            head_type="test node type",
            tail="B",
            tail_type="test node type",
            relation="test relation",
        ),
        Triple(
            head="C",
            head_type="another type",
            tail="D",
            tail_type="test node type",
            relation="test relation",
        ),
        Triple(
            head="A",
            head_type="test node type",
            tail="B",
            tail_type="different type",
            relation="test relation",
        ),
    ]
    rule = MergeNodesRule(
        rule_type="merge_nodes",
        from_node_names=["A", "B"],
        to_node_name="Z",
        node_type="test node type",
    )

    result = merge_nodes_transform(triples, rule)

    assert result[0].head == "Z"
    assert result[0].tail == "Z"
    assert result[1].head == "C"
    assert result[1].tail == "D"
    assert result[2].head == "Z"
    assert result[2].tail == "B"


def test_apply_rules_to_triples():
    triples = [
        Triple(
            head="A",
            head_type="test node type",
            tail="B",
            tail_type="test node type",
            relation="test relation",
        ),
        Triple(
            head="C",
            head_type="another type",
            tail="D",
            tail_type="test node type",
            relation="test relation",
        ),
        Triple(
            head="A",
            head_type="test node type",
            tail="B",
            tail_type="different type",
            relation="test relation",
        ),
    ]
    rule = RuleOut(
        workspace_id=ObjectId(),
        _id=ObjectId(),
        created_by=ObjectId(),
        rule=MergeNodesRule(
            rule_type="merge_nodes",
            from_node_names=["A", "B"],
            to_node_name="Z",
            node_type="test node type",
        ),
    )

    result = apply_rules_to_triples(triples, [rule])

    assert result[0].head == "Z"
    assert result[0].tail == "Z"
    assert result[1].head == "C"
    assert result[1].tail == "D"
    assert result[2].head == "Z"
    assert result[2].tail == "B"
