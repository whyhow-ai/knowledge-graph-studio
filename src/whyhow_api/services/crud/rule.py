"""Rule CRUD operations."""

import logging
from typing import Any, Dict, Optional, Tuple

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase
from pydantic import BaseModel

from whyhow_api.models.common import Triple
from whyhow_api.schemas.rules import (
    MergeNodesRule,
    RuleCreate,
    RuleDocumentModel,
    RuleOut,
)
from whyhow_api.services.crud.base import create_one, get_all, get_all_count

logger = logging.getLogger(__name__)


async def create_rule(
    db: AsyncIOMotorDatabase,
    rule: RuleCreate,
    user_id: ObjectId,
) -> BaseModel:
    """Create a new rule."""
    created_rule = await create_one(
        collection=db.rule,
        document_model=RuleDocumentModel,
        user_id=user_id,
        document=rule,
    )
    return created_rule


async def get_workspace_rules(
    db: AsyncIOMotorDatabase,
    user_id: ObjectId,
    skip: int,
    limit: int,
    order: int,
    workspace_id: ObjectId | None = None,
) -> Tuple[list[BaseModel], int]:
    """Get all workspace rules."""
    collection = db.rule
    pre_filters: Dict[str, Any] = {}
    if workspace_id:
        pre_filters["workspace"] = ObjectId(workspace_id)

    pipeline = [
        {"$match": pre_filters},
    ]

    # Get total count of items in db
    total_count = await get_all_count(
        collection=collection, user_id=user_id, aggregation_query=pipeline
    )

    if total_count == 0:
        logger.info("No rules found.")
        return [], total_count
    else:
        return (
            await get_all(
                collection=collection,
                document_model=RuleDocumentModel,
                user_id=user_id,
                aggregation_query=pipeline,
                skip=skip,
                limit=limit,
                order=order,
            ),
            total_count,
        )


async def get_graph_rules(
    db: AsyncIOMotorDatabase,
    graph_id: ObjectId,
    skip: int,
    limit: int,
    order: int,
    user_id: Optional[ObjectId] = None,
) -> Tuple[list[BaseModel], int]:
    """Get all graph rules."""
    pipeline: list[dict[str, Any]] = [
        {
            "$match": {
                "_id": ObjectId(graph_id),
                **({"created_by": ObjectId(user_id)} if user_id else {}),
            }
        },
        {"$project": {"rules": 1}},
        {"$unwind": "$rules"},
    ]
    total_count = await get_all_count(
        collection=db.graph,
        user_id=user_id,
        aggregation_query=pipeline,
    )
    pipeline.extend(
        [
            {"$sort": {"rules.created_at": order, "rules._id": order}},
            {"$skip": skip},
        ]
    )
    if limit >= 0:
        pipeline.append({"$limit": limit})

    rules = await db.graph.aggregate(pipeline).to_list(None)
    rules = [rule["rules"] for rule in rules]

    if total_count == 0:
        logger.info("No rules found.")
        return [], total_count
    else:
        return rules, total_count


async def delete_rule(
    db: AsyncIOMotorDatabase,
    rule_id: ObjectId,
    user_id: ObjectId,
) -> Optional[Dict[str, Any]]:
    """Delete a rule."""
    rule = await db.rule.find_one(
        {"_id": ObjectId(rule_id), "created_by": ObjectId(user_id)}
    )
    if rule is None:
        return None
    result = await db.rule.delete_one(
        {"_id": ObjectId(rule_id), "created_by": ObjectId(user_id)}
    )
    if result.deleted_count == 1:
        return rule
    else:
        return None


def merge_nodes_transform(
    triples: list[Triple], rule: MergeNodesRule
) -> list[Triple]:
    """Merge nodes rule transformation."""
    for triple in triples:
        if (
            triple.head in rule.from_node_names
            and triple.head_type == rule.node_type
        ):
            triple.head = rule.to_node_name
        if (
            triple.tail in rule.from_node_names
            and triple.tail_type == rule.node_type
        ):
            triple.tail = rule.to_node_name
    return triples


RULE_TRANSFORMS = {
    MergeNodesRule: merge_nodes_transform,
}


def apply_rules_to_triples(
    triples: list[Triple], rules: list[RuleOut]
) -> list[Triple]:
    """
    Apply rules to a list of triples.

    Parameters
    ----------
    triples : list[Triple]
        The triples to update.
    rules : list[RuleOut]
        The rules to apply.

    Returns
    -------
    list[Triple]
        The updated triples.
    """
    updated_triples = triples
    for rule in rules:
        updated_triples = RULE_TRANSFORMS[type(rule.rule)](
            updated_triples, rule.rule
        )
    return updated_triples
