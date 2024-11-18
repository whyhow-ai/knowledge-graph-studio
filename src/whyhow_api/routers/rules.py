"""Rule CRUD router."""

import logging

from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException, Query, status
from motor.motor_asyncio import AsyncIOMotorDatabase

from whyhow_api.dependencies import get_db, get_user
from whyhow_api.schemas.rules import RuleCreate, RuleOut, RulesResponse
from whyhow_api.services.crud.rule import (
    create_rule,
    delete_rule,
    get_workspace_rules,
)
from whyhow_api.utilities.routers import order_query

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Rules"], prefix="/rules")


@router.post(
    "",
    response_model_exclude_none=True,
    response_model=RulesResponse,
)
async def create_rule_endpoint(
    rule: RuleCreate,
    db: AsyncIOMotorDatabase = Depends(get_db),
    user_id: ObjectId = Depends(get_user),
) -> RulesResponse:
    """Create a new rule."""
    created_rule = await create_rule(
        db=db,
        rule=rule,
        user_id=user_id,
    )
    return RulesResponse(
        message="Rule created successfully.",
        status="success",
        count=1,
        rules=[RuleOut.model_validate(created_rule)],
    )


@router.get(
    "",
    response_model_exclude_none=True,
    response_model=RulesResponse,
)
async def read_rules_endpoint(
    workspace_id: str | None = Query(
        None, description="The id of the workspace"
    ),
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=-1, le=50),
    order: int = Depends(order_query),
    db: AsyncIOMotorDatabase = Depends(get_db),
    user_id: ObjectId = Depends(get_user),
) -> RulesResponse:
    """Get all workspace rules."""
    rules, total_count = await get_workspace_rules(
        db=db,
        workspace_id=ObjectId(workspace_id) if workspace_id else None,
        user_id=user_id,
        skip=skip,
        limit=limit,
        order=order,
    )
    return RulesResponse(
        message="Rules retrieved successfully.",
        status="success",
        count=total_count,
        rules=[RuleOut.model_validate(rule) for rule in rules],
    )


@router.delete(
    "/{rule_id}",
    response_model_exclude_none=True,
    response_model=RulesResponse,
)
async def delete_rule_endpoint(
    rule_id: str,
    db: AsyncIOMotorDatabase = Depends(get_db),
    user_id: ObjectId = Depends(get_user),
) -> RulesResponse:
    """Delete a rule."""
    deleted_rule = await delete_rule(
        db=db,
        rule_id=ObjectId(rule_id),
        user_id=user_id,
    )
    if deleted_rule is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Rule not found.",
        )
    return RulesResponse(
        message="Rule deleted successfully.",
        status="success",
        count=1,
        rules=[RuleOut.model_validate(deleted_rule)],
    )
