"""Users router."""

import logging
import re

from auth0.management import Auth0
from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException
from motor.motor_asyncio import AsyncIOMotorDatabase

from whyhow_api.dependencies import get_auth0, get_db, get_user
from whyhow_api.schemas.users import (
    APIKeyOutModel,
    DeleteUserResponse,
    GetAPIKeyResponse,
    GetProvidersDetailsResponse,
    GetUserStatusResponse,
    Provider,
    ProviderConfig,
    SetProvidersDetailsResponse,
)
from whyhow_api.services.crud import user
from whyhow_api.services.crud.base import get_one
from whyhow_api.services.crud.user import generate_api_key, update_api_key

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Users"], prefix="/users")


@router.get(
    "/api_key",
    response_model=GetAPIKeyResponse,
)
async def get_whyhow_api_key(
    db: AsyncIOMotorDatabase = Depends(get_db),
    user_id: ObjectId = Depends(get_user),
) -> GetAPIKeyResponse:
    """Get user's API key."""
    api_key_out = await get_one(
        collection=db["user"],
        document_model=APIKeyOutModel,
        user_id=user_id,
        id=user_id,
    )
    if api_key_out is None:
        raise HTTPException(
            status_code=404,
            detail="API key not found",
        )
    return GetAPIKeyResponse(
        message="API key retrieved successfully.",
        status="success",
        count=1,
        whyhow_api_key=[APIKeyOutModel.model_validate(api_key_out)],
    )


@router.post(
    "/rotate_api_key",
    response_model=GetAPIKeyResponse,
)
async def rotate_whyhow_api_key(
    db: AsyncIOMotorDatabase = Depends(get_db),
    user_id: ObjectId = Depends(get_user),
) -> GetAPIKeyResponse:
    """Rotate user's WhyHow API key."""
    logger.info(f"Rotating API key for user: {user_id}")
    new_api_key = await generate_api_key()

    updated_key = await update_api_key(
        db=db,
        user_id=user_id,
        new_api_key=new_api_key,
    )

    if updated_key is None:
        logger.error(f"API key rotation failed for user: {user_id}")
        raise HTTPException(
            status_code=404,
            detail="API key rotation failed",
        )
    return GetAPIKeyResponse(
        message="New API Key generated.",
        status="success",
        count=1,
        whyhow_api_key=[APIKeyOutModel.model_validate(updated_key)],
    )


@router.put(
    "/set_providers_details",
    response_model=SetProvidersDetailsResponse,
)
async def set_providers_details(
    request: ProviderConfig,
    db: AsyncIOMotorDatabase = Depends(get_db),
    user_id: ObjectId = Depends(get_user),
) -> SetProvidersDetailsResponse:
    """Set providers details."""
    request_provider = request.providers[0].model_dump()
    obfuscated = bool(re.fullmatch(r"\*+", request_provider["api_key"]))

    try:
        user_doc = await db.user.find_one({"_id": user_id}, {"providers": 1})

        if user_doc is None:
            raise HTTPException(
                status_code=404,
                detail="User not found",
            )

        if obfuscated:
            request_provider["api_key"] = user_doc["providers"][0]["api_key"]

        await db.user.update_one(
            {"_id": user_id},
            {"$set": {"providers": [request_provider]}},
        )
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Error setting providers details",
        )

    updated_user = await db.user.find_one({"_id": user_id})

    if updated_user is not None:
        providers_data = updated_user["providers"]
    else:
        providers_data = []
    providers = [Provider(**provider) for provider in providers_data]

    # Obfuscate API key when returning to client
    providers_out = []
    for provider in providers:
        if provider.api_key:
            provider.api_key = "*" * len(provider.api_key)
        providers_out.append(provider)
    return SetProvidersDetailsResponse(
        message="Providers details set successfully",
        status="success",
        count=1,
        providers=providers_out,
    )


@router.get(
    "/providers_details",
    response_model=GetProvidersDetailsResponse,
)
async def get_providers_details(
    db: AsyncIOMotorDatabase = Depends(get_db),
    user_id: ObjectId = Depends(get_user),
) -> ProviderConfig:
    """Get providers details."""
    user = await db.user.find_one({"_id": user_id}, {"providers": 1})
    if user is None:
        raise HTTPException(
            status_code=404,
            detail="User not found",
        )
    if "providers" not in user:
        raise HTTPException(
            status_code=404,
            detail="Providers details not found",
        )
    for provider in user["providers"]:
        if provider["type"] == "llm" and provider["value"] == "whyhow-openai":
            provider["value"] = None

    # Obfuscate API key when returning to client
    for provider in user["providers"]:
        if "api_key" in provider and provider["api_key"]:
            provider["api_key"] = "*" * len(provider["api_key"])
    return GetProvidersDetailsResponse(
        message="Providers details retrieved successfully",
        status="success",
        count=1,
        providers=ProviderConfig(**user).providers,
    )


@router.delete(
    "",
    response_model=DeleteUserResponse,
)
async def delete_user(
    db: AsyncIOMotorDatabase = Depends(get_db),
    user_id: ObjectId = Depends(get_user),
    auth0: Auth0 = Depends(get_auth0),
) -> DeleteUserResponse:
    """Delete user."""
    await user.delete_user(db=db, user_id=user_id, auth0=auth0)
    return DeleteUserResponse(
        message="User deleted successfully",
        status="success",
        count=1,
    )


@router.get(
    "/status",
    response_model=GetUserStatusResponse,
)
async def get_user_status(
    db: AsyncIOMotorDatabase = Depends(get_db),
    user_id: ObjectId = Depends(get_user),
) -> GetUserStatusResponse:
    """Get user status."""
    user = await db.user.find_one({"_id": user_id}, {"active": True})
    if user is None or "active" not in user:
        return GetUserStatusResponse(
            message="User not found",
            status="success",
            count=0,
            active=False,
        )
    return GetUserStatusResponse(
        message="User status retrieved successfully",
        status="success",
        count=1,
        active=user["active"],
    )
