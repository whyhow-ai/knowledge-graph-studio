"""User CRUD operations."""

import logging
import secrets

from auth0.management import Auth0
from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase

from whyhow_api.schemas.users import (
    APIKeyOutModel,
    UserAPIKeyUpdate,
    UserDocumentModel,
)
from whyhow_api.services.crud.base import update_one

logger = logging.getLogger(__name__)


async def get_user(
    db: AsyncIOMotorDatabase, username: str
) -> UserDocumentModel | None:
    """Get a user."""
    user_data = await db.user.find_one({"username": username})
    if user_data is None:
        logger.info(f"User '{username}' not found")
        return None
    else:
        logger.info(f"Found user: {user_data}")
        return UserDocumentModel(**user_data)


async def delete_user(
    db: AsyncIOMotorDatabase, user_id: ObjectId, auth0: Auth0
) -> None:
    """Delete a user."""
    async with await db.client.start_session() as session:
        async with session.start_transaction():
            # Delete the user's chunks
            await db.chunk.delete_many(
                {"created_by": user_id}, session=session
            )
            # Delete the user's documents
            await db.document.delete_many(
                {"created_by": user_id}, session=session
            )
            # Delete the user's graphs
            await db.graph.delete_many(
                {"created_by": user_id}, session=session
            )
            # Delete the user's nodes
            await db.node.delete_many({"created_by": user_id}, session=session)
            # Delete the user's queries
            await db.query.delete_many(
                {"created_by": user_id}, session=session
            )
            # Delete the user's schemas
            await db.schema.delete_many(
                {"created_by": user_id}, session=session
            )
            # Delete the user's triples
            await db.triple.delete_many(
                {"created_by": user_id}, session=session
            )
            # Delete the user's workspaces
            await db.workspace.delete_many(
                {"created_by": user_id}, session=session
            )

            # Delete the auth0 user
            user = await db.user.find_one(
                {"_id": user_id}, {"sub": 1}, session=session
            )
            if user is None:
                raise ValueError(f"User '{user_id}' not found")
            sub = user.get("sub")
            auth0.users.delete(sub)

            # Delete the user
            await db.user.delete_one({"_id": user_id}, session=session)

            # Commit the transaction
            await session.commit_transaction()


async def update_api_key(
    db: AsyncIOMotorDatabase, user_id: ObjectId, new_api_key: str
) -> APIKeyOutModel | None:
    """Update a user's API key in the database."""
    try:
        updated_user = await update_one(
            collection=db["user"],
            document_model=APIKeyOutModel,
            id=user_id,
            document=UserAPIKeyUpdate(api_key=new_api_key),
            user_id=user_id,
        )
        return APIKeyOutModel.model_validate(updated_user)
    except Exception as e:
        logger.error(f"Error updating API key: {e}")
        return None


async def generate_api_key() -> str:
    """Generate a new whyhow api key."""
    new_api_key = secrets.token_hex(16)
    return new_api_key
