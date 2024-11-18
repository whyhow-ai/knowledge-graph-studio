"""Validation utilities."""

from bson import ObjectId
from bson.errors import InvalidId
from fastapi import HTTPException


def safe_object_id(oid: str) -> ObjectId:
    """Convert a string to ObjectId, raising an HTTPException if invalid."""
    try:
        return ObjectId(oid)
    except (InvalidId, TypeError) as e:
        raise HTTPException(status_code=400, detail="Invalid ID format") from e
