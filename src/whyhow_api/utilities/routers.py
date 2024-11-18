"""Routers utilities."""

import logging
import re
from typing import Any, Dict, List

from bson import ObjectId
from fastapi import Query

logger = logging.getLogger(__name__)


def clean_url(url: str) -> str:
    """Clean URL of bson object ids."""
    cleaned_url = re.sub(r"/[a-f\d]{24}(?=/|$)", "", url)

    # Ensure the cleaned URL doesn't end with a slash unless it's the root
    cleaned_url = (
        cleaned_url.rstrip("/") if len(cleaned_url) > 1 else cleaned_url
    )

    return cleaned_url


def list_aggregation(
    user_id: ObjectId | None,
    aggregation_query: List[Dict[str, Any]],
    skip: int | None = None,
    limit: int | None = None,
    order: int | None = None,
    count: bool = False,
) -> List[Dict[str, Any]]:
    """List aggregation query.

    Creates a new aggregation pipeline based on provided parameters, without modifying the input query.

    Parameters
    ----------
    user_id : ObjectId
        The ID of the user to match documents.
    aggregation_query : List[Dict[str, Any]]
        Base aggregation query to which additional stages will be added.
    skip : int, optional
        Number of documents to skip (defaults to None).
    limit : int, optional
        Maximum number of documents to return, -1 for no limit (defaults to None).
    order : int, optional
        Sort order, 1 for ascending, -1 for descending (defaults to None).
    count : bool, optional
        If True, modifies the pipeline to count the documents instead of returning them.

    Returns
    -------
    List[Dict[str, Any]]
        The modified aggregation pipeline.
    """
    # Create a copy of the aggregation query to avoid mutating the original list
    pipeline = list(aggregation_query)

    # Add the match stage at the start
    if user_id:
        pipeline.insert(0, {"$match": {"created_by": user_id}})

    if count:
        pipeline.append({"$count": "total"})
    else:
        if order is not None:
            pipeline.append({"$sort": {"created_at": order, "_id": order}})
        if skip is not None:
            pipeline.append({"$skip": skip})
        if limit is not None:
            if limit >= 0:
                pipeline.append({"$limit": limit})
            elif limit == -1:
                pass  # No limit is applied
            else:
                raise ValueError(
                    "Limit must be greater than or equal to 0 or -1 for unrestricted."
                )

    # logger.info(f"Aggregation pipeline: {pipeline}")

    return pipeline


def order_query(
    order: str = Query(default="descending", enum=["ascending", "descending"])
) -> int:
    """Convert order to 1 or -1."""
    return 1 if order == "ascending" else -1
