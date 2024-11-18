from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from bson import ObjectId

from whyhow_api.models.common import LLMClient
from whyhow_api.services.crud.triple import (
    delete_triple,
    update_triple_embeddings,
)


@pytest.mark.asyncio
async def test_delete_triple():
    triple_id = ObjectId()
    db = MagicMock()
    db.triple.delete_one = AsyncMock(return_value=None)

    session = MagicMock()
    session.start_transaction.return_value = AsyncMock()
    session.commit_transaction = AsyncMock()

    db_client = AsyncMock()
    db_client.start_session.return_value.__aenter__.return_value = session

    user_id = ObjectId()

    await delete_triple(db, user_id, triple_id)
    db.triple.delete_one.assert_awaited_once_with(
        {
            "_id": triple_id,
            "created_by": user_id,
        }
    )


@pytest.mark.asyncio
@patch("whyhow_api.services.crud.triple.embed_triples", new_callable=AsyncMock)
@patch("whyhow_api.services.crud.triple.UpdateOne", new_callable=MagicMock)
async def test_update_triple_embeddings(mock_update_one, mock_embed_triples):
    triple_ids = [ObjectId(), ObjectId()]
    user_id = ObjectId()
    mock_triples = [
        {
            "head": "head1",
            "head_type": "type1",
            "head_properties": {"key1": "value1"},
            "relation": "relation1",
            "relation_properties": {"rel_key": "rel_value"},
            "tail": "tail1",
            "tail_type": "type2",
            "tail_properties": {"key2": "value2"},
        },
        {
            "head": "head2",
            "head_type": "type3",
            "head_properties": {"key3": "value3"},
            "relation": "relation2",
            "relation_properties": {"rel_key2": "rel_value2"},
            "tail": "tail2",
            "tail_type": "type4",
            "tail_properties": {"key4": "value4"},
        },
    ]

    llm_client = MagicMock(spec=LLMClient)

    db = MagicMock()
    db.triple.aggregate.return_value.to_list = AsyncMock(
        return_value=mock_triples
    )
    mock_embed_triples.return_value = ["embedding1", "embedding2"]
    db.triple.bulk_write = AsyncMock(
        return_value=MagicMock(matched_count=len(triple_ids))
    )

    session = MagicMock()

    await update_triple_embeddings(
        db, llm_client, triple_ids, user_id, session=session
    )

    db.triple.aggregate.assert_called_once()
    mock_embed_triples.assert_awaited_once()
    db.triple.bulk_write.assert_awaited_once()


@pytest.mark.asyncio
@patch("whyhow_api.services.crud.triple.embed_triples", new_callable=AsyncMock)
async def test_update_triple_embeddings_no_triples(mock_embed_triples):
    triple_ids = [ObjectId(), ObjectId()]
    user_id = ObjectId()

    db = MagicMock()
    db.triple.aggregate.return_value.to_list = AsyncMock(return_value=[])
    mock_embed_triples.return_value = []

    session = MagicMock()

    with pytest.raises(ValueError, match="No triples found."):
        await update_triple_embeddings(
            db, MagicMock(spec=LLMClient), triple_ids, user_id, session=session
        )

    db.triple.aggregate.assert_called_once()
    mock_embed_triples.assert_not_called()
    db.triple.bulk_write.assert_not_called()
