from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from bson import ObjectId
from pydantic import BaseModel

from whyhow_api.schemas.documents import (
    DocumentDocumentModel,
    DocumentMetadata,
)
from whyhow_api.services.crud.document import (
    delete_document_from_s3,
    get_document_content,
    update_document,
)


@pytest.mark.asyncio
@patch("boto3.client")
async def test_get_documents_from_s3(mock_boto3_client, db_mock, user_id_mock):
    # TODO: Update when file extension is added to document model
    document_id_mock1 = ObjectId()
    document_id_mock2 = ObjectId()
    document_ids_mock = [document_id_mock1, document_id_mock2]

    # Mock db.document.find_one
    async def find_one_mock(query):
        if query["_id"] == document_id_mock1:
            return DocumentDocumentModel(
                _id=document_id_mock1,
                created_by=user_id_mock,
                workspaces=[ObjectId()],
                status="uploaded",
                metadata=DocumentMetadata(
                    format="csv", size=0, filename="test1.csv"
                ),
            ).model_dump()
        elif query["_id"] == document_id_mock2:
            return DocumentDocumentModel(
                _id=document_id_mock2,
                created_by=user_id_mock,
                workspaces=[ObjectId()],
                status="uploaded",
                metadata=DocumentMetadata(
                    format="pdf", size=0, filename="test2.pdf"
                ),
            ).model_dump()

    db_mock.document.find_one = AsyncMock(side_effect=find_one_mock)

    mock_s3_client = MagicMock()
    mock_boto3_client.return_value = mock_s3_client
    mock_s3_client.get_object.return_value = {
        "Body": MagicMock(read=MagicMock(return_value=b"mock_content"))
    }

    documents = [
        await get_document_content(
            document_id=d,
            user_id=user_id_mock,
            db=db_mock,
            bucket="test_bucket",
        )
        for d in document_ids_mock
    ]

    assert len(documents) == 2
    assert documents[0][0] == b"mock_content"
    assert documents[1][0] == b"mock_content"
    assert isinstance(documents[0][1], BaseModel)
    assert documents[0][1].metadata.filename == "test1.csv"
    assert documents[0][1].created_by == user_id_mock
    assert documents[1][1].metadata.filename == "test2.pdf"
    assert documents[1][1].created_by == user_id_mock


@pytest.mark.asyncio
@patch("boto3.client")
async def test_delete_document_from_s3(mock_boto3_client):
    # Mock data
    user_id = ObjectId()
    filename = "test_file.txt"
    settings_mock = MagicMock()
    settings_mock.aws.s3.bucket = "test_bucket"

    # Mock the S3 client and its delete_object method
    mock_s3_client = MagicMock()
    mock_boto3_client.return_value = mock_s3_client
    mock_s3_client.delete_object.return_value = None

    # Call the method under test
    await delete_document_from_s3(user_id, filename, settings_mock)

    mock_boto3_client.assert_called_once_with("s3")
    mock_s3_client.delete_object.assert_called_once_with(
        Bucket="test_bucket",
        Key=f"{user_id}/{filename}",
    )


@pytest.mark.asyncio
@patch("boto3.client")
async def test_delete_document_from_s3_with_exception(mock_boto3_client):
    user_id = ObjectId()
    filename = "test_file.txt"
    settings_mock = MagicMock()
    settings_mock.aws.s3.bucket = "test_bucket"

    mock_s3_client = MagicMock()
    mock_boto3_client.return_value = mock_s3_client
    mock_s3_client.delete_object.side_effect = Exception(
        "Test exception message"
    )

    await delete_document_from_s3(user_id, filename, settings_mock)

    mock_boto3_client.assert_called_once_with("s3")
    mock_s3_client.delete_object.assert_called_once_with(
        Bucket="test_bucket",
        Key=f"{user_id}/{filename}",
    )


@pytest.mark.asyncio
async def test_update_document_successful(user_id_mock):
    document_id = ObjectId()
    workspace_id = ObjectId()

    fake_document = DocumentDocumentModel(
        id=document_id,
        created_by=user_id_mock,
        workspaces=[workspace_id],
        status="uploaded",
        metadata=DocumentMetadata(format="csv", size=0, filename="test1.csv"),
    )

    collection_mock = AsyncMock()
    collection_mock.__getitem__.return_value = collection_mock
    collection_mock.find_one_and_update = AsyncMock(
        return_value=fake_document.model_dump(by_alias=True)
    )

    updated_document = await update_document(
        user_id=user_id_mock,
        collection=collection_mock,
        document=fake_document,
        document_id=document_id,
        workspace_id=workspace_id,
    )

    assert updated_document.id == fake_document.id
    assert updated_document.created_by == user_id_mock
    assert workspace_id in updated_document.workspaces
    assert updated_document.status == "uploaded"
    assert updated_document.metadata.format == "csv"
    assert updated_document.metadata.size == 0
    assert updated_document.metadata.filename == "test1.csv"
