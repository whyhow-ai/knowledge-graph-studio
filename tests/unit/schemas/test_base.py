from abc import ABC
from datetime import datetime

import pytest
from bson import ObjectId
from pydantic import BaseModel, ConfigDict, ValidationError

from whyhow_api.schemas.base import (
    AfterAnnotatedObjectId,
    AnnotatedObjectId,
    BaseDocument,
    BaseRequest,
    BaseResponse,
    DeleteResponseModel,
    FilterBody,
    Graph_Status,
    Status,
)


class Model(BaseModel):
    status: Status | None = None
    graph_status: Graph_Status | None = None
    annotated_object_id: AnnotatedObjectId | None = None
    after_annotated_object_id: AfterAnnotatedObjectId | None = None

    model_config = ConfigDict(
        populate_by_name=True,
        use_enum_values=True,
        from_attributes=True,
        arbitrary_types_allowed=True,
    )


def test_status():
    model = Model(status="success")
    assert model.status == "success"
    with pytest.raises(ValidationError):
        Model(status="invalid")


def test_graph_status():
    model = Model(graph_status="ready")
    assert model.graph_status == "ready"
    with pytest.raises(ValidationError):
        Model(graph_status="invalid")


def test_annotated_object_id():
    mock_id = ObjectId()
    model = Model(annotated_object_id=str(mock_id))
    assert model.annotated_object_id == str(mock_id)


def test_after_annotated_object_id_with_valid_string():
    mock_id = ObjectId()
    model = Model(after_annotated_object_id=str(mock_id))
    assert isinstance(model.after_annotated_object_id, ObjectId)


def test_after_annotated_object_id_with_valid_object_id():
    mock_id = ObjectId()
    model = Model(after_annotated_object_id=mock_id)
    assert isinstance(model.after_annotated_object_id, ObjectId)


def test_after_annotated_object_id_with_invalid_string():
    with pytest.raises(ValidationError):
        Model(after_annotated_object_id="invalid")


def test_after_annotated_object_id_with_invalid_type():
    with pytest.raises(ValidationError):
        Model(after_annotated_object_id={})


def test_after_annotated_object_id_none():
    model = Model(after_annotated_object_id=None)
    assert model.after_annotated_object_id is None


def test_after_annotated_object_id_with_integer():
    with pytest.raises(ValidationError):
        Model(after_annotated_object_id=123)


class TestFilterBody:
    def test_create_filter_body_with_dict(self):
        filter_body = FilterBody(filters={"key1": "value1", "key2": "value2"})
        assert filter_body.filters == {"key1": "value1", "key2": "value2"}

    def test_create_filter_body_with_none(self):
        filter_body = FilterBody(filters=None)
        assert filter_body.filters is None

    def test_create_filter_body_with_invalid_data(self):
        with pytest.raises(ValidationError):
            FilterBody(filters="invalid data")


class TestDeleteResponseModel:
    def test_create_delete_response_model(self):
        delete_response_model = DeleteResponseModel(
            message="Deleted successfully", status="success"
        )
        assert delete_response_model.message == "Deleted successfully"
        assert delete_response_model.status == "success"

    def test_create_delete_response_model_with_invalid_status(self):
        with pytest.raises(ValidationError):
            DeleteResponseModel(
                message="Deleted successfully", status="invalid"
            )

    def test_create_delete_response_model_with_invalid_message(self):
        with pytest.raises(ValidationError):
            DeleteResponseModel(message=123, status="success")


class TestBaseDocument:
    def test_create_base_document(self, user_id_mock):
        base_document = BaseDocument(id=ObjectId(), created_by=user_id_mock)
        assert isinstance(base_document.id, ObjectId)
        assert isinstance(base_document.created_at, datetime)
        assert isinstance(base_document.updated_at, datetime)
        assert isinstance(base_document.created_by, ObjectId)

    def test_create_base_document_with_invalid_id(self, user_id_mock):
        with pytest.raises(ValidationError):
            BaseDocument(id="invalid", created_by=user_id_mock)

    def test_create_base_document_with_invalid_created_by(self):
        with pytest.raises(ValidationError):
            BaseDocument(id=ObjectId(), created_by="invalid")


class TestBaseRequest:
    def test_create_base_request(self):
        base_request = BaseRequest()
        assert isinstance(base_request, BaseModel)
        assert isinstance(base_request, ABC)

    def test_base_request_forbids_extra_fields(self):
        with pytest.raises(ValidationError):
            BaseRequest(extra_field="extra")


class TestBaseResponse:
    def test_create_base_response(self):
        base_response = BaseResponse(message="message", status="success")
        assert isinstance(base_response, BaseModel)
        assert isinstance(base_response, ABC)

    def test_base_response_ignores_extra_fields(self):
        base_response = BaseResponse(
            message="message", status="success", extra_field="extra"
        )
        assert not hasattr(base_response, "extra_field")
