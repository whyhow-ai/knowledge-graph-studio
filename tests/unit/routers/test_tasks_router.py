from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest
from bson import ObjectId
from dateutil import parser

from whyhow_api.dependencies import get_db, get_user


class TestTasks:

    @pytest.fixture
    def task_document_mock(self):
        now = datetime.now(timezone.utc)
        return {
            "_id": ObjectId(),
            "created_at": now,
            "updated_at": now,
            "created_by": ObjectId(),
            "start_time": now,
            "end_time": now + timedelta(hours=1),
            "status": "success",
            "result": "Graph constructed",
        }

    def test_get_task_successful(self, client, task_document_mock):
        fake_find_one = AsyncMock(return_value=task_document_mock)
        fake_db = AsyncMock()
        fake_db.task.find_one = fake_find_one

        client.app.dependency_overrides[get_db] = lambda: fake_db
        client.app.dependency_overrides[get_user] = lambda: task_document_mock[
            "created_by"
        ]

        response = client.get(f"/tasks/{task_document_mock['_id']}")
        assert response.status_code == 200

        data = response.json()
        assert data["message"] == "Task retrieved successfully."
        assert data["status"] == "success"
        assert data["count"] == 1
        assert data["task"]["_id"] == str(task_document_mock["_id"])
        assert data["task"]["created_by"] == str(
            task_document_mock["created_by"]
        )
        assert (
            parser.isoparse(data["task"]["created_at"])
            == task_document_mock["created_at"]
        )
        assert (
            parser.isoparse(data["task"]["updated_at"])
            == task_document_mock["updated_at"]
        )
        assert (
            parser.isoparse(data["task"]["start_time"])
            == task_document_mock["start_time"]
        )
        assert (
            parser.isoparse(data["task"]["end_time"])
            == task_document_mock["end_time"]
        )
        assert data["task"]["status"] == task_document_mock["status"]
        assert data["task"]["result"] == task_document_mock["result"]

    def test_get_task_not_found(self, client, task_document_mock):
        fake_find_one = AsyncMock(return_value=None)
        fake_db = AsyncMock()
        fake_db.task.find_one = fake_find_one

        client.app.dependency_overrides[get_db] = lambda: fake_db
        client.app.dependency_overrides[get_user] = lambda: task_document_mock[
            "created_by"
        ]

        response = client.get(f"/tasks/{task_document_mock['_id']}")
        assert response.status_code == 404

        data = response.json()
        assert data["detail"] == "Task not found."

    def test_get_task_invalid_id(self, client, task_document_mock):
        fake_db = AsyncMock()

        client.app.dependency_overrides[get_db] = lambda: fake_db
        client.app.dependency_overrides[get_user] = lambda: task_document_mock[
            "created_by"
        ]

        response = client.get("/tasks/invalid_id")
        assert response.status_code == 400

        data = response.json()
        assert data["detail"] == "Invalid task ID."
