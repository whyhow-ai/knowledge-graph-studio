from unittest.mock import AsyncMock

import pytest
from bson import ObjectId

from whyhow_api.dependencies import get_db, get_user
from whyhow_api.schemas.users import ProviderConfig


class TestSetProvidersDetails:

    # def test_get_api_key_successful(self, client, monkeypatch):

    #     # Get API Key
    #     fake_get_one_api_key = AsyncMock(return_value=APIKeyOutModel(api_key='1234567890abcdef1234567890abcdef'))
    #     monkeypatch.setattr("whyhow_api.services.crud.base.get_one", fake_get_one_api_key)

    #     # Validate API Key
    #     fake_model_validate = AsyncMock(return_value={'api_key': '1234567890abcdef1234567890abcdef'})
    #     monkeypatch.setattr("whyhow_api.schemas.users.APIKeyOutModel.model_validate", fake_model_validate)

    #     client.app.dependency_overrides[get_db] = lambda: AsyncMock()
    #     client.app.dependency_overrides[get_user] = lambda: ObjectId()

    #     # Request API Key
    #     response = client.get(f"/users/api_key")
    #     print(response)
    #     assert response.status_code == 200

    #     # Validate Response
    #     data = response.json()
    #     assert data["message"] == "API key retrieved successfully."
    #     assert data["status"] == "success"
    #     assert data["count"] == 1

    #     # Validate output
    #     expected_api_key_out = [{'api_key': '1234567890abcdef1234567890abcdef'}]
    #     assert expected_api_key_out == data["whyhow_api_key"]

    @pytest.mark.skip(reason="Requires review.")
    def test_set_providers_details_successful(self, client, monkeypatch):
        request = {
            "providers": [
                {
                    "type": "llm",
                    "value": "byo-openai",
                    "api_key": "api key 1",
                    "metadata": {
                        "byo-openai": {
                            "language_model_name": "3",
                            "embedding_name": "4",
                        },
                        "byo-azure-openai": {
                            "api_version": "1",
                            "azure_endpoint": "2",
                            "language_model_name": "3",
                            "embedding_name": "4",
                        },
                    },
                }
            ]
        }
        providers_details_request_mock = ProviderConfig(**request)

        fake_db = AsyncMock()
        fake_db.user.update_one = AsyncMock()
        fake_db.user.update_one.return_value = {"modified_count": 1}

        client.app.dependency_overrides[get_db] = lambda: fake_db
        client.app.dependency_overrides[get_user] = lambda: ObjectId()

        response = client.put(
            "/users/set_providers_details",
            json=providers_details_request_mock.model_dump(),
        )
        assert response.status_code == 200

        data = response.json()
        assert data["message"] == "Providers details set successfully"
        assert data["status"] == "success"

    def test_set_providers_details_failure(self, client, monkeypatch):
        request = {
            "providers": [
                {
                    "type": "llm",
                    "value": "byo-openai",
                    "api_key": "api key 1",
                    "metadata": {
                        "byo-openai": {
                            "language_model_name": "3",
                            "embedding_name": "4",
                        },
                        "byo-azure-openai": {
                            "api_version": "1",
                            "azure_endpoint": "2",
                            "language_model_name": "3",
                            "embedding_name": "4",
                        },
                    },
                }
            ]
        }
        providers_details_request_mock = ProviderConfig(**request)

        fake_db = AsyncMock()
        fake_db.user.update_one = AsyncMock()
        fake_db.user.update_one.side_effect = Exception("Database error")

        client.app.dependency_overrides[get_db] = lambda: fake_db
        client.app.dependency_overrides[get_user] = lambda: ObjectId()

        response = client.put(
            "/users/set_providers_details",
            json=providers_details_request_mock.model_dump(),
        )
        assert response.status_code == 400

        data = response.json()
        assert data["detail"] == "Error setting providers details"
