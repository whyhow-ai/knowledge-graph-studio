import pytest
from pydantic import SecretStr

from whyhow_api.config import Settings, SettingsMongoDB


def test_settings_placeholder():
    pass


def test_openai_placeholder():
    pass


def test_mongodb_settings_uri_with_valid_credentials():
    settings = SettingsMongoDB(
        username="test_user",
        password=SecretStr("test_password"),
        host="test_host",
    )

    expected_uri = "mongodb+srv://test_user:test_password@test_host"
    assert settings.uri == expected_uri


def test_mongodb_settings_uri_with_invalid_credentials():
    # Test with missing username
    settings = SettingsMongoDB(password=SecretStr("pass"), host="localhost")
    with pytest.raises(ValueError):
        _ = settings.uri

    # Test with missing password
    settings = SettingsMongoDB(username="user", host="localhost")
    with pytest.raises(ValueError):
        _ = settings.uri

    # Test with missing host
    settings = SettingsMongoDB(username="user", password=SecretStr("pass"))
    with pytest.raises(ValueError):
        _ = settings.uri


def test_aws_settings(monkeypatch):
    monkeypatch.setenv("WHYHOW__AWS__S3__BUCKET", "test_bucket_123")
    monkeypatch.setenv("WHYHOW__AWS__S3__PRESIGNED_POST_EXPIRATION", "5555")
    monkeypatch.setenv("WHYHOW__AWS__S3__PRESIGNED_POST_MAX_BYTES", "9999")

    settings = Settings()

    assert settings.aws.s3.bucket == "test_bucket_123"
    assert settings.aws.s3.presigned_post_expiration == 5555
    assert settings.aws.s3.presigned_post_max_bytes == 9999
