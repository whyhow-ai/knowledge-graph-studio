import pytest

from whyhow_api.utilities.routers import clean_url


@pytest.mark.parametrize(
    "url, expected",
    [
        ("/path/to/resource/60b8d295f9657c0c88d37d9b/", "/path/to/resource"),
        ("/60b8d295f9657c0c88d37d9b/", "/"),
        ("/path/to/resource/", "/path/to/resource"),
        (
            "/60b8d295f9657c0c88d37d9b/path/60b8d295f9657c0c88d37d9b/resource/",
            "/path/resource",
        ),
        ("/path/to/resource", "/path/to/resource"),
        ("", ""),
    ],
)
def test_clean_url(url, expected):
    result = clean_url(url)
    assert result == expected
