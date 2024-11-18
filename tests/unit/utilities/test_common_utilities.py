import string
from unittest.mock import AsyncMock, Mock

import pytest

from whyhow_api.utilities.common import (
    compress_triples,
    count_frequency,
    dict_to_tuple,
    embed_texts,
    remove_punctuation,
    tuple_to_dict,
)


def test_load_schema_and_patterns_placeholder():
    pass


class TestCompressTriples:
    def test_compress_triples(self):
        triples = [
            ("Jerry", "friends with", "Kramer"),
            ("Jerry", "friends with", "Elaine"),
            ("Jerry", "friends with", "George"),
        ]
        compressed_output = compress_triples(triples)
        expected_output = "Jerry friends with Elaine, George, Kramer"
        assert compressed_output == expected_output

    def test_compress_triples_with_duplicates(self):
        triples = [
            ("Jerry", "friends with", "Kramer"),
            ("Jerry", "friends with", "Elaine"),
            ("Jerry", "friends with", "George"),
            ("Jerry", "friends with", "Kramer"),  # Duplicate entry
        ]
        compressed_output = compress_triples(triples)
        expected_output = "Jerry friends with Elaine, George, Kramer"
        assert compressed_output == expected_output

    def test_compress_triples_with_different_relations(self):
        triples = [
            ("Jerry", "friends with", "Kramer"),
            ("Jerry", "enemies with", "Newman"),
            ("Jerry", "acquaintances with", "Elaine"),
        ]
        compressed_output = compress_triples(triples)
        expected_output = "Jerry acquaintances with Elaine\nJerry enemies with Newman\nJerry friends with Kramer"  # noqa: E501
        assert compressed_output == expected_output

    def test_compress_triples_empty_input(self):
        triples = []
        compressed_output = compress_triples(triples)
        assert compressed_output == ""

    def test_compress_triples_single_triple(self):
        triples = [("Jerry", "friends with", "Kramer")]
        compressed_output = compress_triples(triples)
        expected_output = "Jerry friends with Kramer"
        assert compressed_output == expected_output

    def test_compress_triples_with_spaces_in_relation(self):
        triples = [("Jerry", "friends_with", "Kramer")]
        compressed_output = compress_triples(triples)
        expected_output = "Jerry friends with Kramer"
        assert compressed_output == expected_output

    def test_compress_triples_with_spaces_in_entities(self):
        triples = [("Jerry Seinfeld", "friends with", "Kramer Smith")]
        compressed_output = compress_triples(triples)
        expected_output = "Jerry Seinfeld friends with Kramer Smith"
        assert compressed_output == expected_output


class TestRemovePunctuation:
    def test_remove_punctuation_basic(self):
        text = "Hello, world!"
        expected_result = "Hello world"
        assert remove_punctuation(text) == expected_result

    def test_remove_punctuation_empty_string(self):
        text = ""
        expected_result = ""
        assert remove_punctuation(text) == expected_result

    def test_remove_punctuation_no_punctuation(self):
        text = "This is a test without punctuation"
        expected_result = text
        assert remove_punctuation(text) == expected_result

    def test_remove_punctuation_all_punctuation(self):
        text = string.punctuation
        expected_result = ""
        assert remove_punctuation(text) == expected_result

    def test_remove_punctuation_mixed(self):
        text = "Hello, world! How are you today?"
        expected_result = "Hello world How are you today"
        assert remove_punctuation(text) == expected_result


class TestCountFrequency:
    def test_count_frequency_basic(self):
        search_str = "hello"
        data_dict = {
            "1": "Hello world! Hello there!",
            "2": "Howdy, hello, hi!",
        }
        expected_result = {"1": 2, "2": 1}
        assert count_frequency(search_str, data_dict) == expected_result

    def test_count_frequency_no_occurrences(self):
        search_str = "python"
        data_dict = {"1": "Hello world!", "2": "Howdy, hello, hi!"}
        expected_result = {}
        assert count_frequency(search_str, data_dict) == expected_result

    def test_count_frequency_empty_input(self):
        search_str = ""
        data_dict = {}
        expected_result = {}
        assert count_frequency(search_str, data_dict) == expected_result

    def test_count_frequency_case_sensitive(self):
        search_str = "Hello"
        data_dict = {"1": "hello world!", "2": "Howdy, hello, hi!"}
        expected_result = {"1": 1, "2": 1}
        assert count_frequency(search_str, data_dict) == expected_result

    def test_count_frequency_punctuation(self):
        search_str = "world"
        data_dict = {"1": "Hello, world!", "2": "world, world!"}
        expected_result = {"1": 1, "2": 2}
        assert count_frequency(search_str, data_dict) == expected_result


class MockResponse:
    def __init__(self, embedding):
        self.embedding = embedding


class TestEmbedTexts:
    @pytest.mark.asyncio
    async def test_embed_texts_basic(self, monkeypatch):
        # Create mock data response
        mock_response_data = Mock()
        mock_response_data.data = [
            MockResponse([0.1, 0.2, 0.3]),
            MockResponse([0.4, 0.5, 0.6]),
        ]

        # Mock the llm_client with AsyncMock
        llm_client_mock = AsyncMock()
        # Set the return value of llm_client.client.embeddings.create
        llm_client_mock.client.embeddings.create.return_value = (
            mock_response_data
        )

        # Test input
        texts = ["Hello", "World"]

        # Calling the function
        embeddings = await embed_texts(llm_client_mock, texts)

        # Assertions
        assert len(embeddings) == 2
        assert embeddings[0] == [0.1, 0.2, 0.3]
        assert embeddings[1] == [0.4, 0.5, 0.6]

    @pytest.mark.asyncio
    async def test_embed_texts_large_input(self):
        # Create mock data response
        mock_response_data = Mock()
        mock_response_data.data = [
            MockResponse([0.1, 0.2, 0.3]),
            MockResponse([0.4, 0.5, 0.6]),
        ]

        # Mocking the LLM client
        llm_client_mock = AsyncMock()
        # Set the return value of llm_client.client.embeddings.create
        llm_client_mock.client.embeddings.create.return_value = (
            mock_response_data
        )

        # Test input with more than 2048 items
        texts = ["Text"] * 2049

        # Calling the function
        with pytest.raises(RuntimeError):
            await embed_texts(
                llm_client=llm_client_mock, texts=texts, batch_size=2049
            )


class TestDictToTuple:
    def test_dict_to_tuple_simple(self):
        d = {"key1": "value1", "key2": "value2"}
        expected = (("key1", "value1"), ("key2", "value2"))
        result = dict_to_tuple(d)
        assert result == expected

    def test_dict_to_tuple_nested(self):
        d = {
            "key1": "value1",
            "key2": {"subkey1": "subvalue1", "subkey2": "subvalue2"},
        }
        expected = (
            ("key1", "value1"),
            ("key2", (("subkey1", "subvalue1"), ("subkey2", "subvalue2"))),
        )
        result = dict_to_tuple(d)
        assert result == expected

    def test_dict_to_tuple_empty(self):
        d = {}
        expected = ()
        result = dict_to_tuple(d)
        assert result == expected

    def test_dict_to_tuple_complex(self):
        d = {
            "key1": "value1",
            "key2": {
                "subkey1": "subvalue1",
                "subkey2": {"subsubkey1": "subsubvalue1"},
            },
        }
        expected = (
            ("key1", "value1"),
            (
                "key2",
                (
                    ("subkey1", "subvalue1"),
                    ("subkey2", (("subsubkey1", "subsubvalue1"),)),
                ),
            ),
        )
        result = dict_to_tuple(d)
        assert result == expected


class TestTupleToDict:
    def test_tuple_to_dict_simple(self):
        t = (("key1", "value1"), ("key2", "value2"))
        expected = {"key1": "value1", "key2": "value2"}
        result = tuple_to_dict(t)
        assert result == expected

    def test_tuple_to_dict_nested(self):
        t = (
            ("key1", "value1"),
            ("key2", (("subkey1", "subvalue1"), ("subkey2", "subvalue2"))),
        )
        expected = {
            "key1": "value1",
            "key2": {"subkey1": "subvalue1", "subkey2": "subvalue2"},
        }
        result = tuple_to_dict(t)
        assert result == expected

    def test_tuple_to_dict_empty(self):
        t = ()
        expected = {}
        result = tuple_to_dict(t)
        assert result == expected

    def test_tuple_to_dict_complex(self):
        t = (
            ("key1", "value1"),
            (
                "key2",
                (
                    ("subkey1", "subvalue1"),
                    ("subkey2", (("subsubkey1", "subsubvalue1"),)),
                ),
            ),
        )
        expected = {
            "key1": "value1",
            "key2": {
                "subkey1": "subvalue1",
                "subkey2": {"subsubkey1": "subsubvalue1"},
            },
        }
        result = tuple_to_dict(t)
        assert result == expected
