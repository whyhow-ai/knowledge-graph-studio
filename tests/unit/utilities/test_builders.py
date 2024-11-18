from unittest.mock import AsyncMock, Mock

import pytest

from whyhow_api.models.common import Triple
from whyhow_api.utilities.builders import OpenAIBuilder


class TestOpenAIBuilder:
    @pytest.mark.skip(reason="Requires review and integration with new logic")
    @pytest.mark.asyncio
    async def test_fetch_triples(self):
        openai_client = AsyncMock()
        chunk = Mock()
        pattern = Mock()
        completions_config = Mock()

        response_mock = AsyncMock()
        response_mock.choices.__getitem__.return_value.message.content = (
            '["Alice,knows,Bob", "Bob,hates,Dan"]'
        )
        openai_client.chat.completions.create.return_value = response_mock

        triples = await OpenAIBuilder.fetch_triples(
            openai_client=openai_client,
            chunk=chunk,
            pattern=pattern,
            completions_config=completions_config,
        )

        assert len(triples) == 2
        assert Triple(head="Alice", relation="knows", tail="Bob") in triples
        assert Triple(head="Bob", relation="hates", tail="Dan") in triples

    @pytest.mark.skip(reason="Requires review and integration with new logic")
    @pytest.mark.asyncio
    async def test_fetch_triples_error(self):
        openai_client = AsyncMock()
        chunk = Mock()
        pattern = Mock()
        completions_config = Mock()

        openai_client.chat.completions.create.side_effect = Exception(
            "Some error"
        )

        triples = await OpenAIBuilder.fetch_triples(
            openai_client=openai_client,
            chunk=chunk,
            pattern=pattern,
            completions_config=completions_config,
        )

        assert triples == []

    @pytest.mark.skip(reason="Requires review and integration with new logic")
    @pytest.mark.asyncio
    async def test_extract_zeroshot_triples(self):
        openai_client = AsyncMock()
        chunk = Mock()
        prompt = "Some prompt"
        completions_config = Mock()

        response_mock = AsyncMock()
        response_mock.choices.__getitem__.return_value.message.content = (
            '["Alice,knows,Bob", "Bob,hates,Dan"]'
        )
        openai_client.chat.completions.create.return_value = response_mock

        triples = await OpenAIBuilder.extract_zeroshot_triples(
            openai_client=openai_client,
            chunk=chunk,
            prompt=prompt,
            completions_config=completions_config,
        )

        assert len(triples) == 2
        assert Triple(head="Alice", relation="knows", tail="Bob") in triples
        assert Triple(head="Bob", relation="hates", tail="Dan") in triples

    @pytest.mark.skip(reason="Requires review and integration with new logic")
    @pytest.mark.asyncio
    async def test_extract_zeroshot_triples_error(self):
        openai_client = AsyncMock()
        chunk = Mock()
        prompt = "Some prompt"
        completions_config = Mock()

        openai_client.chat.completions.create.side_effect = Exception(
            "Some error"
        )

        triples = await OpenAIBuilder.extract_zeroshot_triples(
            openai_client=openai_client,
            chunk=chunk,
            prompt=prompt,
            completions_config=completions_config,
        )

        assert triples == []

    @pytest.mark.skip(reason="Requires review and integration with new logic")
    @pytest.mark.asyncio
    async def test_parse_response_into_triples(self):
        response_mock = Mock()
        response_mock.choices.__getitem__.return_value.message.content = (
            '["Alice,knows,Bob", "Bob,hates,Dan"]'
        )

        triples = OpenAIBuilder.parse_response_into_triples(
            response=response_mock
        )

        assert len(triples) == 2
        assert Triple(head="Alice", relation="knows", tail="Bob") in triples
        assert Triple(head="Bob", relation="hates", tail="Dan") in triples

    @pytest.mark.skip(reason="Requires review and integration with new logic")
    @pytest.mark.asyncio
    async def test_parse_response_into_triples_error(self):
        response_mock = Mock()
        response_mock.choices.__getitem__.return_value.message.content = None

        triples = OpenAIBuilder.parse_response_into_triples(
            response=response_mock
        )

        assert triples == []

    @pytest.mark.skip(reason="Requires review and integration with new logic")
    @pytest.mark.asyncio
    async def test_extract_triples(self):
        openai_client = AsyncMock()
        chunk = Mock()
        completions_config = Mock()
        prompts = ["Some prompt", "Another prompt"]

        response_mock = AsyncMock()
        response_mock.choices.__getitem__.return_value.message.content = (
            '["Alice,knows,Bob", "Bob,hates,Dan"]'
        )
        openai_client.chat.completions.create.return_value = response_mock

        triples = await OpenAIBuilder.extract_triples(
            openai_client=openai_client,
            chunk=chunk,
            completions_config=completions_config,
            prompts=prompts,
        )

        assert len(triples) == 2
        assert Triple(head="Alice", relation="knows", tail="Bob") in triples
        assert Triple(head="Bob", relation="hates", tail="Dan") in triples

    @pytest.mark.skip(reason="Requires review and integration with new logic")
    @pytest.mark.asyncio
    async def test_extract_triples_error(self):
        openai_client = AsyncMock()
        chunk = Mock()
        completions_config = Mock()
        prompts = ["Some prompt", "Another prompt"]

        openai_client.chat.completions.create.side_effect = Exception(
            "Some error"
        )

        triples = await OpenAIBuilder.extract_triples(
            openai_client=openai_client,
            chunk=chunk,
            completions_config=completions_config,
            prompts=prompts,
        )

        assert triples == []
