"""Graph builders."""

import asyncio
import inspect
import json
import logging
import time
from abc import ABC, abstractmethod
from json.decoder import JSONDecodeError
from typing import Any, List, Optional, Tuple

# import backoff
# import openai
import logfire
import spacy
import spacy.cli
from openai.types.chat import ChatCompletion

from whyhow_api.dependencies import LLMClient
from whyhow_api.models.common import (
    Entity,
    OpenAICompletionsConfig,
    OpenAIDirectivesConfig,
    SchemaEntity,
    SchemaRelation,
    SchemaTriplePattern,
    TextWithEntities,
    Triple,
    TriplePattern,
)
from whyhow_api.schemas.base import ErrorDetails
from whyhow_api.schemas.chunks import ChunkDocumentModel
from whyhow_api.schemas.schemas import GeneratedSchema
from whyhow_api.utilities.config import (
    create_schema_guided_graph_prompt,
    create_zeroshot_graph_prompt,
)

logger = logging.getLogger(__name__)


class TextEntityExtractor(ABC):
    """Text entity extractor interface."""

    @abstractmethod
    def load(self) -> None:
        """Load the underlying model necessary for entity extraction."""

    @abstractmethod
    def extract_entities(self, text: str) -> TextWithEntities:
        """
        Extract entities from the given text.

        Parameters
        ----------
        text : str
            The input text from which to extract entities.

        Returns
        -------
        TextWithEntities
            A data structure containing the original text and the extracted entities.
        """


class SpacyEntityExtractor(TextEntityExtractor):
    """A text entity extractor based on spaCy's named entity recognition (NER)."""

    def __init__(
        self,
        model_name: str = "en_core_web_sm",
        model_disables: List[str] = ["parser", "lemmatizer"],
    ):
        self.model_name = model_name
        self.model_disables = model_disables
        self.model: spacy.language.Language | None = None

    def download_and_load_spacy_model(self) -> spacy.language.Language:
        """Attempt to load a spaCy model by its name.

        If the model is not found, it tries to download it
        and then loads it. Additional keyword arguments can be passed to specify which pipeline components
        to disable or for other loading options.

        Returns
        -------
        Language
            The loaded spaCy model.
        """
        try:
            return spacy.load(
                name=self.model_name, disable=self.model_disables
            )
        except OSError:
            print(f"{self.model_name} model not found. Downloading...")
            spacy.cli.download(self.model_name)
            return spacy.load(
                name=self.model_name, disable=self.model_disables
            )

    def load(self) -> None:
        """Load the spaCy model specified by model_name."""
        self.model = self.download_and_load_spacy_model()

    def extract_entities(self, text: str) -> TextWithEntities:
        """Extract entities using the loaded spaCy model."""
        if self.model is None:
            self.load()
        doc = self.model(text)  # type: ignore[misc]
        entities = [
            Entity(text=ent.text, label=ent.label_) for ent in doc.ents
        ]
        return TextWithEntities(text=text, entities=entities)


class OpenAIBuilder:
    """OpenAI API builder."""

    def __init__(
        self,
        llm_client: LLMClient,
        seed_entity_extractor: type[TextEntityExtractor],
    ):
        self.llm_client = llm_client
        self.seed_entity_extractor = seed_entity_extractor()

    # @backoff.on_exception(
    #     backoff.expo, openai.RateLimitError, max_time=60, max_tries=5
    # )
    @staticmethod
    async def fetch_triples(
        llm_client: LLMClient,
        chunk: ChunkDocumentModel,
        pattern: SchemaTriplePattern,
        completions_config: OpenAICompletionsConfig,
        # **kwargs,
    ) -> List[Triple] | None:
        """Fetch triples based on a schema pattern using the OpenAI API."""
        _content_str = create_schema_guided_graph_prompt(
            text=chunk.content, pattern=pattern
        )

        # Logfire trace of LLM client
        logfire.instrument_openai(llm_client.client)

        response = None
        try:
            if llm_client.metadata.language_model_name:
                completions_config.model = (
                    llm_client.metadata.language_model_name
                )
            response = await llm_client.client.chat.completions.create(
                messages=[{"role": "system", "content": _content_str}],
                **completions_config.model_dump(),
                # **kwargs,
            )
        except Exception as e:
            logger.error(f"Failed to fetch triple: {e}")

        if response is not None:
            message_content = response.choices[0].message.content
        else:
            return None

        res_entities = []
        try:
            response_text = response.choices[0].message.content.strip()

            if response_text.startswith("```") and response_text.endswith(
                "```"
            ):
                response_text = response_text[7:-3].strip()

            res_entities = json.loads(response_text)

        except JSONDecodeError as je:
            logger.error(
                f"Failed to parse message content - {je}: {message_content}"
            )
        except Exception as e:
            logger.error(f"Unexpected error parsing message content: {e}")

        if isinstance(res_entities, list):
            triples = []
            for entities in res_entities:
                head, tail = entities
                triples.append(
                    Triple(
                        head=head,
                        head_type=pattern.head.name,
                        head_properties={"chunks": [chunk.id]},
                        relation=pattern.relation.name,
                        relation_properties={"chunks": [chunk.id]},
                        tail=tail,
                        tail_type=pattern.tail.name,
                        tail_properties={"chunks": [chunk.id]},
                    )
                )
        return triples

    @staticmethod
    def parse_response_into_triples(response: ChatCompletion) -> List[Triple]:
        """Parse OpenAI response.

        Parse the response from the OpenAI API into a list
        of Triple objects.
        """
        try:
            if response is None or response.choices is None:
                return []
            message_content = response.choices[0].message.content
            if message_content is None:
                return []
            res_triples = json.loads(message_content)
            if not isinstance(res_triples, list):
                return []
            return [
                Triple(
                    head=t[0],
                    relation=t[1],
                    tail=t[2],
                )
                for t in (
                    t.strip().split(",")
                    for t in res_triples
                    if isinstance(t, str) and t.count(",") == 2
                )
            ]
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from response: {e}")
            return []
        except Exception as e:
            logger.error(f"Error processing response into triples: {e}")
            return []

    @staticmethod
    async def extract_zeroshot_triples(
        llm_client: LLMClient,
        chunk: ChunkDocumentModel,
        prompt: str,
        completions_config: OpenAICompletionsConfig,
    ) -> List[Triple] | None:
        """Extract triples using zero-shot prompts."""
        # Logfire trace of LLM client
        logfire.instrument_openai(llm_client.client)

        if llm_client.metadata.language_model_name:
            completions_config.model = llm_client.metadata.language_model_name
        extract_triples_response = (
            await llm_client.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": create_zeroshot_graph_prompt(
                            text=chunk.content, context=prompt
                        ),
                    }
                ],
                **completions_config.model_dump(),
            )
        )
        # logger.info(f"extract_triples_response: {extract_triples_response}")
        return OpenAIBuilder.parse_response_into_triples(
            response=extract_triples_response
        )

    @staticmethod
    async def extract_triples(
        llm_client: LLMClient,
        chunk: ChunkDocumentModel,
        completions_config: OpenAICompletionsConfig,
        patterns: List[SchemaTriplePattern],
    ) -> Optional[list[Triple]]:
        """
        Extract triples.

        Extracts triples from a text chunk using patterns, processing them in parallel.

        Parameters
        ----------
        llm_client : LLMClient
            The OpenAI or Azure OpenAI client instance.
        chunk : ChunkDocumentModel
            The chunk document model for triple extraction.
        completions_config : OpenAICompletionsConfig
            Configuration for OpenAI completion requests.
        patterns : List[SchemaTriplePattern] | None, optional
            A list of schema triple patterns for guided triple
            extraction, defaults to None.

        Returns
        -------
        List[Triple] | None
            A list of extracted triples if successful, or None
            if an error occurs.

        Raises
        ------
        ValueError
            If neither prompts nor patterns are provided, indicating
            that the necessary parameters for extraction are missing.
        """
        try:

            # Logfire trace of LLM client
            logfire.instrument_openai(llm_client.client)

            task_list = []

            if patterns:
                task_source: List[SchemaTriplePattern] = patterns
                extractor = OpenAIBuilder.fetch_triples
            else:
                raise ValueError("Patterns must be provided for extraction")

            # Generate tasks
            for item in task_source:
                task = extractor(
                    llm_client,
                    chunk,
                    item,
                    completions_config,
                )
                task_list.append(task)

            # Gather tasks and aggregate results
            triple_results = await asyncio.gather(*task_list)

            # Ensure triple_results does not contain None
            triples = [
                item
                for sublist in triple_results
                if sublist is not None
                for item in sublist
            ]

            return triples

        except Exception as e:
            logger.error(f"Failed to extract triples: {e}")
            return None

    # @staticmethod
    # async def extract_triples_from_batch(
    #     llm_client: LLMClient,
    #     chunks: List[ChunkDocumentModel],
    #     completions_config: OpenAICompletionsConfig,
    #     patterns: List[SchemaTriplePattern] | None = None,
    #     prompts: List[str] | None = None,
    # ) -> List[Triple]:
    #     """
    #     Extract triples from a batch of chunks.

    #     Extracts triples from a batch of text chunks using either
    #     specified prompts or patterns.

    #     Parameters
    #     ----------
    #     llm_client : LLMClient
    #         The OpenAI or Azure OpenAI client instance.
    #     chunks : List[ChunkDocumentModel]
    #         A list of chunk document models for triple extraction.
    #     completions_config : OpenAICompletionsConfig
    #         Configuration for OpenAI completion requests.
    #     patterns : List[SchemaTriplePattern] | None, optional
    #         A list of schema triple patterns for guided triple
    #         extraction, defaults to None.
    #     prompts : List[str] | None, optional
    #         A list of prompts for zero-shot triple extraction,
    #         defaults to None.

    #     Returns
    #     -------
    #     List[Triple]
    #         A list of extracted triples.

    #     Raises
    #     ------
    #     ValueError
    #         If neither prompts nor patterns are provided, indicating
    #         that the necessary parameters for extraction are missing.

    #     Notes
    #     -----
    #     The function concatenates the content of all chunks and selects
    #     the appropriate extraction method based on the provided inputs.
    #     It assigns chunk IDs to the extracted triples for traceability.
    #     """
    #     combined_content = " ".join(
    #         [
    #             chunk.content
    #             for chunk in chunks
    #             if isinstance(chunk.content, str)
    #         ]
    #     )

    #     if patterns:
    #         triples = []
    #         for pattern in patterns:
    #             pattern_triples = await OpenAIBuilder.fetch_triples(
    #                 llm_client=llm_client,
    #                 chunk=ChunkDocumentModel(
    #                     id="batch",
    #                     content=combined_content,
    #                     data_type="string",
    #                     workspaces=[],
    #                     created_by=ObjectId(),  # Or pass a valid user ID
    #                 )
    #                 pattern=pattern,
    #                 completions_config=completions_config,
    #             )
    #             if pattern_triples:
    #                 triples.extend(pattern_triples)
    #     elif prompts:
    #         triples = await OpenAIBuilder.extract_zeroshot_triples(
    #             llm_client=llm_client,
    #             chunk=ChunkDocumentModel(
    #                 id="batch", content=combined_content, data_type="string"
    #             ),
    #             prompt=prompts[
    #                 0
    #             ],  # Assuming we use the first prompt for the batch
    #             completions_config=completions_config,
    #         )
    #     else:
    #         raise ValueError("Either patterns or prompts must be provided")

    #     # Assign chunk IDs to triples
    #     if triples is not None:
    #         for triple in triples:
    #         triple.head_properties["chunks"] = [chunk.id for chunk in chunks]
    #         triple.relation_properties["chunks"] = [
    #             chunk.id for chunk in chunks
    #         ]
    #         triple.tail_properties["chunks"] = [chunk.id for chunk in chunks]

    #     return triples

    async def improve_entities_matching(
        self,
        extracted_entities: List[str],
        graph_entities: List[str],
        user_query: str,
        matched_entities: List[str],
        directives: OpenAIDirectivesConfig,
        completions_config: OpenAICompletionsConfig,
    ) -> List[str]:
        """
        Improves the matching of entities using the OpenAI API.

        This function formats a prompt using the provided entities and user query, sends the prompt to the OpenAI API,
        and returns the improved entities from the response.

        Parameters
        ----------
        extracted_entities : List[str]
            A list of entities extracted from the user's query.
        graph_entities : List[str]
            A list of entities in the graph.
        user_query : str
            The user's query.
        matched_entities : List[str]
            A list of entities that were matched in the graph.
        directives : OpenAIDirectivesConfig
            Configuration for OpenAI directives.
        completions_config : OpenAICompletionsConfig
            Configuration for OpenAI completions.

        Returns
        -------
        List[str]
            The improved entities from the response from the OpenAI API.

        Raises
        ------
        json.decoder.JSONDecodeError
            If the response from the OpenAI API cannot be parsed as JSON.
        """
        prompt = directives.improve_matched_entities.format(
            extracted_entities=json.dumps(extracted_entities, indent=2),
            graph_entities=json.dumps(graph_entities, indent=2),
            user_query=user_query,
            matched_entities=json.dumps(matched_entities, indent=2),
        )

        if self.llm_client.metadata.language_model_name:
            completions_config.model = (
                self.llm_client.metadata.language_model_name
            )
        response = await self.llm_client.client.chat.completions.create(
            messages=[{"role": "system", "content": prompt}],
            **completions_config.model_dump(),
        )

        formatted_response = response.choices[0].message.content.strip()

        formatted_response = formatted_response.replace(
            "```json\n", ""
        ).replace("\n```", "")

        try:
            improved_entities = json.loads(formatted_response)
        except json.decoder.JSONDecodeError:
            print("Failed to parse formatted response as JSON")
            improved_entities = []

        return improved_entities

    async def improve_relations_matching(
        self,
        extracted_relations: List[str],
        graph_relations: List[str],
        user_query: str,
        matched_relations: List[str],
        directives: OpenAIDirectivesConfig,
        completions_config: OpenAICompletionsConfig,
    ) -> List[str]:
        """
        Improves the matching of relations using the OpenAI API.

        This function formats a prompt using the provided relations and user query, sends the prompt to the OpenAI API,
        and returns the improved relations from the response.

        Parameters
        ----------
        extracted_relations : List[str]
            A list of relations extracted from the user's query.
        graph_relations : List[str]
            A list of relations in the graph.
        user_query : str
            The user's query.
        matched_relations : List[str]
            A list of relations that were matched in the graph.
        directives : OpenAIDirectivesConfig
            Configuration for OpenAI directives.
        completions_config : OpenAICompletionsConfig
            Configuration for OpenAI completions.

        Returns
        -------
        List[str]
            The improved relations from the response from the OpenAI API.

        Raises
        ------
        json.decoder.JSONDecodeError
            If the response from the OpenAI API cannot be parsed as JSON.
        """
        prompt = directives.improve_matched_relations.format(
            extracted_relations=json.dumps(extracted_relations, indent=2),
            graph_relations=json.dumps(graph_relations, indent=2),
            user_query=user_query,
            matched_relations=json.dumps(matched_relations, indent=2),
        )

        if self.llm_client.metadata.language_model_name:
            completions_config.model = (
                self.llm_client.metadata.language_model_name
            )
        response = await self.llm_client.client.chat.completions.create(
            messages=[{"role": "system", "content": prompt}],
            **completions_config.model_dump(),
        )

        formatted_response = response.choices[0].message.content.strip()

        formatted_response = formatted_response.replace(
            "```json\n", ""
        ).replace("\n```", "")

        try:
            improved_relations = json.loads(formatted_response)
        except json.decoder.JSONDecodeError:
            print("Failed to parse formatted response as JSON")
            improved_relations = []

        return improved_relations

    @staticmethod
    async def generate_schema(
        llm_client: LLMClient, questions: List[str]
    ) -> Tuple[GeneratedSchema, list[ErrorDetails]]:
        """Generate a schema.

        Generates a basic schema from a list of questions using the LLM client.
        This implementation does not consider the `fields` of entities.
        """
        try:
            service_start_time = time.time()
            logfire.instrument_openai(llm_client.client)
            errors: list[ErrorDetails] = []

            async def execute_prompt(
                prompt: str,
            ) -> dict[str, list[Any]] | None:
                """Execute a prompt and return the response."""
                response = await llm_client.client.chat.completions.create(
                    messages=[
                        {
                            "role": "system",
                            "content": prompt,
                        }
                    ],
                    model=(
                        llm_client.metadata.language_model_name
                        if llm_client.metadata.language_model_name
                        else "gpt-4o-mini"
                    ),
                    temperature=0.1,
                    max_tokens=4000,
                )
                if response.choices[0].message.content is not None:
                    response_text = response.choices[0].message.content.strip()
                else:
                    response_text = ""
                try:
                    # Remove markdown formatting if present
                    if "```json" in response_text:
                        start = (
                            response_text.find("```json") + len("```json") + 1
                        )
                        end = response_text.rfind("```")
                        response_text = response_text[start:end].strip()
                    return json.loads(response_text)
                except JSONDecodeError as je:
                    logger.error(
                        f"Failed to parse message content {je}: {response_text}"
                    )
                    return None

            async def generate_json_schema(
                idx: int, question: str
            ) -> dict[str, list[Any]] | None:
                """Generate a JSON schema for a given question."""
                logger.info(f'Extracting schema for question: "{question}"')

                per_question_schema_extraction_prompt = (
                    lambda focus_question, other_questions: f"""Given the primary question, create a knowledge graph schema that encapsulates the abstract and semantic essence of the query: {focus_question}\n
                Consider the primary question in relation to the following secondary questions: {','.join(other_questions)}.\n
                Entities in the schema should be abstract concepts (e.g., character, clothing, company, animal, drug, etc.), and relations should be verbs or verbal phrases (e.g., friends with, wears, runs, etc.). Each entity and relation must be described abstractly (e.g., "an animate being, typically a person in a story") and should have illustrative examples (e.g., "an animate being, typically a person in a story, e.g., Harry Potter, Ron Weasley,").
                Output only the entities and relations where the names use spaces between words. Exclude any entities or relations that contain underscores (_) or use camelCase (e.g., include "operates in" but exclude "operates_in" and "OperatesIn").
                All entities, relations, and patterns should be lowercase.

                The schema should be formatted as a JSON object that adheres to the following `Schema` Pydantic model. Ensure the output is in plain JSON and parsable via `json.loads()`.

                {inspect.getsource(SchemaEntity)}
                {inspect.getsource(SchemaRelation)}
                {inspect.getsource(TriplePattern)}
                {inspect.getsource(GeneratedSchema)}
                """
                )

                generation_prompt = per_question_schema_extraction_prompt(
                    focus_question=question,
                    other_questions=questions[:idx] + questions[idx + 1 :],
                )
                return await execute_prompt(prompt=generation_prompt)

            def merge_and_validate_schemas(
                json_schemas: list[dict[str, list[Any]]],
            ) -> GeneratedSchema:
                """Merge and validate the generated schemas."""
                json_schemas = [
                    schema for schema in json_schemas if schema is not None
                ]

                schema: dict[str, list[Any]] = {
                    "entities": [],
                    "relations": [],
                    "patterns": [],
                }

                # Track unique entity, relation, and pattern names
                seen_entity_names = set()
                seen_relation_names = set()
                seen_pattern_keys = set()

                # Merge and remove duplicates
                for json_schema in json_schemas:
                    # Add unique entities
                    for entity in json_schema["entities"]:
                        if entity["name"] not in seen_entity_names:
                            schema["entities"].append(entity)
                            seen_entity_names.add(entity["name"])

                    # Add unique relations
                    for relation in json_schema["relations"]:
                        if relation["name"] not in seen_relation_names:
                            schema["relations"].append(relation)
                            seen_relation_names.add(relation["name"])

                    # Build sets of valid entity and relation names
                    valid_entity_names = set(seen_entity_names)
                    valid_relation_names = set(seen_relation_names)

                    # Add unique patterns
                    for pattern in json_schema["patterns"]:
                        pattern_key = (
                            pattern["head"],
                            pattern["relation"],
                            pattern["tail"],
                        )

                        # Check if head, tail, relation exists
                        if (
                            pattern["head"] in valid_entity_names
                            and pattern["tail"] in valid_entity_names
                            and pattern["relation"] in valid_relation_names
                        ):

                            # Add pattern if it is valid and unique
                            if pattern_key not in seen_pattern_keys:
                                schema["patterns"].append(pattern)
                                seen_pattern_keys.add(pattern_key)
                        else:
                            print(f"Invalid pattern {pattern}. Not found.")

                try:
                    return GeneratedSchema(**schema)
                except Exception as e:
                    logger.info(
                        f"Unable to parse generated schema - {e}: {schema}"
                    )
                    raise ValueError("Unable to parse generated schema.")

            json_schemas = await asyncio.gather(
                *[
                    generate_json_schema(idx, question)
                    for idx, question in enumerate(questions)
                ]
            )
            if json_schemas is None:
                raise ValueError("Failed to generate schemas.")
            generated_schema = merge_and_validate_schemas(
                json_schemas=[
                    schema for schema in json_schemas if schema is not None
                ]
            )
            service_end_time = time.time()  # Capture the end time
            service_duration = (
                service_end_time - service_start_time
            )  # Calculate the duration
            logger.info(f"Schema generated in {service_duration:.4f} seconds")
            print(f"Schema generated in {service_duration:.4f} seconds")
            return generated_schema, errors

        except Exception as e:
            logger.error(f"Failed to generate schema: {e}")
            errors.append(
                ErrorDetails(
                    message="Failed to generate schema. Please try again or reformat your questions.",
                    level="critical",
                )
            )
            return (
                GeneratedSchema(entities=[], relations=[], patterns=[]),
                errors,
            )
