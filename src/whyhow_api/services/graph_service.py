"""Graph service."""

import asyncio
import json
import logging
import time
import typing
from abc import ABC, abstractmethod
from collections import defaultdict
from json.decoder import JSONDecodeError
from typing import Any, DefaultDict, Dict, List, Mapping, Sequence, Set, Tuple

import openai
import tiktoken
from bson import ObjectId
from fastapi import BackgroundTasks
from langchain_core.prompts import ChatPromptTemplate
from motor.motor_asyncio import (
    AsyncIOMotorClient,
    AsyncIOMotorClientSession,
    AsyncIOMotorCollection,
    AsyncIOMotorDatabase,
)
from pymongo import UpdateOne

from whyhow_api.config import Settings
from whyhow_api.dependencies import LLMClient
from whyhow_api.exceptions import NotFoundException
from whyhow_api.models.common import (
    EntityField,
    SchemaEntity,
    SchemaRelation,
    SchemaTriplePattern,
    StructuredSchemaEntity,
    StructuredSchemaTriplePattern,
    TriplePattern,
)
from whyhow_api.schemas.base import ErrorDetails, get_utc_now
from whyhow_api.schemas.chunks import ChunkDocumentModel
from whyhow_api.schemas.graphs import (
    ChunkFilters,
    CreateGraphBody,
    GraphDocumentModel,
    GraphStateErrorsUpdate,
    QueryGraphRequest,
    Triple,
)
from whyhow_api.schemas.nodes import (
    NodeDocumentModel,
    NodeWithId,
    NodeWithIdAndSimilarity,
)
from whyhow_api.schemas.queries import QueryDocumentModel, QueryParameters
from whyhow_api.schemas.rules import RuleOut
from whyhow_api.schemas.schemas import SchemaCreate, SchemaDocumentModel
from whyhow_api.schemas.tasks import TaskDocumentModel
from whyhow_api.schemas.triples import TripleDocumentModel, TripleWithId
from whyhow_api.services.crud.base import create_one, get_one, update_one
from whyhow_api.services.crud.chunks import get_chunks
from whyhow_api.services.crud.graph import list_triples, list_triples_by_ids
from whyhow_api.services.crud.rule import apply_rules_to_triples
from whyhow_api.services.crud.task import create_task
from whyhow_api.services.crud.triple import (
    convert_triple_to_text,
    update_triple_embeddings,
)
from whyhow_api.utilities.builders import OpenAIBuilder, SpacyEntityExtractor
from whyhow_api.utilities.common import (
    check_existing,
    clean_text,
    dict_to_tuple,
    tuple_to_dict,
)
from whyhow_api.utilities.config import (
    create_schema_guided_graph_prompt,
    openai_completions_configs,
)
from whyhow_api.utilities.cypher_export import generate_cypher_statements

logger = logging.getLogger(__name__)

AUTOGEN_DESCRIPTION = "auto-generated"


class RateLimiter:
    """Rate limiter for API requests."""

    def __init__(self, rpm_limit: int, tpm_limit: int):
        self.rpm_limit = rpm_limit
        self.tpm_limit = tpm_limit
        self.requests = 0
        self.tokens = 0
        self.last_reset = time.time()
        self.last_log_time = time.time()  # Track when the last log was printed
        logger.info(
            f"RateLimiter initialised with rpm_limit={rpm_limit} and tpm_limit={tpm_limit}"
        )

    async def wait(self, tokens: int) -> None:
        """Wait for the rate limiter to allow the request."""
        current_time = time.time()
        time_passed = current_time - self.last_reset
        if time_passed >= 60:
            logger.info("Resetting rate limiter counters")
            self.requests = 0
            self.tokens = 0
            self.last_reset = current_time
            time_passed = 0

        while (
            self.requests >= self.rpm_limit
            or self.tokens + tokens > self.tpm_limit
        ):
            current_time = time.time()
            if current_time - self.last_log_time >= 1.0:
                percent_requests_consumed = (
                    self.requests / self.rpm_limit
                ) * 100
                percent_tokens_consumed = (
                    (self.tokens + tokens) / self.tpm_limit
                ) * 100
                percent_time_remaining = (1 - time_passed / 60) * 100
                logger.info(
                    f"Rate limit reached: requests={self.requests} ({percent_requests_consumed:.2f}% consumed), "
                    f"tokens={self.tokens + tokens} ({percent_tokens_consumed:.2f}% consumed), "
                    f"time until reset={60 - time_passed:.2f}s ({percent_time_remaining:.2f}% remaining)"
                )
                self.last_log_time = current_time  # Update the last log time

            await asyncio.sleep(0.1)
            current_time = time.time()
            time_passed = current_time - self.last_reset
            if time_passed >= 60:
                logger.info(
                    "Resetting rate limiter counters due to time interval"
                )
                self.requests = 0
                self.tokens = 0
                self.last_reset = current_time
                time_passed = (
                    0  # Reset the timer since we are resetting the counters
                )

        self.requests += 1
        self.tokens += tokens
        percent_requests_consumed = (self.requests / self.rpm_limit) * 100
        percent_tokens_consumed = (self.tokens / self.tpm_limit) * 100
        logger.info(
            f"Request allowed: total requests={self.requests} ({percent_requests_consumed:.2f}% consumed), "
            f"total tokens={self.tokens} ({percent_tokens_consumed:.2f}% consumed)"
        )


DEFAULT_SEED_ENTITY_EXTRACTOR = SpacyEntityExtractor

template = """
    Answer the question based only on the following context:
    {context}

    If there is no context, say "No context provided.  Please add more specific information to the graph, or ask a more specific question based on the entities and relations."

    Question: {question}
    Use natural language and be concise.
    Answer:
"""  # noqa: E501
FINAL_CHAT_PROMPT = ChatPromptTemplate.from_template(template)


async def fork_schema(
    db: AsyncIOMotorDatabase,
    schema_id: ObjectId,
    graph_name: str,
    user_id: ObjectId,
) -> ObjectId:
    """Fork a schema.

    Parameters
    ----------
    db : AsyncIOMotorDatabase
        The database connection.
    schema_id : ObjectId
        The ID of the schema to fork.
    graph_name : str
        The name of the graph to use in forked schema name.
    user_id : ObjectId
        The ID of the user forking the schema.

    Returns
    -------
    - ObjectId: The ID of the forked schema.

    Raises
    ------
    - NotFoundException: If the schema is not found.
    """
    template_schema = await db.schema.find_one(
        {"_id": schema_id, "created_by": user_id}
    )
    if template_schema is None:
        raise NotFoundException("Schema not found.")

    template_schema["name"] = (
        f"{template_schema['name']} (graph: {graph_name})"
    )
    dt_now = get_utc_now()
    template_schema["created_at"] = dt_now
    template_schema["updated_at"] = dt_now
    template_schema["created_by"] = user_id
    del template_schema["_id"]  # Remove the id to create a new schema
    forked_schema = await db.schema.insert_one(template_schema)

    logger.info(f"Schema forked: {forked_schema.inserted_id}")

    return ObjectId(forked_schema.inserted_id)


async def get_and_separate_chunks_on_data_type(
    collection: AsyncIOMotorCollection, chunk_ids: List[ObjectId]
) -> dict[str, list[ChunkDocumentModel]]:
    """Get and separate chunks based on their data type."""
    cursor = collection.find({"_id": {"$in": chunk_ids}}, {"embedding": 0})
    db_chunks = await cursor.to_list(None)
    logger.info(f"Found {len(db_chunks)} DB Chunks")

    # Split chunks based on `data_type`
    chunks: DefaultDict[str, list[ChunkDocumentModel]] = defaultdict(list)
    for c in db_chunks:
        chunk_model = ChunkDocumentModel(**c)
        chunks[chunk_model.data_type].append(chunk_model)

    return dict(chunks)


def node_keys(t: Triple) -> Tuple[Tuple[str, str], Tuple[str, str]]:
    """Create keys for head and tail nodes."""
    return (t.head, t.head_type), (t.tail, t.tail_type)


def triple_key(t: Triple) -> Tuple[str, str, str, str, str]:
    """Create a key for a triple."""
    return (t.head, t.head_type, t.relation, t.tail, t.tail_type)


def merge_dicts(d1: Dict[str, Any], d2: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two dictionaries, combining values of matching keys only if they are different.

    Examples
    --------
    >>> d1 = {"a": 1, "b": 2}
    >>> d2 = {"b": 3, "c": 4}
    >>> merge_dicts(d1, d2)
    {'a': 1, 'b': [2, 3], 'c': 4}

    >>> d1 = {"a": [1, 2], "b": [3]}
    >>> d2 = {"a": [4], "b": [5], "c": [6]}
    >>> merge_dicts(d1, d2)
    {'a': [1, 2, 4], 'b': [3, 5], 'c': [6]}

    >>> d1 = {"a": {"x": 1}, "b": {"y": 2}}
    >>> d2 = {"a": {"z": 3}, "b": {"y": 4}, "c": {"w": 5}}
    >>> merge_dicts(d1, d2)
    {'a': {'x': 1, 'z': 3}, 'b': {'y': [2, 4]}, 'c': {'w': 5}}

    >>> d1 = {"a": [1, 2], "b": {"x": 3}}
    >>> d2 = {"a": {"y": 4}, "b": [5]}
    >>> merge_dicts(d1, d2)
    {'a': [1, 2, {'y': 4}], 'b': [{'x': 3}, 5]}

    >>> d1 = {"a": [1, 2], "b": 3}
    >>> d2 = {"a": 4, "b": [5, 6]}
    >>> merge_dicts(d1, d2)
    {'a': [1, 2, 4], 'b': [3, 5, 6]}

    >>> d1 = {"a": 1, "b": 2}
    >>> d2 = {"a": 1, "b": 3}
    >>> merge_dicts(d1, d2)
    {'a': 1, 'b': [2, 3]}
    """
    for key, value in d2.items():
        if key in d1:
            if d1[key] != value:  # Check if the values are different
                if isinstance(d1[key], list) and isinstance(value, list):
                    d1[key].extend(value)
                elif isinstance(d1[key], dict) and isinstance(value, dict):
                    d1[key] = merge_dicts(d1[key], value)
                elif isinstance(d1[key], list):
                    d1[key] = d1[key] + [value]
                elif isinstance(value, list):
                    d1[key] = [d1[key]] + value
                else:
                    d1[key] = [d1[key], value]
        else:
            d1[key] = value
    return d1


async def create_node_id_map(
    db: AsyncIOMotorDatabase,
    node_names: set[str],
    node_types: set[str],
    graph_id: ObjectId,
    user_id: ObjectId,
    session: AsyncIOMotorClientSession | None = None,
) -> dict[tuple[str, str], ObjectId]:
    """Create a map of node names to node IDs.

    Notes
    -----
    - Retrieve all node IDs with a query (assuming name and type combination is unique)
    """
    query = {
        "graph": graph_id,
        "created_by": user_id,
        "name": {"$in": list(node_names)},
        "type": {"$in": list(node_types)},
    }
    cursor = db.node.find(query, session=session)

    node_id_map = {}
    async for node in cursor:
        node_id_map[(node["name"], node["type"])] = node["_id"]
    return node_id_map


def extract_properties_from_fields(fields: list[EntityField]) -> str:
    """Extract properties from a list of fields."""
    if len(fields) == 0:
        return ""
    # Create a list of property strings for each field
    property_lists = []

    for field in fields:
        if len(field.properties) == 0:
            continue  # Skip fields with no properties

        # Create the property string for the field
        if len(field.properties) == 1:
            property_string = field.properties[0]
        else:
            property_string = (
                ", ".join(field.properties[:-1])
                + ", and "
                + field.properties[-1]
            )

        property_lists.append(property_string)

    # Combine all property lists
    return ", ".join(property_lists)


def convert_pattern_to_text(pattern: SchemaTriplePattern) -> str:
    """Convert a pattern to a natural language text string."""
    sentence = f"The {clean_text(pattern.head.name.lower())} ({clean_text(pattern.head.description.lower())})"

    if pattern.head.fields:
        sentence += f" with properties {extract_properties_from_fields(pattern.head.fields)},"

    sentence += f" {clean_text(pattern.relation.name.lower())}"
    sentence += f" the {clean_text(pattern.tail.name.lower())} ({clean_text(pattern.tail.description.lower())})"

    if pattern.tail.fields:
        sentence += f" with properties {extract_properties_from_fields(pattern.tail.fields)}"
    return sentence


@typing.no_type_check
def merge_dicts_query(field_name: str, field_value: Any) -> dict[str, Any]:
    """Merge dictionaries query."""
    return {
        "$arrayToObject": {
            "$map": {
                "input": {
                    "$concatArrays": [
                        {
                            "$objectToArray": {
                                "$ifNull": [f"${field_name}", {}]
                            }
                        },
                        {
                            "$map": {
                                "input": {"$objectToArray": field_value},
                                "as": "newField",
                                "in": {
                                    "k": "$$newField.k",
                                    "v": {
                                        "$cond": {
                                            "if": {
                                                "$in": [
                                                    "$$newField.k",
                                                    {
                                                        "$map": {
                                                            "input": {
                                                                "$objectToArray": {
                                                                    "$ifNull": [
                                                                        f"${field_name}",
                                                                        {},
                                                                    ]
                                                                }
                                                            },
                                                            "as": "existingField",
                                                            "in": "$$existingField.k",
                                                        }
                                                    },
                                                ]
                                            },
                                            "then": {
                                                "$let": {
                                                    "vars": {
                                                        "existingValue": {
                                                            "$arrayElemAt": [
                                                                {
                                                                    "$filter": {
                                                                        "input": {
                                                                            "$objectToArray": {
                                                                                "$ifNull": [
                                                                                    f"${field_name}",
                                                                                    {},
                                                                                ]
                                                                            }
                                                                        },
                                                                        "cond": {
                                                                            "$eq": [
                                                                                "$$this.k",
                                                                                "$$newField.k",
                                                                            ]
                                                                        },
                                                                    }
                                                                },
                                                                0,
                                                            ]
                                                        }
                                                    },
                                                    "in": {
                                                        "$cond": {
                                                            "if": {
                                                                "$eq": [
                                                                    {
                                                                        "$type": "$$existingValue.v"
                                                                    },
                                                                    "array",
                                                                ]
                                                            },
                                                            "then": {
                                                                "$concatArrays": [
                                                                    "$$existingValue.v",
                                                                    [
                                                                        "$$newField.v"
                                                                    ],
                                                                ]
                                                            },
                                                            "else": {
                                                                "$cond": {
                                                                    "if": {
                                                                        "$eq": [
                                                                            "$$existingValue.v",
                                                                            "$$newField.v",
                                                                        ]
                                                                    },
                                                                    "then": "$$existingValue.v",
                                                                    "else": [
                                                                        "$$existingValue.v",
                                                                        "$$newField.v",
                                                                    ],
                                                                }
                                                            },
                                                        }
                                                    },
                                                }
                                            },
                                            "else": "$$newField.v",
                                        }
                                    },
                                },
                            }
                        },
                    ]
                },
                "as": "mergedField",
                "in": "$$mergedField",
            }
        }
    }


def merge_lists_query(field_name: str, field_value: Any) -> dict[str, Any]:
    """Merge lists query."""
    return {
        "$setUnion": [
            {"$ifNull": [f"${field_name}", []]},
            field_value,
        ]
    }


async def build_graph(
    db: AsyncIOMotorDatabase,
    db_client: AsyncIOMotorClient,
    llm_client: LLMClient,
    graph_id: ObjectId,
    user_id: ObjectId,
    triples: list[Triple],
    task_id: ObjectId | None = None,
) -> None:
    """Build a graph from triples."""
    try:
        logger.info(f"Populating graph with ID: {graph_id}")

        # Split triples into batches of 1000
        batch_size = 1000
        triple_chunks = [
            triples[i : i + batch_size]
            for i in range(0, len(triples), batch_size)
        ]

        # Process each chunk one by one
        for batch_index, chunk in enumerate(triple_chunks):
            logger.info(
                f"Processing batch {batch_index + 1}/{len(triple_chunks)}"
            )

            async with await db_client.start_session() as session:
                async with session.start_transaction():
                    # -- Create nodes
                    node_operations = []
                    node_names = set()
                    node_types = set()

                    for triple in chunk:  # Process each triple in the chunk
                        for node, properties in [
                            (
                                NodeDocumentModel(
                                    name=triple.head,
                                    type=triple.head_type,
                                    created_by=user_id,
                                    graph=graph_id,
                                ),
                                triple.head_properties,
                            ),
                            (
                                NodeDocumentModel(
                                    name=triple.tail,
                                    type=triple.tail_type,
                                    created_by=user_id,
                                    graph=graph_id,
                                ),
                                triple.tail_properties,
                            ),
                        ]:
                            chunks = properties.pop("chunks", [])
                            validated_chunks = await check_existing(
                                db, "chunk", chunks, {"created_by": user_id}
                            )
                            node.properties = properties
                            node.chunks = validated_chunks
                            node_operations.append(
                                UpdateOne(
                                    {
                                        "name": node.name,
                                        "type": node.type,
                                        "graph": graph_id,
                                        "created_by": user_id,
                                    },
                                    [
                                        {
                                            "$set": {
                                                "properties": merge_dicts_query(
                                                    "properties",
                                                    node.properties,
                                                ),
                                                "chunks": merge_lists_query(
                                                    "chunks", node.chunks
                                                ),
                                                "created_at": {
                                                    "$ifNull": [
                                                        "$created_at",
                                                        node.created_at,
                                                    ]
                                                },
                                                "updated_at": node.updated_at,
                                            },
                                        },
                                    ],
                                    upsert=True,
                                )
                            )
                            node_names.add(node.name)
                            node_types.add(node.type)

                    # Execute bulk insert for nodes
                    if node_operations:
                        await db.node.bulk_write(
                            node_operations, session=session
                        )
                    logger.info("Nodes created")

                    node_id_map = await create_node_id_map(
                        db=db,
                        node_names=node_names,
                        node_types=node_types,
                        graph_id=graph_id,
                        user_id=user_id,
                        session=session,
                    )

                    # Fetch all node chunks
                    all_nodes = await db.node.find(
                        {
                            "_id": {"$in": list(node_id_map.values())},
                            "graph": graph_id,
                            "created_by": user_id,
                        },
                        {"_id": 1, "chunks": 1},
                        session=session,
                    ).to_list(None)

                    # Create a dictionary for quick lookup
                    node_chunks = {
                        node["_id"]: node.get("chunks", [])
                        for node in all_nodes
                    }

                    # Prepare triple documents using node IDs from the map
                    triple_operations = []
                    triple_filters = []
                    for triple in chunk:
                        properties = triple.relation_properties
                        chunks = properties.pop("chunks", [])
                        validated_chunks = await check_existing(
                            db, "chunk", chunks, {"created_by": user_id}
                        )
                        triple_model = TripleDocumentModel(
                            head_node=node_id_map[
                                (triple.head, triple.head_type)
                            ],
                            tail_node=node_id_map[
                                (triple.tail, triple.tail_type)
                            ],
                            type=triple.relation,
                            properties=properties,
                            chunks=validated_chunks,
                            created_by=user_id,
                            graph=graph_id,
                        )
                        triple_filters.append(
                            {
                                "head_node": triple_model.head_node,
                                "tail_node": triple_model.tail_node,
                                "type": triple_model.type,
                                "graph": triple_model.graph,
                                "created_by": triple_model.created_by,
                            }
                        )

                        # Use the pre-fetched node chunks
                        head_chunks = node_chunks.get(triple_model.head_node)
                        if head_chunks is None:
                            raise NotFoundException(
                                f"Failed to find head node: {triple_model.head_node}"
                            )

                        tail_chunks = node_chunks.get(triple_model.tail_node)
                        if tail_chunks is None:
                            raise NotFoundException(
                                f"Failed to find tail node: {triple_model.tail_node}"
                            )

                        # Compute intersection of chunks
                        intersected_chunks = list(
                            set(head_chunks) & set(tail_chunks)
                        )

                        triple_operations.append(
                            UpdateOne(
                                triple_filters[-1],
                                [
                                    {
                                        "$set": {
                                            "properties": merge_dicts_query(
                                                "properties",
                                                triple_model.properties,
                                            ),
                                            "chunks": {
                                                "$setUnion": [
                                                    intersected_chunks,
                                                    merge_lists_query(
                                                        "chunks",
                                                        triple_model.chunks,
                                                    ),
                                                ]
                                            },
                                            "created_at": {
                                                "$ifNull": [
                                                    "$created_at",
                                                    triple_model.created_at,
                                                ]
                                            },
                                            "updated_at": triple_model.updated_at,
                                        },
                                    },
                                ],
                                upsert=True,
                            )
                        )

                    # Execute bulk insert for triples
                    if triple_operations:
                        await db.triple.bulk_write(
                            triple_operations, session=session
                        )
                    logger.info(f"Triples created for batch {batch_index + 1}")

                    # Embed triples
                    if triple_filters:
                        updated_triples = await db.triple.find(
                            {"$or": triple_filters},
                            {"_id": 1},
                            session=session,
                        ).to_list(None)
                        await update_triple_embeddings(
                            db=db,
                            llm_client=llm_client,
                            triple_ids=[
                                ObjectId(t["_id"]) for t in updated_triples
                            ],
                            user_id=user_id,
                            session=session,
                        )
                    logger.info(
                        f"Triple embeddings updated for batch {batch_index + 1}"
                    )

                    # If task_id is provided, update task status
                    if task_id:
                        await db.task.update_one(
                            {"_id": task_id},
                            {
                                "$set": {
                                    "result": f"Processing batch {batch_index + 1}/{len(triple_chunks)}",
                                }
                            },
                            session=session,
                        )

                    # Commit the transaction
                    await session.commit_transaction()

                logger.info(f"Chunk {batch_index + 1} processed successfully")

        # If task_id is provided, update task status
        if task_id:
            await db.task.update_one(
                {"_id": task_id},
                {
                    "$set": {
                        "end_time": get_utc_now(),
                        "status": "success",
                        "result": "Graph constructed",
                    }
                },
            )

        # Update graph status to 'ready' after all chunks are processed
        await update_one(
            collection=db["graph"],
            document_model=GraphDocumentModel,
            id=ObjectId(graph_id),
            document=GraphStateErrorsUpdate(status="ready"),
            user_id=user_id,
        )

        logger.info("Graph constructed'")
    except NotFoundException as e:
        logger.error(f"Failed to build/update graph: {e}", exc_info=True)
        if task_id:
            await db.task.update_one(
                {"_id": task_id},
                {
                    "$set": {
                        "end_time": get_utc_now(),
                        "status": "failed",
                        "result": str(e),
                    }
                },
                session=session,
            )
        await update_one(
            collection=db["graph"],
            document_model=GraphDocumentModel,
            id=ObjectId(graph_id),
            document=GraphStateErrorsUpdate(
                status="failed",
                errors=[
                    ErrorDetails(
                        message=str(e),
                        level="critical",
                    )
                ],
            ),
            user_id=user_id,
        )
        raise
    except Exception as e:
        logger.error(f"Failed to build/update graph: {e}", exc_info=True)
        if task_id:
            await db.task.update_one(
                {"_id": task_id},
                {
                    "$set": {
                        "end_time": get_utc_now(),
                        "status": "failed",
                        "result": "Failed to build/update graph",
                    }
                },
                session=session,
            )
        await update_one(
            collection=db["graph"],
            document_model=GraphDocumentModel,
            id=ObjectId(graph_id),
            document=GraphStateErrorsUpdate(
                status="failed",
                errors=[
                    ErrorDetails(
                        message="Failed to build/update graph",
                        level="critical",
                    )
                ],
            ),
            user_id=user_id,
        )
        raise


async def extract_graph_triples(
    llm_client: LLMClient,
    chunks: list[ChunkDocumentModel],
    tokenizer: tiktoken.core.Encoding,
    patterns: list[SchemaTriplePattern],
    rate_limiter: RateLimiter,
) -> list[Triple]:
    """Extract triples from chunks."""
    try:

        all_triples: List[Triple] = []

        async def process_chunk(
            chunk: ChunkDocumentModel,
        ) -> List[Triple] | None:
            """Process a chunk."""
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    # Ensure chunk.content is a string
                    if isinstance(chunk.content, dict):
                        content_str = json.dumps(chunk.content)
                    else:
                        content_str = chunk.content

                    # Estimate tokens using tiktoken
                    # Use the first pattern as an estimate
                    estimated_tokens = len(
                        tokenizer.encode(
                            create_schema_guided_graph_prompt(
                                text=content_str, pattern=patterns[0]
                            )
                        )
                    )
                    await rate_limiter.wait(estimated_tokens)

                    return await OpenAIBuilder.extract_triples(
                        llm_client=llm_client,
                        chunk=chunk,
                        completions_config=openai_completions_configs.triple,
                        patterns=patterns,
                    )
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(
                            f"Failed to process chunk after {max_retries} attempts: {e}"
                        )
                        raise
                    await asyncio.sleep(2**attempt)  # Exponential backoff

            return None

        tasks = [process_chunk(chunk) for chunk in chunks]

        chunk_results = await asyncio.gather(*tasks)

        for triples in chunk_results:
            if triples is not None:
                all_triples.extend(triples)

        semantic_triples = all_triples
        logger.info(f"Extracted {len(semantic_triples)} semantic triples")

        return all_triples

    except Exception as e:
        logger.error(f"Failed to extract triples: {e}")
        raise


def create_structured_patterns(
    patterns: list[SchemaTriplePattern],
) -> list[StructuredSchemaTriplePattern]:
    """Create structured patterns from schema triple patterns.

    This process converts entity `fields` into pairwise patterns for structured extraction
    """
    structured_patterns: list[StructuredSchemaTriplePattern] = []
    for p in patterns:
        head = p.head
        tail = p.tail
        relation = p.relation.name

        head_fields = head.fields
        tail_fields = tail.fields

        for hf in head_fields:
            for tf in tail_fields:
                structured_patterns.append(
                    StructuredSchemaTriplePattern(
                        head=StructuredSchemaEntity(name=head.name, field=hf),
                        relation=relation,
                        tail=StructuredSchemaEntity(name=tail.name, field=tf),
                    )
                )

    logger.info(f"Created {len(structured_patterns)} structured Patterns")
    return structured_patterns


def extract_structured_graph_triples(
    patterns: list[SchemaTriplePattern], chunks: list[ChunkDocumentModel]
) -> list[Triple]:
    """Extract structured triples from chunks.

    Creates patterns from combinations of provided fields which are used to
    extract structured triples from the provided chunks. Any patterns that have
    objects (dicts) as their head or tail are skipped. Only chunks that are of `data_type`
    'object' are considered for structured triples.

    Notes
    -----
    - This will inevitably create patterns that may have no matches, but that's okay.
    """
    # Convert entity `fields` into pairwise patterns for structured extraction
    structured_patterns = create_structured_patterns(patterns)

    structured_triples: list[Triple] = []
    for c in chunks:
        if c.data_type != "object":
            logger.info("Skipping chunk - data_type is not 'object'")
            continue
        for p in structured_patterns:
            # Head/tail are stringified; if no value is provided, it defaults to "Unnamed"
            has_head_field = p.head.field.name in c.content
            if not has_head_field:
                logger.info(
                    f"Skipping structured triple - head field not found in chunk content - {p.head.field.name}"
                )
                continue

            head_value = c.content.get(p.head.field.name)  # type: ignore
            head = "Unnamed" if (head_value is None) else str(head_value)
            head_type = p.head.name
            relation: str = p.relation
            has_tail_field = p.tail.field.name in c.content
            if not has_tail_field:
                logger.info(
                    f"Skipping structured triple - tail field not found in chunk content - {p.tail.field.name}"
                )
                continue
            tail_value = c.content.get(p.tail.field.name)  # type: ignore
            tail = "Unnamed" if (tail_value is None) else str(tail_value)
            tail_type = p.tail.name

            print(f"head: {head}, tail: {tail}")
            head_properties: dict[str, Any] = (
                {}
                if not p.head.field.properties
                else {
                    prop: c.content.get(prop)  # type: ignore
                    for prop in p.head.field.properties
                }
            )
            head_properties["chunks"] = [c.id]
            tail_properties: dict[str, Any] = (
                {}
                if p.tail.field.properties is None
                else {
                    prop: c.content.get(prop)  # type: ignore
                    for prop in p.tail.field.properties
                }
            )
            tail_properties["chunks"] = [c.id]
            structured_triples.append(
                Triple(
                    head=head,
                    head_type=head_type,
                    head_properties=head_properties,
                    relation=relation,
                    relation_properties={"chunks": [c.id]},
                    tail=tail,
                    tail_type=tail_type,
                    tail_properties=tail_properties,
                )
            )

    return structured_triples


async def apply_rules(
    db: AsyncIOMotorDatabase,
    extracted_triples: list[Triple],
    workspace_id: ObjectId,
    graph_id: ObjectId,
    user_id: ObjectId,
    errors: list[ErrorDetails],
) -> list[Triple]:
    """
    Apply workspace rules to the extracted triples.

    Parameters
    ----------
    db : AsyncIOMotorDatabase
        The database connection.
    extracted_triples : list[Triple]
        The extracted triples.
    workspace_id : ObjectId
        The ID of the workspace.
    graph_id : ObjectId
        The ID of the graph.
    user_id : ObjectId
        The ID of the user.
    errors : list[ErrorDetails]
        The list of errors.

    Returns
    -------
    - list[Triple]: The updated triples.
    """
    # Get workspace rules
    rules = await db.rule.find(
        {"workspace": workspace_id, "created_by": user_id},
    ).to_list(None)

    workspace_rules = [RuleOut(**rule) for rule in rules]

    # Apply workspace rules to the triples
    updated_triples = apply_rules_to_triples(
        extracted_triples, workspace_rules
    )

    # Check that graph `rules` field is not existing
    graph = await db.graph.find_one(
        {
            "_id": graph_id,
            "workspace": workspace_id,
            "rules": {"$exists": True},
        }
    )
    if graph is None:
        # Append workspace rules to the graph `rules` field
        result = await db.graph.update_one(
            {"_id": graph_id}, {"$push": {"rules": {"$each": rules}}}
        )

        if result.modified_count == 0:
            errors.append(
                ErrorDetails(
                    message="Failed to update graph with workspace rules",
                    level="error",
                )
            )
            await update_one(
                collection=db["graph"],
                document_model=GraphDocumentModel,
                id=ObjectId(graph_id),
                document=GraphStateErrorsUpdate(
                    status="failed", errors=errors
                ),
                user_id=user_id,
            )
            raise

    return updated_triples


async def chunk_filters_to_triples(
    db: AsyncIOMotorDatabase,
    filters: dict[str, Any],
    llm_client: LLMClient,
    user_id: ObjectId,
    workspace_id: ObjectId,
    max_chunks: int,
    settings: Settings,
    tiktoken_encoder: tiktoken.core.Encoding,
    rate_limiter: RateLimiter,
    patterns: list[SchemaTriplePattern],
) -> list[Triple]:
    """Convert chunk filters to triples."""
    logger.info(f"All chunk filters: {filters}")
    _chunks = await db.chunk.find(filters, {"_id": 1}).to_list(None)
    chunk_ids = [c["_id"] for c in _chunks]
    logger.info(f"Found {len(chunk_ids)} possible chunks")

    extracted_triples: list[Triple] = []

    async def process_pattern(
        pattern: SchemaTriplePattern,
    ) -> List[Triple]:
        full_pattern = convert_pattern_to_text(pattern)
        logger.info(f"Processing pattern: {full_pattern}")

        # ONLY STRINGS ARE RETRIEVED BY VECTOR SEARCH AS THIS HAS OVERHEAD
        # OBJECTS ARE NOT AS THERE IS NO LLM OVERHEAD
        string_chunk_models = await get_chunks(
            collection=db["chunk"],
            llm_client=llm_client,
            user_id=user_id,
            include_embeddings=False,
            filters={
                "workspaces": {"$in": [workspace_id]},
                "seed_concept": full_pattern,
                "data_type": "string",
                "_id": {"$in": chunk_ids},
            },
            limit=max_chunks,
            populate=False,
        )
        if len(string_chunk_models) == 0 or string_chunk_models is None:
            logger.info(f"No string chunks found for pattern: {full_pattern}")
            string_chunk_models = []
        logger.info(
            f"Found {len(string_chunk_models)} string chunks for pattern: {full_pattern}"
        )

        object_chunk_models = await get_chunks(
            collection=db["chunk"],
            llm_client=llm_client,
            user_id=user_id,
            include_embeddings=False,
            filters={
                "data_type": "object",
                "_id": {"$in": chunk_ids},
            },
            limit=(
                max_chunks
                if settings.api.restrict_structured_chunk_retrieval
                else -1
            ),
            populate=False,
        )
        if len(object_chunk_models) == 0 or object_chunk_models is None:
            logger.info(f"No object chunks found for pattern: {full_pattern}")
            object_chunk_models = []
        logger.info(
            f"Found {len(object_chunk_models)} object chunks for pattern: {full_pattern}"
        )

        chunk_models = string_chunk_models + object_chunk_models

        logger.info(f"chunk_models: {chunk_models[:2]}")

        if chunk_models is None:
            logger.warning(f"No chunks found for pattern: {pattern}")
            return []

        logger.info(
            f"Retrieved {len(chunk_models)} chunks for pattern: {pattern}"
        )

        string_chunks = [
            chunk for chunk in chunk_models if chunk.data_type == "string"
        ]
        object_chunks = [
            chunk for chunk in chunk_models if chunk.data_type == "object"
        ]

        logger.info(
            f"String chunks: {len(string_chunks)}, Object chunks: {len(object_chunks)}"
        )

        unstructured_triples = []
        if string_chunks:
            unstructured_triples = await extract_graph_triples(
                llm_client=llm_client,
                patterns=[pattern],
                chunks=string_chunks,
                tokenizer=tiktoken_encoder,
                rate_limiter=rate_limiter,
            )
            logger.info(
                f"Extracted {len(unstructured_triples)} unstructured triples for pattern: {pattern}"
            )

        structured_triples = []
        if object_chunks:
            structured_triples = extract_structured_graph_triples(
                patterns=[pattern],
                chunks=object_chunks,
            )
            logger.info(
                f"Extracted {len(structured_triples)} structured triples for pattern: {pattern}"
            )

        return [*structured_triples, *unstructured_triples]

    # Process patterns concurrently using asyncio.gather
    pattern_results = await asyncio.gather(
        *[process_pattern(pattern) for pattern in patterns]
    )

    # Flatten the results and extend extracted_triples
    for triples in pattern_results:
        extracted_triples.extend(triples)

    logger.info(f"Extracted a total of {len(extracted_triples)} triples")

    return extracted_triples


async def create_or_update_graph(
    db: AsyncIOMotorDatabase,
    db_client: AsyncIOMotorClient,
    llm_client: LLMClient,
    user_id: ObjectId,
    graph_id: ObjectId,
    workspace_id: ObjectId,
    schema_id: ObjectId,
    settings: Settings,
    filters: ChunkFilters | None = None,
) -> None:
    """
    Create or update a graph.

    Parameters
    ----------
    db : AsyncIOMotorDatabase
        The database connection.
    llm_client : LLMClient
        The OpenAI or Azure OpenAI client for generating responses.
    user_id : ObjectId
        The ID of the user creating the graph.
    graph_id : ObjectId
        The unique identifier for the new graph.
    workspace_id : ObjectId
        The ID of the workspace where the graph will be located.
    schema_id : ObjectId
        The ID of the schema to be used for the graph.
    filters : ChunkFilters | None, optional
        Filters to apply for chunk retrieval when creating the graph.
    settings : Settings
        The settings for the API.

    Returns
    -------
    - None
    """
    logger.info(f"Creating graph with filters: {filters}")
    errors: list[ErrorDetails] = []

    schema = await db.schema.find_one(
        {
            "_id": schema_id,
            "workspace": workspace_id,
            "created_by": user_id,
        }
    )
    if schema is None:
        errors.append(
            ErrorDetails(
                message="Schema not found",
                level="error",
            )
        )
        raise ValueError("Schema not found")

    patterns = [SchemaTriplePattern(**p) for p in schema["patterns"]]
    logger.info(f"Number of patterns: {len(patterns)}")

    # Ensure we don't exceed settings.api.max_patterns
    if len(patterns) > settings.api.max_patterns:
        logger.warning(
            f"Number of patterns ({len(patterns)}) provided exceeds the limit ({settings.api.max_patterns}). Using first {settings.api.max_patterns} patterns."
        )
        patterns = patterns[: settings.api.max_patterns]

    MAX_CHUNKS = settings.api.max_chunk_pattern_product // len(patterns)
    logger.info(f"Maximum chunks per pattern: {MAX_CHUNKS}")

    tiktoken_encoder = tiktoken.encoding_for_model(
        settings.generative.openai.model
    )
    rate_limiter = RateLimiter(
        settings.generative.openai.rpm_limit,
        settings.generative.openai.tpm_limit,
    )
    try:

        # Find all the possible chunks based on the provided filters
        all_chunk_filters = {
            "created_by": user_id,
            "workspaces": workspace_id,
            **(filters.mql_filter if filters else {}),
        }
        extracted_triples = await chunk_filters_to_triples(
            db=db,
            filters=all_chunk_filters,
            llm_client=llm_client,
            user_id=user_id,
            workspace_id=workspace_id,
            max_chunks=MAX_CHUNKS,
            settings=settings,
            tiktoken_encoder=tiktoken_encoder,
            rate_limiter=rate_limiter,
            patterns=patterns,
        )

        updated_triples = await apply_rules(
            db=db,
            extracted_triples=extracted_triples,
            workspace_id=workspace_id,
            graph_id=graph_id,
            user_id=user_id,
            errors=errors,
        )

        # Create graph from triples in the database
        await build_graph(
            db=db,
            db_client=db_client,
            llm_client=llm_client,
            graph_id=graph_id,
            triples=updated_triples,
            user_id=user_id,
        )
        logger.info(
            f"Graph created/updated successfully with graph_id: {graph_id}"
        )

    except ValueError as e:
        logger.error(
            f"Failed to build/update graph: {e}. Updating graph status.",
            exc_info=True,
        )
        errors.append(
            ErrorDetails(
                message="Failed to build/update graph",
                level="critical",
            )
        )
        await update_one(
            collection=db["graph"],
            document_model=GraphDocumentModel,
            id=ObjectId(graph_id),
            document=GraphStateErrorsUpdate(status="failed", errors=errors),
            user_id=user_id,
        )
        raise
    except openai.RateLimitError as e:
        logger.error(
            f"Rate limit reached: {e}. Updating graph status.",
            exc_info=True,
        )
        errors.append(
            ErrorDetails(
                message="Rate limit reached",
                level="critical",
            )
        )
        await update_one(
            collection=db["graph"],
            document_model=GraphDocumentModel,
            id=ObjectId(graph_id),
            document=GraphStateErrorsUpdate(status="failed", errors=errors),
            user_id=user_id,
        )
        raise

    except Exception as e:
        logger.error(
            f"Failed to build/update graph: {e}. Updating graph status.",
            exc_info=True,
        )
        errors.append(
            ErrorDetails(
                message="Failed to build/update graph",
                level="critical",
            )
        )
        await update_one(
            collection=db["graph"],
            document_model=GraphDocumentModel,
            id=ObjectId(graph_id),
            document=GraphStateErrorsUpdate(status="failed", errors=errors),
            user_id=user_id,
        )
        raise


class QueryProcessor(ABC):
    """Query processor interface."""

    @abstractmethod
    async def query(
        self, request: QueryGraphRequest
    ) -> QueryDocumentModel | None:
        """Perform a query."""
        pass


class MixedQueryProcessor(QueryProcessor):
    """Mixed query service."""

    def __init__(
        self,
        db: AsyncIOMotorDatabase,
        graph_id: ObjectId,
        user_id: ObjectId,
        workspace_id: ObjectId,
        schema_id: ObjectId,
        llm_client: LLMClient,
        settings: Settings,
    ):
        self.db = db
        self.graph_id = graph_id
        self.user_id = user_id
        self.workspace_id = workspace_id
        self.schema_id = schema_id
        self.llm_client = llm_client
        self.settings = settings

    async def _retrieve_entities_and_relation_types(
        self,
        entities: list[str] | None = None,
        relations: list[str] | None = None,
    ) -> tuple[list[str], list[str]]:
        """Retrieve entities and relations types from the associated schema.

        This function uses the provided `schema_id` to find the associated schema and retrieve the entities and relations types.
        This is necessary for instances where the user does not provide these explicitly, for example via an interface.

        If the user has supplied either the entities or relations, they are not updated from the retrieved schema.

        Parameters
        ----------
        entities
            A list of entity types.
        relations
            A list of relation types.

        Returns
        -------
        tuple
            A tuple containing the entities and relations types.
        """
        # Get schema associated with graph
        schema = await self.db.schema.find_one(
            {
                "_id": self.schema_id,
                "created_by": self.user_id,
                "workspace": self.workspace_id,
            }
        )

        # If no schema is found and both entities and relations are not provided, raise an error
        if schema is None and entities is None and relations is None:
            raise ValueError(
                "Schema not found and no entities or relations provided."
            )

        if entities is None:
            entities = (
                [e["name"] for e in schema["entities"]]
                if schema and schema.get("entities") is not None
                else []
            )

        if relations is None:
            relations = (
                [r["name"] for r in schema["relations"]]
                if schema and schema.get("relations") is not None
                else []
            )

        return entities, relations

    async def _retrieve_filtered_triple_and_node_ids(
        self,
        entities: list[str],
        relations: list[str],
        values: list[str] | None = None,
    ) -> tuple[list[ObjectId], list[ObjectId]]:
        """Retrieve filtered triple and node IDs based on entities, relations, and values.

        This function is used to perform a preliminary filtering of triples and nodes based on the provided entities, relations, and values.

        Parameters
        ----------
        entities
            A list of entity types.
        relations
            A list of relation types.
        values
            A list of entity values e.g. the `name` of entities.

        Returns
        -------
        tuple
            A tuple containing the filtered node and triple IDs.
        """
        logger.info(
            f"Entities: {entities}, Relations: {relations}, Values: {values}"
        )

        node_query = {
            "graph": self.graph_id,
            "created_by": self.user_id,
            "$and": [
                {"type": {"$in": entities}},
                {"name": {"$in": values}} if values else {},
            ],
        }
        matched_nodes = await self.db.node.find(
            node_query,
            {"_id": 1},
        ).to_list(None)
        matched_node_ids = [node["_id"] for node in matched_nodes]

        triple_query = {
            "graph": self.graph_id,
            "created_by": self.user_id,
            "type": {"$in": relations},
            "$or": [
                {"head_node": {"$in": matched_node_ids}},
                {"tail_node": {"$in": matched_node_ids}},
            ],
        }
        matched_triples = await self.db.triple.find(triple_query).to_list(None)
        matched_triple_ids = [triple["_id"] for triple in matched_triples]

        return matched_node_ids, matched_triple_ids

    async def _retrieve_triples(
        self,
        triple_ids: list[ObjectId],
    ) -> tuple[list[NodeWithId], list[TripleWithId]]:
        """Retrieve triples and associated nodes based on triple IDs.

        Parameters
        ----------
        triple_ids
            A list of triple IDs.

        Returns
        -------
        tuple
            A tuple containing the nodes and triples.
        """
        pipeline: list[dict[str, Any]] = [
            {
                "$match": {
                    "graph": self.graph_id,
                    "created_by": self.user_id,
                    "_id": {"$in": triple_ids},
                }
            },
            {
                "$lookup": {
                    "from": "node",
                    "localField": "head_node",
                    "foreignField": "_id",
                    "as": "head_node",
                }
            },
            {
                "$lookup": {
                    "from": "node",
                    "localField": "tail_node",
                    "foreignField": "_id",
                    "as": "tail_node",
                }
            },
            {
                "$project": {
                    "head": {"$arrayElemAt": ["$head_node.name", 0]},
                    "head_type": {"$arrayElemAt": ["$head_node.type", 0]},
                    "head_id": {"$arrayElemAt": ["$head_node._id", 0]},
                    "head_properties": {
                        "$arrayElemAt": ["$head_node.properties", 0]
                    },
                    "head_chunks": {"$arrayElemAt": ["$head_node.chunks", 0]},
                    "relation": "$type",
                    "relation_properties": "$properties",
                    "relation_chunks": "$chunks",
                    "tail": {"$arrayElemAt": ["$tail_node.name", 0]},
                    "tail_type": {"$arrayElemAt": ["$tail_node.type", 0]},
                    "tail_id": {"$arrayElemAt": ["$tail_node._id", 0]},
                    "tail_properties": {
                        "$arrayElemAt": ["$tail_node.properties", 0]
                    },
                    "tail_chunks": {"$arrayElemAt": ["$tail_node.chunks", 0]},
                }
            },
            {
                "$replaceRoot": {
                    "newRoot": {
                        "_id": "$_id",
                        "chunks": "$relation_chunks",
                        "head_node": {
                            "_id": "$head_id",
                            "name": "$head",
                            "label": "$head_type",
                            "properties": "$head_properties",
                            "chunks": "$head_chunks",
                        },
                        "relation": {
                            "name": "$relation",
                            "properties": "$relation_properties",
                        },
                        "tail_node": {
                            "_id": "$tail_id",
                            "name": "$tail",
                            "label": "$tail_type",
                            "properties": "$tail_properties",
                            "chunks": "$tail_chunks",
                        },
                    }
                }
            },
        ]

        response = await self.db.triple.aggregate(pipeline).to_list(None)
        triples = [TripleWithId(**t) for t in response]
        unique_nodes = []
        unique_node_ids = set()
        for triple in response:
            for node_data in [triple["head_node"], triple["tail_node"]]:
                node_id = node_data["_id"]
                if node_id not in unique_node_ids:
                    unique_nodes.append(NodeWithId(**node_data))
                    unique_node_ids.add(node_id)

        return unique_nodes, triples

    async def _sim_search(
        self, query: str, include_chunks: bool, triple_ids: list[ObjectId]
    ) -> list[dict[str, Any]]:
        """Perform a similarity search.

        Parameters
        ----------
        query
            The query to search for.
        include_chunks
            Whether to include chunks in the search.
        triple_ids
            The list of triple IDs to limit the search for, e.g. for structured subgraph filtering.

        Returns
        -------
        list
            A list of similar triples.
        """
        # Embed query
        # TODO: ENSURE THIS IS **EXACTLY** THE SAME AS THE TRIPLE EMBEDDING MODEL
        response = await self.llm_client.client.embeddings.create(
            input=[query],
            model=(
                self.llm_client.metadata.embedding_name
                if self.llm_client.metadata.embedding_name
                else "text-embedding-3-small"
            ),
            dimensions=1024,  # ONLY WORKS FOR TEXT-EMBEDDING-3-* models
        )
        query_vector = response.data[0].embedding
        logger.info(f"query embedded with {len(query_vector)} dimensions")

        # if len(triple_ids) == 0:
        #     print("triple ids == 0")
        #     return []

        # Find semantically similar triples
        pipeline: list[dict[str, Any]] = [
            {
                "$vectorSearch": {
                    "index": "triple_vector_index",
                    "path": "embedding",
                    "filter": {
                        "created_by": {"$eq": self.user_id},
                        "graph": {"$eq": self.graph_id},
                    },
                    "queryVector": query_vector,
                    "numCandidates": self.settings.api.query_sim_triple_candidates,
                    "limit": self.settings.api.query_sim_triple_limit,
                }
            },
            {
                "$project": {
                    "embedding": 0,
                    "score": {"$meta": "vectorSearchScore"},
                }
            },
            {
                "$lookup": {
                    "from": "node",
                    "localField": "head_node",
                    "foreignField": "_id",
                    "as": "head_node",
                }
            },
            {
                "$lookup": {
                    "from": "node",
                    "localField": "tail_node",
                    "foreignField": "_id",
                    "as": "tail_node",
                }
            },
        ]

        if len(triple_ids) > 0:
            pipeline[0]["$vectorSearch"]["filter"]["_id"] = {"$in": triple_ids}

        if include_chunks:
            pipeline.append(
                {
                    "$lookup": {
                        "from": "chunk",
                        "localField": "chunks",
                        "foreignField": "_id",
                        "as": "chunks",
                    }
                }
            )

        pipeline.append(
            {
                "$project": {
                    "_id": 1,
                    "score": 1,
                    "head": {"$arrayElemAt": ["$head_node.name", 0]},
                    "head_type": {"$arrayElemAt": ["$head_node.type", 0]},
                    "head_id": {"$arrayElemAt": ["$head_node._id", 0]},
                    "head_properties": {
                        "$arrayElemAt": ["$head_node.properties", 0]
                    },
                    "relation": "$type",
                    "relation_properties": "$properties",
                    "tail": {"$arrayElemAt": ["$tail_node.name", 0]},
                    "tail_type": {"$arrayElemAt": ["$tail_node.type", 0]},
                    "tail_id": {"$arrayElemAt": ["$tail_node._id", 0]},
                    "tail_properties": {
                        "$arrayElemAt": ["$tail_node.properties", 0]
                    },
                    **(
                        {
                            "chunks_content": {
                                "$map": {
                                    "input": {"$slice": ["$chunks", 8]},
                                    "as": "chunk",
                                    "in": {"content": "$$chunk.content"},
                                }
                            }
                        }
                        if include_chunks
                        else {}
                    ),
                }
            }
        )

        triples = await self.db.triple.aggregate(pipeline).to_list(None)

        return triples

    async def _relevance_check(
        self, query: str, triples: list[dict[str, Any]]
    ) -> list[dict[str, Any]] | None:
        """
        Evaluate the relevance of each triple in relation to a given query using a language model.

        This method constructs a prompt with the question and a list of triples, asks the language model to determine which triples are relevant by returning a JSON list of indices, and filters the triples based on these indices.

        Parameters
        ----------
        query
            The natural language query to which the relevance of triples is evaluated.
        triples
            A list of dictionaries where each dictionary contains 'head', 'relation', and 'tail' keys representing a triple.

        Returns
        -------
        list
            A list of triples that are deemed relevant to the query if any are found, otherwise None.

        Raises
        ------
        JSONDecodeError
            If there is an error in parsing the model's response.
        Exception
            If an error occurs during the retrieval or processing of the model's response.
        """
        # Build the prompt to include all triples
        prompt = f"Given the question: '{query}', evaluate the relevance of each triple listed below. Respond with a JSON list of indices representing the relevant triples only.\n\n"
        for index, triple in enumerate(triples):
            prompt += f"{index}: {convert_triple_to_text(triple, include_chunks=False)}\n"
        prompt += "\nProvide your response as a JSON array of indices. For example, [0, 2] if the first and third triples are relevant."

        # Send the prompt to the LLM
        response = await self.llm_client.client.chat.completions.create(
            messages=[{"role": "system", "content": prompt}],
            model="gpt-4o",
            temperature=0.1,
            max_tokens=2000,
        )

        try:
            # Extract content and clean up the formatting
            response_content = response.choices[0].message.content
            logger.info(f"relevance check response: {response_content}")
            if response_content is None:
                return None

            raw_content = response_content.strip()
            cleaned_content = (
                raw_content.strip("`").replace("json", "").strip()
            )
            relevant_indices = json.loads(cleaned_content)
            relevant_triples = [triples[i] for i in relevant_indices]
        except JSONDecodeError as je:
            logger.error(f"Failed to parse answer (JSON decoding error): {je}")
            return None
        except Exception as e:
            logger.error(f"Failed to process relevance check: {e}")
            return None

        return relevant_triples if relevant_triples else None

    async def _summarise(
        self, query: str, triples: list[dict[str, Any]], include_chunks: bool
    ) -> str | None:
        """Summarise the relevant triples and the query.

        Parameters
        ----------
        query
            the query that was asked.
        triples
            list of triples that are relevant to the query.
        include_chunks
            whether to include the associated chunks of text in the summarisation.

        Returns
        -------
        str | None
            An optional concise answer based on the facts and the query if any are found.
        """
        triples_str = " ".join(
            [
                convert_triple_to_text(t, include_chunks=include_chunks)
                for t in triples
            ]
        )
        summarisation_prompt = (
            "Provide a concise answer based on these facts"
            + (
                f" and associated chunks of text: {triples_str}. "
                if include_chunks
                else f": {triples_str}. "
            )
            + f"Question: '{query}'. Answer explicitly, using minimal words and without any additional commentary or prose."
        )

        response = await self.llm_client.client.chat.completions.create(
            messages=[{"role": "system", "content": summarisation_prompt}],
            model="gpt-4o",
            temperature=0.1,
            max_tokens=2000,
        )
        response_content = response.choices[0].message.content
        if response_content is None:
            return None
        return response_content.strip()

    async def query(
        self, request: QueryGraphRequest
    ) -> QueryDocumentModel | None:
        """Perform query.

        Parameters
        ----------
        request
            The query request.

        Returns
        -------
        QueryDocumentModel | None
            An optional query document model.

        Raises
        ------
        ValueError
            If the query creation fails for any reason.
        """
        try:
            # Check whether the user has explicitly sent filters
            entities = request.entities
            relations = request.relations

            if entities is None or relations is None:
                logger.info("Retrieving entities and relations from schema")
                # These have not been provided, so get them from the associated schema
                entities, relations = (
                    await self._retrieve_entities_and_relation_types(
                        entities=entities, relations=relations
                    )
                )
            logger.info(
                f"Using entities: {entities}, relations: {relations}, values: {request.values} for query."
            )

            query = request.query
            return_answer = request.return_answer
            include_chunks = request.include_chunks
            response = (
                "Unfortunately, we couldn’t find an answer this time. Feel free to ask another question or provide additional context!"
                if return_answer
                else None
            )
            output_triples: list[TripleWithId] = []
            output_nodes: list[NodeWithId] = []

            # Prepare the query document model
            query_model = QueryDocumentModel(
                id=ObjectId(),
                created_by=str(self.user_id),
                query=QueryParameters(
                    content=query,
                    return_answer=return_answer,
                    include_chunks=include_chunks,
                    values=request.values if request.values else [],
                    entities=entities,
                    relations=relations,
                ),
                graph=self.graph_id,
                status="pending",
            )

            # Create query document in the database
            # Note: Cannot use `create_one` as query content needs to be set to None
            #       which is dropped in this function due to `exclude_none=True`.
            db_created_query = await self.db.query.insert_one(
                query_model.model_dump(by_alias=True)
            )
            db_query = await self.db.query.find_one(
                {
                    "_id": db_created_query.inserted_id,
                    "graph": self.graph_id,
                    "created_by": self.user_id,
                }
            )
            if db_query is None:
                raise ValueError("Failed to create query")
            created_query = QueryDocumentModel(**db_query)

            logger.info(f"created query: {created_query.id}")

            node_ids, triple_ids = (
                await self._retrieve_filtered_triple_and_node_ids(
                    entities=entities,
                    relations=relations,
                    values=request.values,
                )
            )

            logger.info(
                f"node_ids: {len(node_ids)}, triple_ids: {len(triple_ids)}"
            )

            if query is None:
                output_nodes, output_triples = await self._retrieve_triples(
                    triple_ids=triple_ids
                )
            else:
                # Perform semantic search
                similar_triples = await self._sim_search(
                    query=query,
                    include_chunks=include_chunks,
                    triple_ids=triple_ids,
                )

                if similar_triples:
                    logger.info(
                        f"similar triples found: {len(similar_triples)}"
                    )

                    # Perform relevance check
                    relevant_triples = await self._relevance_check(
                        query=query,
                        triples=similar_triples,
                    )
                    if relevant_triples:
                        logger.info(
                            f"relevant triples found: {len(relevant_triples)}"
                        )

                        if return_answer:

                            # Summarise the relevant triples
                            summary = await self._summarise(
                                query=query,
                                triples=relevant_triples,
                                include_chunks=include_chunks,
                            )

                            if summary is not None:
                                response = summary

                        # Populate the triples and nodes for query creation
                        output_triples = await list_triples_by_ids(
                            db=self.db,
                            user_id=self.user_id,
                            graph_id=self.graph_id,
                            triple_ids=[t["_id"] for t in relevant_triples],
                        )
                        output_nodes = []
                        output_node_ids = set()
                        for triple in output_triples:
                            head_node = triple.head_node
                            tail_node = triple.tail_node

                            if head_node.id not in output_node_ids:
                                output_nodes.append(head_node)
                                output_node_ids.add(head_node.id)

                            if tail_node.id not in output_node_ids:
                                output_nodes.append(tail_node)
                                output_node_ids.add(tail_node.id)

            # Prepare the query document model
            created_query.status = "success"
            created_query.response = response
            created_query.triples = output_triples
            created_query.nodes = output_nodes

            await self.db.query.update_one(
                {"_id": ObjectId(created_query.id)},
                {"$set": created_query.model_dump(by_alias=True)},
            )

            if created_query is None or created_query.id is None:
                raise ValueError("Failed to create query")
            return created_query

        except Exception as e:
            logger.error(f"Failed to perform query: {e}", exc_info=True)
            created_query = QueryDocumentModel.model_validate(created_query)
            await self.db.query.update_one(
                {"_id": ObjectId(created_query.id)},
                {
                    "$set": {
                        "status": "failed",
                    }
                },
            )
            raise


async def merge_nodes(
    db: AsyncIOMotorDatabase,
    graph_id: ObjectId,
    user_id: ObjectId,
    from_nodes: List[ObjectId],
    to_node: ObjectId,
) -> NodeWithId:
    """Merge nodes.

    Merge two nodes in the graph.

    Parameters
    ----------
    db : AsyncIOMotorDatabase
        The MongoDB database.
    graph_id : ObjectId
        The ID of the graph.
    from_nodes : List[ObjectId]
        The IDs of the nodes to merge.
    to_node : ObjectId
        The ID of the node to merge to.

    Returns
    -------
    Node
        The merged node.
    """
    # Validate node types are the same
    from_node_docs = await db.node.find(
        {"_id": {"$in": from_nodes}, "graph": graph_id, "created_by": user_id},
        {"type": 1, "properties": 1},
    ).to_list(None)
    to_node_doc = await db.node.find_one(
        {"_id": to_node, "graph": graph_id, "created_by": user_id},
        {"type": 1, "properties": 1},
    )

    if (
        not from_node_docs
        or len(from_node_docs) != len(from_nodes)
        or not to_node_doc
    ):
        raise ValueError("Nodes not found")

    if any([node["type"] != to_node_doc["type"] for node in from_node_docs]):
        raise ValueError("Node types do not match")

    # Merge node properties
    properties = to_node_doc.get("properties", {})
    for node in from_node_docs:
        properties = merge_dicts(properties, node.get("properties", {}))

    # Merge nodes
    async with await db.client.start_session() as session:
        async with session.start_transaction():
            # Update all triples that point to the from_nodes
            await db.triple.update_many(
                {
                    "graph": graph_id,
                    "created_by": user_id,
                    "head_node": {"$in": from_nodes},
                },
                {"$set": {"head_node": to_node}},
                session=session,
            )
            await db.triple.update_many(
                {
                    "graph": graph_id,
                    "created_by": user_id,
                    "tail_node": {"$in": from_nodes},
                },
                {"$set": {"tail_node": to_node}},
                session=session,
            )

            # Delete the from_nodes
            await db.node.delete_many(
                {
                    "_id": {"$in": from_nodes},
                    "graph": graph_id,
                    "created_by": user_id,
                },
                session=session,
            )

            # Update the to_node with the merged properties
            await db.node.update_one(
                {"_id": to_node, "graph": graph_id, "created_by": user_id},
                {"$set": {"properties": properties}},
                session=session,
            )

            # Commit the transaction
            await session.commit_transaction()

    merged_node = await db.node.find_one(
        {"_id": to_node, "graph": graph_id, "created_by": user_id}
    )

    if merged_node is None:
        raise ValueError("Failed to merge nodes")

    return NodeWithId(
        _id=merged_node["_id"],
        name=merged_node["name"],
        label=merged_node["type"],
        properties=merged_node.get("properties", {}),
    )


def clusters_pipeline(
    name: str, type: str, user_id: ObjectId, graph_id: ObjectId
) -> list[dict[str, Any]]:
    """Generate a pipeline to cluster nodes based on name and type.

    Generate a pipeline to cluster nodes based on the name and type of the nodes.

    Parameters
    ----------
    name : str
        The name of the node.
    type : str
        The type of the node.
    user_id : ObjectId
        The ID of the user.
    graph_id : ObjectId
        The ID of the graph.

    Returns
    -------
    list[dict[str, Any]]
        A list of dictionaries representing the pipeline.
    """
    return [
        {
            "$search": {
                "index": "node_index",
                "text": {
                    "query": name,
                    "path": "name",
                    "fuzzy": {"maxEdits": 1},
                },
            }
        },
        {"$match": {"type": type, "graph": graph_id, "created_by": user_id}},
        {
            "$project": {
                "_id": 1,
                "name": 1,
                "label": "$type",
                "properties": 1,
                "similarity": {"$meta": "searchScore"},
            }
        },
        {"$match": {"similarity": {"$gt": len(name) / 5}}},
        {"$group": {"_id": None, "nodes": {"$push": "$$ROOT"}}},
        {"$project": {"_id": 0, "nodes": 1}},
    ]


async def get_similar_nodes(
    db: AsyncIOMotorDatabase,
    graph_id: ObjectId,
    user_id: ObjectId,
    limit: int = 10,
) -> list[list[NodeWithIdAndSimilarity]]:
    """Get similar nodes using fuzzy matching.

    Get similar nodes using fuzzy matching based on the name and type of the nodes.

    Parameters
    ----------
    db : AsyncIOMotorDatabase
        The MongoDB database instance.
    graph_id : ObjectId
        The ID of the graph to query.
    user_id : ObjectId
        The ID of the user.
    limit : int, optional
        The maximum number of similar nodes to return, by default 10.

    Returns
    -------
    list[list[NodeWithIdAndSimilarity]]
        A list of lists of NodeWithIdAndSimilarity objects.
    """
    # Pipeline to get all unique names and types
    names_pipeline: Sequence[Mapping[str, Any]] = [
        {"$match": {"graph": graph_id, "created_by": user_id}},
        {"$group": {"_id": "$name", "type": {"$first": "$type"}}},
        {"$project": {"_id": 0, "name": "$_id", "type": 1}},
    ]

    # Get all unique names and types
    names_cursor = db.node.aggregate(names_pipeline)
    names = await names_cursor.to_list(None)

    # Create clusters for each name and type
    clusters = await asyncio.gather(
        *[
            db.node.aggregate(
                clusters_pipeline(
                    name_doc["name"], name_doc["type"], user_id, graph_id
                )
            ).to_list(None)
            for name_doc in names
        ]
    )
    filtered_clusters = [
        cluster
        for sublist in clusters
        for cluster in sublist
        if len(sublist[0]["nodes"]) > 1
    ]

    # Create a list of nodes sorted by ID
    similar_nodes = [
        sorted(cluster["nodes"], key=lambda x: x["_id"])
        for cluster in filtered_clusters
    ]

    # Convert the list into a set of tuples
    similar_nodes_set = set(
        [
            tuple(dict_to_tuple(node) for node in cluster)
            for cluster in similar_nodes
        ]
    )

    # Convert the set of tuples into a list of dictionaries
    similar_nodes_list = [
        [tuple_to_dict(node) for node in cluster]
        for cluster in similar_nodes_set
    ]

    # Convert the list of dictionaries into a list of NodeWithIdAndSimilarity objects
    similar_nodes_node_with_id = [
        [NodeWithIdAndSimilarity(**node) for node in cluster]
        for cluster in similar_nodes_list
    ]

    # Remove clusters with duplicate names
    def unique_cluster_names(
        clusters: list[list[NodeWithIdAndSimilarity]],
    ) -> list[list[NodeWithIdAndSimilarity]]:
        seen = set()
        unique_clusters = []
        for cluster in clusters:
            cluster_names = sorted([node.name for node in cluster])
            if tuple(cluster_names) not in seen:
                seen.add(tuple(cluster_names))
                unique_clusters.append(cluster)
        return unique_clusters

    similar_nodes_node_with_id = unique_cluster_names(
        similar_nodes_node_with_id
    )

    # Sort the list of NodeWithIdAndSimilarity objects by similarity
    for cluster in similar_nodes_node_with_id:
        cluster.sort(key=lambda x: x.similarity, reverse=True)

    # Sort the similar_nodes_node_with_id list by the sum of the similarities of the nodes in each cluster without the first node (the node the cluster is based on))
    similar_nodes_node_with_id.sort(
        key=lambda x: sum([node.similarity for node in x[1:]]) / len(x[1:]),
        reverse=True,
    )

    # Limit the number of similar nodes
    similar_nodes_node_with_id = similar_nodes_node_with_id[:limit]

    return similar_nodes_node_with_id


async def export_graph_to_cypher(
    db: AsyncIOMotorDatabase,
    graph_id: ObjectId,
    user_id: ObjectId,
) -> List[str]:
    """Export graph to Cypher.

    Export the graph to Cypher format.

    Parameters
    ----------
    db : AsyncIOMotorDatabase
        The MongoDB database connection.
    graph_id : ObjectId
        The ID of the graph.
    user_id : ObjectId
        The ID of the user.

    Returns
    -------
    str
        The Cypher representation of
        the graph.
    """
    triples, _ = await list_triples(
        collection=db["triple"],
        graph_id=graph_id,
        skip=0,
        limit=-1,  # No limit
        order=-1,
        user_id=user_id,
    )

    # Convert TripleWithId objects to dictionaries

    # Ensure triples is not None before iterating
    if triples is not None:
        # Convert TripleWithId objects to dictionaries
        triple_dicts = [triple.model_dump(by_alias=True) for triple in triples]
    else:
        triple_dicts = []

    cypher = generate_cypher_statements(triple_dicts)
    return cypher


async def get_graph_create_details(
    db: AsyncIOMotorDatabase,
    body: CreateGraphBody,
    settings: Settings,
    user_id: ObjectId,
) -> Tuple[int, int, float, float] | None:
    """Get graph creation details including chunk limit based on a provided schema, and estimated time and cost, assuming maximum chunk and pattern sizes."""
    # Retrieve the schema and check pattern limits
    schema = await db.schema.find_one(
        {
            "_id": ObjectId(body.schema_),
            "created_by": user_id,
            "workspace": ObjectId(body.workspace),
        }
    )
    if schema is None:
        raise ValueError("Schema not found.")

    pattern_count = len(schema["patterns"])
    if pattern_count > settings.api.max_patterns:
        raise ValueError(
            f"The schema has too many patterns ({pattern_count}). A maximum of {settings.api.max_patterns} patterns is allowed.",
        )

    # Find the selected chunk count based on the provided filters
    all_chunk_filters = {
        "created_by": user_id,
        "workspaces": ObjectId(body.workspace),
        **(body.filters.mql_filter if body.filters else {}),
    }
    # logger.info(f"All chunk filters: {all_chunk_filters}")
    _chunks = await db.chunk.aggregate(
        [
            {"$project": {"embedding": 0}},
            {"$match": all_chunk_filters},
            {"$count": "count"},
        ]
    ).to_list(None)
    chunks_selected = _chunks[0]["count"] if _chunks else 0
    # logger.info(f"Found {chunks_selected} possible chunks")

    # Calculation of allowed chunks
    chunks_allowed = settings.api.max_chunk_pattern_product // pattern_count

    # Estimations for costs and time
    CHARS_PER_PATTERN_TYPE = 16  # 16 characters per head/tail type (e.g. character, spell, ...) and relation type (e.g., interacts with, casts, ...)
    CHARS_PER_PATTERN_DESCRIPTION = (
        64  # 64 characters per head/tail description and relation description
    )
    PATTERN_PARTS = 3  # head, relation, tail
    PROMPT_OVERHEAD = 1024  # LLM prompt details for triple extraction
    est_pattern_size = PATTERN_PARTS * (
        CHARS_PER_PATTERN_TYPE + CHARS_PER_PATTERN_DESCRIPTION
    )
    est_total_chars = (
        est_pattern_size * pattern_count + PROMPT_OVERHEAD
    ) * chunks_selected
    logger.info(f"Estimated total characters: {est_total_chars}")

    CHAR_PER_TOKEN = 4
    cost_per_token = settings.generative.openai.input_token_cost
    est_input_cost = (
        est_total_chars // CHAR_PER_TOKEN * cost_per_token
    )  # Cost to encode prompt and patterns

    ENTITIES_PER_RESPONSE = (
        8  # 8 entities per response e.g. "Harry Potter", "Ron Weasley", ...
    )
    CHARS_PER_ENTITY_VALUE = (
        32  # 32 characters per entity value e.g. "Harry Potter"
    )
    CHARS_PER_RESPONSE = ENTITIES_PER_RESPONSE * CHARS_PER_ENTITY_VALUE
    output_cost_per_token = settings.generative.openai.output_token_cost
    est_output_cost = (
        (CHARS_PER_RESPONSE * chunks_allowed)
        // CHAR_PER_TOKEN
        * output_cost_per_token
    )  # Cost to generate entities

    # TPM breaches will face a 60 second wait time per breach
    est_total_input_output_chars = est_total_chars + (
        CHARS_PER_RESPONSE * chunks_selected
    )
    tpm_breaches = (
        est_total_input_output_chars
        // CHAR_PER_TOKEN
        // settings.generative.openai.tpm_limit
    )
    logger.info(f"Estimated TPM breaches: {tpm_breaches}")

    TIME_PER_CHUNK_PATTERN = 0.5  # seconds
    time_waiting = 60 * tpm_breaches  # seconds
    time_processing = TIME_PER_CHUNK_PATTERN * chunks_selected  # seconds
    est_time = time_waiting + time_processing

    return (
        chunks_selected,
        chunks_allowed,
        est_input_cost + est_output_cost,
        est_time,
    )


async def create_base_graph(
    name: str,
    user_id: ObjectId,
    workspace_id: ObjectId,
    schema_id: ObjectId | None,
    db: AsyncIOMotorDatabase,
) -> GraphDocumentModel:
    """Create a base graph with optional schema id.

    For graphs created from triples, schemas can be derived from the data so aren't mandatory.
    """
    # Check that workspace exists for the user
    if workspace_id:
        workspace_exists = await db.workspace.find_one(
            {"_id": workspace_id, "created_by": user_id}
        )
        if workspace_exists is None:
            raise NotFoundException("Workspace not found.")

    graph = GraphDocumentModel(
        id=None,
        name=name,
        created_by=user_id,
        workspace=workspace_id,
        schema_=schema_id,
        status="creating",
    )
    new_graph = await db.graph.insert_one(
        graph.model_dump(by_alias=True, exclude_none=True)
    )

    retrieved_graph = await db.graph.find_one(
        {
            "_id": new_graph.inserted_id,
            "created_by": user_id,
            "workspace": workspace_id,
        }
    )

    if retrieved_graph is None:
        raise NotFoundException("Failed to retrieve graph.")

    graph.id = ObjectId(retrieved_graph["_id"])
    logger.info("Base graph created")

    return graph


async def create_schema_from_triples(
    name: str,
    triples: list[Triple],
    user_id: ObjectId,
    db: AsyncIOMotorDatabase,
    workspace: ObjectId,
) -> SchemaDocumentModel:
    """Create a schema from triples."""
    if len(triples) == 0:
        raise ValueError("No triples provided.")

    # If a schema is not provided, one is auto-generated.
    logger.info("No schema provided. Creating schema from triples.")

    derived_entity_types: Set[str] = set()
    derived_relation_types: Set[str] = set()
    derived_patterns: Set[Tuple[str, str, str]] = set()
    for triple in triples:
        head_type = triple.head_type
        tail_type = triple.tail_type
        relation = triple.relation
        derived_entity_types.add(head_type)
        derived_entity_types.add(tail_type)
        derived_relation_types.add(relation)
        derived_patterns.add((head_type, relation, tail_type))

    logger.info(f"Derived Entity types: {derived_entity_types}")
    logger.info(f"Derived Relation types: {derived_relation_types}")
    logger.info(f"Derived Patterns: {derived_patterns}")

    entities: list[SchemaEntity] = [
        SchemaEntity(name=e, description=AUTOGEN_DESCRIPTION)
        for e in derived_entity_types
    ]
    relations: list[SchemaRelation] = [
        SchemaRelation(name=r, description=AUTOGEN_DESCRIPTION)
        for r in derived_relation_types
    ]
    patterns: list[TriplePattern] = [
        TriplePattern(
            head=p[0],
            relation=p[1],
            tail=p[2],
            description=AUTOGEN_DESCRIPTION,
        )
        for p in derived_patterns
    ]

    # Create a schema
    schema = await create_one(
        collection=db["schema"],
        document_model=SchemaDocumentModel,
        user_id=user_id,
        document=SchemaCreate(
            name=f"{AUTOGEN_DESCRIPTION} (graph: {name})",
            entities=entities,
            relations=relations,
            patterns=patterns,
            workspace=workspace,
        ),
    )
    if schema is None:
        raise ValueError("Failed to create schema from triples.")
    logger.info(f"Schema created: {schema}")

    return SchemaDocumentModel.model_validate(schema)


def validate_triples(
    patterns: list[SchemaTriplePattern],
    triples: list[Triple],
    strict_mode: bool = True,
) -> None | Set[Tuple[str, str, str]]:
    """Validate triples against schema patterns.

    Parameters
    ----------
    patterns:
        List of schema patterns.
    triples:
        List of triples to validate.
    strict_mode:
        If True, any invalid triples will throw an exception.
        If False, invalid triples will be added and the schema will be extended.

    Returns
    -------
    None | Set[Tuple[str, str, str]]
        If `strict_mode` is False, returns a set of new patterns that can be added to the schema.

    Raises
    ------
    ValueError: If a triple does not match a schema pattern.
    """
    patterns_for_validation: Set[Tuple[str, str, str]] = set(
        (p.head.name, p.relation.name, p.tail.name) for p in patterns
    )
    logger.info(f"Schema patterns for validation: {patterns_for_validation}")
    new_patterns: Set[Tuple[str, str, str]] = set()

    # Validate triples
    for idx, triple in enumerate(triples):
        if (
            triple.head_type,
            triple.relation,
            triple.tail_type,
        ) not in patterns_for_validation:

            if strict_mode:
                logger.info(
                    f"Triple {triple.head_type} {triple.relation} {triple.tail_type} not in schema."
                )
                raise ValueError(
                    f"Triple {idx} not in schema: {triple.model_dump()}."
                )
            else:
                logger.info(
                    f"Triple {triple.head_type} {triple.relation} {triple.tail_type} not in schema. Saving."
                )
                new_patterns.add(
                    (triple.head_type, triple.relation, triple.tail_type)
                )

    if strict_mode:
        return None
    return new_patterns


async def extend_schema(
    db: AsyncIOMotorDatabase,
    schema_id: ObjectId,
    user_id: ObjectId,
    entity_types: Set[str] = set(),
    relation_types: Set[str] = set(),
    patterns: Set[Tuple[str, str, str]] = set(),
) -> None:
    """Extend schema with new patterns.

    Parameters
    ----------
    db:
        The MongoDB database connection.
    schema_id:
        The ID of the schema.
    user_id:
        The ID of the user.
    patterns:
        The new patterns to add to the schema.

    Raises
    ------
    NotFoundException: If the schema is not found.
    """
    derived_entity_types: Set[str] = entity_types
    derived_relation_types: Set[str] = relation_types

    for p in patterns:
        derived_entity_types.add(p[0])
        derived_entity_types.add(p[2])
        derived_relation_types.add(p[1])

    logger.info(f"New Patterns: {patterns}")
    logger.info(f"Derived Entity types: {derived_entity_types}")
    logger.info(f"Derived Relation types: {derived_relation_types}")

    # Get the existing schema
    schema = await get_one(
        collection=db["schema"],
        document_model=SchemaDocumentModel,
        user_id=user_id,
        id=schema_id,
    )
    schema = SchemaDocumentModel.model_validate(schema)

    if schema is None:
        logger.info(f"Schema with id '{schema_id}' does not exist.")
        raise NotFoundException("Schema not found.")

    existing_entity_types = set(
        e.name for e in schema.entities if schema.entities is not None
    )
    existing_relation_types = set(
        r.name for r in schema.relations if schema.relations is not None
    )

    new_entity_types = derived_entity_types - existing_entity_types
    new_relation_types = derived_relation_types - existing_relation_types

    new_entities: list[SchemaEntity] = [
        SchemaEntity(name=e, description=AUTOGEN_DESCRIPTION)
        for e in new_entity_types
    ]
    new_relations: list[SchemaRelation] = [
        SchemaRelation(name=r, description=AUTOGEN_DESCRIPTION)
        for r in new_relation_types
    ]
    new_patterns: list[SchemaTriplePattern] = [
        SchemaTriplePattern(
            head=SchemaEntity(name=p[0], description=AUTOGEN_DESCRIPTION),
            relation=SchemaRelation(
                name=p[1], description=AUTOGEN_DESCRIPTION
            ),
            tail=SchemaEntity(name=p[2], description=AUTOGEN_DESCRIPTION),
            description=AUTOGEN_DESCRIPTION,
        )
        for p in patterns
    ]

    # Update the schema
    schema.entities.extend(new_entities)
    schema.relations.extend(new_relations)
    schema.patterns.extend(new_patterns)

    await db.schema.update_one(
        {"_id": schema_id}, {"$set": schema.model_dump()}
    )
    logger.info(
        f'Schema "{schema_id}" extended with new entity types, relation types, or patterns.'
    )


async def create_or_update_graph_from_triples(
    background_tasks: BackgroundTasks,
    triples: list[Triple],
    db: AsyncIOMotorDatabase,
    db_client: AsyncIOMotorClient,
    user_id: ObjectId,
    llm_client: LLMClient,
    graph_name: str | None = None,
    graph_id: ObjectId | None = None,
    workspace_id: ObjectId | None = None,
    schema_id: ObjectId | None = None,
    strict_mode: bool = True,
) -> TaskDocumentModel:
    """Create or update a graph from triples.

    Either populates an existing graph or creates a new graph and populates with triples.
    """
    # Check that workspace exists for the user
    if workspace_id:
        workspace_exists = await db.workspace.find_one(
            {"_id": workspace_id, "created_by": user_id}
        )
        if workspace_exists is None:
            raise NotFoundException("Workspace not found.")

    if graph_id is None:
        if workspace_id is None:
            raise ValueError("No graph provided and no workspace provided.")

        if graph_name is None:
            raise ValueError("No graph provided and no graph name provided.")

        logger.info(f'Creating base graph "{graph_name}"')
        graph = await create_base_graph(
            name=graph_name,
            user_id=user_id,
            workspace_id=workspace_id,
            schema_id=schema_id,
            db=db,
        )
        graph_id = ObjectId(graph.id) if graph.id else None

    # Get graph details
    db_graph = await db.graph.find_one(
        {"_id": graph_id, "created_by": user_id},
        {"name": 1, "schema_id": 1, "workspace": 1},
    )
    if db_graph is None:
        raise NotFoundException("Graph not found.")
    db_graph_name = db_graph.get("name")
    schema_id = db_graph.get("schema_id", None)
    db_workspace_id = db_graph.get("workspace")

    schema: SchemaDocumentModel | None = None
    if schema_id is None:
        # User did not provide a schema, create one from the triples
        schema = await create_schema_from_triples(
            name=db_graph_name,
            triples=triples,
            workspace=db_workspace_id,
            db=db,
            user_id=user_id,
        )
    else:
        # User provided an existing schema, use this to validate the provided triples.
        # If `strict_mode` is True, any invalid triples will throw an exception.
        # If `strict_mode` is False, invalid triples will be added and the schema will be extended.
        logger.info("Schema provided")

        schema = await get_one(
            collection=db["schema"],
            document_model=SchemaDocumentModel,  # type: ignore
            user_id=user_id,
            id=schema_id,
        )
        if schema is None:
            logger.info(f"Schema with id '{schema_id}' does not exist.")
            raise NotFoundException("Schema not found.")

        # Validate triples against schema patterns
        logger.info(f"Checking triples in strict mode: {strict_mode}")
        new_patterns = validate_triples(
            patterns=schema.patterns, triples=triples, strict_mode=strict_mode
        )

        # If new patterns and strict_mode=False, extend it
        if new_patterns is not None:
            logger.info("New patterns detected - extending schema.")
            await extend_schema(
                db=db,
                schema_id=schema_id,
                user_id=user_id,
                patterns=new_patterns,
            )

    # Update graph with `schema_id`
    await db.graph.update_one(
        {"_id": graph_id}, {"$set": {"schema_id": ObjectId(schema.id)}}
    )

    updated_triples = await apply_rules(
        db=db,
        extracted_triples=triples,
        workspace_id=db_workspace_id,
        graph_id=graph_id,  # type: ignore
        user_id=user_id,
        errors=[],
    )

    task = await create_task(
        _db=db,
        _user_id=user_id,
        _background_tasks=background_tasks,
        func=build_graph,
        db=db,
        db_client=db_client,
        llm_client=llm_client,
        graph_id=graph_id,
        user_id=user_id,
        triples=updated_triples,
    )

    return task
