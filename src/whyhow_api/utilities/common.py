"""Common utilities."""

import logging
import string
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Set, Tuple

import logfire
from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase

from whyhow_api.exceptions import NotFoundException
from whyhow_api.models.common import LLMClient
from whyhow_api.schemas.base import AfterAnnotatedObjectId
from whyhow_api.schemas.graphs import Triple

logger = logging.getLogger(__name__)


async def embed_texts(
    llm_client: LLMClient, texts: list[str], batch_size: int = 2048
) -> List[Any]:
    """Embed a list of texts using the OpenAI API."""
    # Logfire trace of LLM client
    logfire.instrument_openai(llm_client.client)

    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        logger.info(f"Processing batch from {i} to {i + batch_size}")

        if len(batch) > 2048:
            raise RuntimeError("Texts must be 2048 items or less.")

        # Get embeddings for the batch of texts
        response = await llm_client.client.embeddings.create(
            input=batch,
            model=(
                llm_client.metadata.embedding_name
                if llm_client.metadata.embedding_name
                else "text-embedding-3-small"
            ),
            dimensions=1536,
        )
        batch_embeddings = [d.embedding for d in response.data]
        all_embeddings.extend(batch_embeddings)

    logger.info(f"Finished processing {len(texts)} texts.")
    return all_embeddings


def compress_triples(triples: List[Tuple[str, str, str]]) -> str:
    """
    Compress semantic triples.

    Compress a list of triples into a more compact string format by
    aggregating similar relationships under a single head entity.

    Parameters
    ----------
    triples : List[Tuple[str, str, str]]
        A list of triples in the format (head, relation, tail).

    Returns
    -------
    str
        A compressed string representation where multiple tails are grouped
        under the same head and relation in a single line,
        sorted alphabetically.

    Example
    -------
    triples = [
        ('Jerry', 'friends with', 'Kramer'),
        ('Jerry', 'friends with', 'Elaine'),
        ('Jerry', 'friends with', 'George')
    ]
    compressed_output = compress_triples(triples)
    Outputs: "Jerry friends with Elaine, George, Kramer"
    """
    structured_data: DefaultDict[Any, DefaultDict[Any, Set[str]]] = (
        defaultdict(lambda: defaultdict(set))
    )  # Use sets to automatically drop duplicates

    for head, relation, tail in triples:
        # Normalize the relation string
        relation = relation.replace("_", " ").lower()
        # Add tail to the set associated with the (head, relation) key
        structured_data[head][relation].add(tail)

    # Building the output string
    output = []
    for head, relations in sorted(structured_data.items()):
        for relation, tails in sorted(relations.items()):
            output.append(f"{head} {relation} {', '.join(sorted(tails))}")
    return "\n".join(output)


def create_chunk_triples(
    triples: List[Triple],
    chunk_map: Dict[str, str],
) -> List[Triple]:
    """Create linked triples for each chunk in the chunk_map."""
    linked_triples: List[Triple] = []

    if len(triples) == 0:
        return linked_triples
    processed = (
        {}
    )  # Cache to store processed counts for each unique identifier

    # Create a unique set of identifiers from all heads and tails
    identifiers = set((t.head, t.head_type) for t in triples) | set(
        (t.tail, t.tail_type) for t in triples
    )

    # Process each unique identifier only once
    for identifier, type_name in identifiers:
        processed[(identifier, type_name)] = count_frequency(
            search_str=identifier, data_dict=chunk_map
        )

    # Create linked triples for each identifier based
    # on the pre-processed data
    for identifier, type_name in identifiers:
        counts = processed[(identifier, type_name)]
        for chunk_id, freq in counts.items():
            linked_triples.append(
                Triple(
                    head=chunk_id,
                    head_type="Chunk",
                    tail=identifier,
                    # Set tail_type based on the original triple's tail_type
                    tail_type=type_name,
                    relation="Contains",
                    head_properties={
                        "id": chunk_id,
                        "text": chunk_map[chunk_id],
                    },
                    relation_properties={"count": freq},
                )
            )

    return linked_triples


def remove_punctuation(text: str) -> str:
    """Remove punctuation from text."""
    translator = str.maketrans("", "", string.punctuation)
    return text.translate(translator)


def count_frequency(
    search_str: str, data_dict: Dict[str, str]
) -> Dict[str, int]:
    """
    Count frequency of mentions.

    Search for search_str within the 'text' of each entry
    in the dictionary and count the occurrences.

    Parameters
    ----------
    search_str : str
        The string to search for.
    data_dict : Dict[str, str]
        The dictionary to search through. Expected to have a structure
        {"id": "str"}.

    Returns
    -------
    Dict[str, int]
        A dictionary with the structure {id: frequency, ...} where `id`
        is from the nested dictionaries and `frequency` is the count
        of how many times search_str appears within the text of each entry.
    """
    # Initialize a dictionary to hold the frequency count
    # for each id
    frequency_count: Dict[str, int] = {}

    # Normalize search_str for consistent matching
    normalized_search_str = remove_punctuation(search_str.lower())

    # Iterate through each key, value pair in the data_dict
    for _id, text in data_dict.items():
        # Ensure the existence of the "text" key and normalize
        # the text for consistent matching
        text = remove_punctuation(text.lower())

        # Count occurrences of search_str in text
        occurrences = text.count(normalized_search_str)

        # Update the frequency count for the corresponding id with the
        # count of occurrences
        if occurrences > 0:
            frequency_count[_id] = frequency_count.get(_id, 0) + occurrences

    return frequency_count


def dict_to_tuple(d: Dict[str, Any]) -> tuple[tuple[str, Any], ...]:
    """Convert a dictionary to a tuple.

    Recursively convert a dictionary to a tuple of tuples.

    Parameters
    ----------
    d : Dict[str, Any]
        The dictionary to convert.

    Returns
    -------
    Tuple[Tuple[str, Any], ...]
        A tuple of tuples representing the dictionary.
    """
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = dict_to_tuple(v)
    return tuple(d.items())


def tuple_to_dict(t: tuple[Any, ...]) -> Dict[str, Any]:
    """Convert a tuple to a dictionary.

    Recursively convert a tuple to a dictionary.

    Parameters
    ----------
    t : Tuple[Any, ...]
        The tuple to convert.

    Returns
    -------
    Dict[str, Any]
        A dictionary representing the tuple.
    """
    d = {}
    for k, v in t:
        if isinstance(v, tuple):
            v = tuple_to_dict(v)
        d[k] = v
    return d


def clean_text(text: str) -> str:
    """Clean text by allowing comma, semicolons, periods, and spaces; replacing underscores with spaces for a more natural read."""
    allowed_chars = {",", ";", "."}  # Set of allowed punctuation
    return (
        "".join(
            (
                char
                if char.isalnum() or char in allowed_chars or char == " "
                else " "
            )
            for char in text
        )
        .replace("_", " ")
        .strip()
    )


async def check_existing(
    db: AsyncIOMotorDatabase,
    collection: str,
    ids: Any,
    additional_query: dict[str, Any] = {},
) -> list[AfterAnnotatedObjectId]:
    """
    Check if the provided IDs exist in the collection.

    Parameters
    ----------
    db : AsyncIOMotorDatabase
        The database to check the IDs against.
    collection : str
        The collection to check the IDs against.
    ids : list[ObjectId]
        The list of IDs to check.
    additional_query : dict[str, Any], optional
        Additional query to filter the IDs, by default {}.

    Raises
    ------
    NotFoundException
        If any of the IDs do not exist in the collection.
    """
    casted_ids: list[AfterAnnotatedObjectId] = []

    if ids:
        # Convert the IDs to a list of ObjectIds
        try:
            casted_ids = [ObjectId(id_) for id_ in ids]
        except Exception as e:
            raise NotFoundException(
                f"Invalid {collection.capitalize()} ID: {str(e)}"
            )

        # Get the list of IDs that do not exist in the collection
        found_ids = (
            await db[collection]
            .find({"_id": {"$in": casted_ids}, **additional_query}, {"_id": 1})
            .distinct("_id")
        )
        missing_ids = set(casted_ids) - set(found_ids)

        if missing_ids:
            raise NotFoundException(
                f"{collection.capitalize()} IDs not found: {', '.join(str(id_) for id_ in missing_ids)}"
            )

    return casted_ids
