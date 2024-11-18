# src/whyhow_api/utilities/cypher_export.py

"""Cypher exports."""

from collections import defaultdict
from typing import Any, Dict, List


def generate_cypher_statements(triples: List[Dict[str, Any]]) -> List[str]:
    """
    Generate Cypher statements for creating nodes and relationships in Neo4j.

    This function takes a list of triples representing a graph structure and generates
    Cypher statements to create the corresponding nodes and relationships in Neo4j.
    It also creates unique constraints for each node label to ensure data integrity.

    Parameters
    ----------
    triples : List[Dict[str, Any]]
        A list of dictionaries, where each dictionary represents a triple containing
        'head_node', 'relation', and 'tail_node' information.

    Returns
    -------
    str
        A string containing Cypher statements to create constraints, nodes, and relationships.

    Notes
    -----
    - The function creates unique constraints on the 'name' property for each unique node label.
    - It uses MERGE statements to create nodes and relationships, preventing duplication.
    - Special characters in node names are escaped to prevent Cypher syntax errors.

    Example
    -------
    >>> triples = [
    ...     {
    ...         "head_node": {"label": "Person", "name": "Alice"},
    ...         "relation": {"name": "KNOWS"},
    ...         "tail_node": {"label": "Person", "name": "Bob"}
    ...     }
    ... ]
    >>> print(generate_cypher_statements(triples))
    CREATE CONSTRAINT unique_Person_name IF NOT EXISTS FOR (n:Person) REQUIRE n.name IS UNIQUE;
    MERGE (h:Person {name: 'Alice'}) MERGE (t:Person {name: 'Bob'}) MERGE (h)-[:`KNOWS`]->(t);
    """
    unique_labels = set()
    node_properties = defaultdict(set)
    relationships = []

    for triple in triples:
        head_label = triple["head_node"]["label"]
        tail_label = triple["tail_node"]["label"]
        unique_labels.update([head_label, tail_label])

        node_properties[head_label].add("name")
        node_properties[tail_label].add("name")

        relationships.append(
            (
                head_label,
                escape_string(triple["head_node"]["name"]),
                triple["relation"]["name"],
                tail_label,
                escape_string(triple["tail_node"]["name"]),
            )
        )

    cypher_statements = []

    # Create constraints
    for label in unique_labels:
        cypher_statements.append(
            f"CREATE CONSTRAINT unique_{label}_name IF NOT EXISTS FOR (n:{label}) REQUIRE n.name IS UNIQUE;"
        )

    # Create nodes and relationships
    for (
        head_label,
        head_name,
        rel_type,
        tail_label,
        tail_name,
    ) in relationships:
        cypher_statements.append(
            f"MERGE (h:{head_label} {{name: '{head_name}'}}) "
            f"MERGE (t:{tail_label} {{name: '{tail_name}'}}) "
            f"MERGE (h)-[:`{rel_type}`]->(t);"
        )

    return cypher_statements


def escape_string(s: str) -> str:
    r"""
    Escape special characters in a string for use in Cypher queries.

    This function escapes special characters that could interfere with Cypher syntax,
    ensuring that the string can be safely used in Cypher statements.

    Parameters
    ----------
    s : str
        The input string to be escaped.

    Returns
    -------
    str
        The input string with special characters escaped.

    Notes
    -----
    The function escapes the following characters:
    - Backslash (\)
    - Single quote (')
    - Double quote (")
    - Newline (\n)
    - Carriage return (\r)
    - Tab (\t)

    Example
    -------
    >>> print(escape_string("Alice's \"quote\""))
    Alice\'s \\"quote\\"
    """
    return (
        s.replace("\\", "\\\\")
        .replace("'", "\\'")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )
