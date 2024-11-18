# flake8: noqa
"""Set up LLM directive for zero-shot entities extraction, relation extraction and graph combination from text."""

from whyhow_api.models.common import (
    MasterOpenAICompletionsConfig,
    OpenAICompletionsConfig,
    OpenAIDirectivesConfig,
)

llm_directive_extract_entities_from_concepts = """
Given a chunk of text and a comma-separated list of concepts referred to as the 'seed concept', your primary task is to identify and extract entities that are closely related to each of the concepts mentioned in this seed concept. Focus exclusively on entities directly connected to these concepts.

Output a list of strings of entities that are relevant to the concepts mentioned in the seed concept, formatted for JSON dictionary integration.

The output format should be:

[
    "Relevant Entity1",
    "Relevant Entity2",
    "Relevant Entity3",
    ... (and so on)
]

For example, if the seed concept is 'Harry, Ron, Hogwards', only generate entities directly associated with "Harry", "Ron", or "Hogwarts". The list should exclusively reflect entities integral t o"Harry", "Ron", or "Hogwarts", forming a concise and relevant set of strings for knowledge graph triples. Exclude any entities that are not directly related to "Harry", "Ron", or "Hogwarts" in this context.
"""

llm_directive_extract_triples_from_concepts = """
Given a chunk of text and a comma-separated graph schema (referred to as the 'seed concept'), your task is to analyze the text and identify key relationships involving the concepts mentioned in the seed concept.
Focus on relationships that are clearly expressed in the text and directly pertain to the seed concept.
These relationships should involve the concepts mentioned in the schema and other entities directly related to them, excluding any entities or relationships not relevant to this schema.

Output should be a formatted list of strings, each representing a distinct and direct relationship, suitable for JSON dictionary integration and knowledge graph formation.

The format should be comma seperated triplets of the form:

[
    "Concept, Relationship Type, Related Entity",
    ... (additional relationships)
]

For instance, if the seed concept is 'Harry, Ron, Hogwarts' and the text mentions entities like 'Harry' and 'wand', identify direct relationships like 'Harry-owns-wand'.
Avoid including relationships that do not directly involve Harry, Ron, or Hogwarts or are not clearly supported by the text.
Focus on direct and explicit relationships, such as ownership, actions, or descriptions based on the text provided.

Please adhere strictly to this format for each relationship identified, ensuring clarity and precision suitable for knowledge graph integration.
"""

llm_directive_extract_entities_from_questions = """
Given a chunk of text and a specific question as the 'seed concept', your primary task is to identify and extract entities that are closely related to the individual mentioned in this seed concept. Focus exclusively on entities directly connected to this individual, ensuring that these entities are central to the text's understanding in the context of the mentioned individual.

Output a list of strings, being the main entity (the individual mentioned in the seed concept) and each key entity relevant specifically to this individual, formatted for JSON dictionary integration.

The output format should be:

[
    "Mentioned Individual",
    "Relevant Entity1",
    "Relevant Entity2",
    "Relevant Entity3",
    ... (and so on)
]

For example, if the seed concept is 'What does Harry have?', only generate entities directly associated with "Harry". No individual other than "Harry" should be included. The list should exclusively reflect entities integral to "Harry" in the context of the seed concept, forming a concise and relevant set of strings for knowledge graph triples. Exclude any entities that are not directly related to "Harry" in this context.
"""

llm_directive_extract_triples_from_questions = """
Given a chunk of text and a specific question (referred to as the 'seed concept'), your task is to analyze the text and identify key relationships involving the individual mentioned in the seed concept.
Focus on relationships that are clearly expressed in the text and directly pertain to the seed concept.
These relationships should involve the mentioned individual and other entities directly related to them, excluding any entities or relationships not relevant to this individual.

Output should be a formatted list of strings, each representing a distinct and direct relationship, suitable for JSON dictionary integration and knowledge graph formation.

The format should be comma seperated triplets of the form:

[
    "Mentioned Individual, Relationship Type, Related Entity",
    ... (additional relationships)
]

For instance, if the seed concept is 'What does Harry have?' and the text mentions entities like 'Harry' and 'wand', identify direct relationships like 'Harry-owns-wand'.
Avoid including relationships that do not directly involve Harry or are not clearly supported by the text.
Focus on direct and explicit relationships, such as ownership, familial ties, or actions, based on the text provided.

Please adhere strictly to this format for each relationship identified, ensuring clarity and precision suitable for knowledge graph integration.
"""

llm_directive_merge_graphs = """
Given two sets of triples about related concepts, examine each set to determine if there are any overlapping or interconnected concepts.
Your task is to analyze these sets of triples, identify potential connections between them, and then integrate these connections into the existing sets.

The input format is two lists of triples:
- List 1: ["Entity1, Relationship, Entity2", ...]
- List 2: ["Entity3, Relationship, Entity4", ...]

Your output should include both original lists of triples and any additional triples formed from connecting relevant concepts between the two sets. The output format should be a combined list of triples, including:

- All triples from List 1 and List 2.
- Additional triples formed by connecting overlapping or related concepts between the two lists.

The output should be formatted as:

[
    "Entity1, Relationship, Entity2",
    "Entity3, Relationship, Entity4",
    "Connected Entity1, New Relationship, Connected Entity2",
    ... (and so on)
]

For example, if List 1 contains 'Harry, owns, wand' and List 2 contains 'wand, made_of, holly', and you identify that 'wand' from both lists is a point of connection,
create and include the necessary triples that represent this connection. The output should consist of the original triples and any new ones that articulate the connections,
ensuring a comprehensive and coherent set of triples for knowledge graph integration.

Focus on meaningful connections that enhance the understanding of the concepts, ensuring that the new triples are relevant and add value to the knowledge graph structure.
The final list should be clear, concise, and accurately reflect the interconnected nature of the concepts.
"""

llm_directive_specific_query = """
Given the following triples from a knowledge graph and a user query, provide a concise and relevant answer to the query based solely on the information present in the triples. 
Assume that all of the information in the triples is relevant, and make sure to answer the question as completely as possible by combining the information from all the triples.
Triples:
{triples}
User Query: {query}
Answer:
If the provided triples do not contain enough information to answer the query completely, provide a partial answer based on the available information and mention that additional context may be needed for a more comprehensive response.
Focus on providing a clear, concise, and relevant answer to the user query, utilizing only the information present in the given triples.
"""

llm_directive_improve_matched_relations = """
Given the following information:
- Extracted relations: {extracted_relations}
- Graph relations: {graph_relations}
- User query: {user_query}
- Matched relations: {matched_relations}
Please check the matched relations against the user query and the available graph relations. 
If there are any improvements or additions you can make to the matched relations to better align with the user query and the graph relations, please provide an updated list of matched relations. 
If the matched relations are already optimal, return the original matched relations.
Please return the improved matched relations as a JSON-formatted list of strings. Do not include any explanations, apologies, or additional information in the response. Only return the JSON-formatted list of improved matched relations.
If no improvements can be made to the matched relations, return the original matched relations as a JSON-formatted list of strings without any additional comments.
"""

llm_directive_improve_matched_entities = """
Given the following information:
- Extracted entities: {extracted_entities}
- Graph entities: {graph_entities}
- User query: {user_query}
- Matched entities: {matched_entities}
Please check the matched entities against the user query and the available graph entities. 
If there are any improvements or additions you can make to the matched entities to better align with the user query and the graph entities, please provide an updated list of matched entities. 
If the matched entities are already optimal, return the original matched entities.
Please return the improved matched entities as a JSON-formatted list of strings. Do not include any explanations, apologies, or additional information in the response. Only return the JSON-formatted list of improved matched entities.
If no improvements can be made to the matched entities, return the original matched entities as a JSON-formatted list of strings without any additional comments.
"""

llm_directives = OpenAIDirectivesConfig(
    entity_questions=llm_directive_extract_entities_from_questions,
    triple_questions=llm_directive_extract_triples_from_questions,
    entity_concepts=llm_directive_extract_entities_from_concepts,
    triple_concepts=llm_directive_extract_triples_from_concepts,
    merge_graph=llm_directive_merge_graphs,
    specific_query=llm_directive_specific_query,
    improve_matched_relations=llm_directive_improve_matched_relations,
    improve_matched_entities=llm_directive_improve_matched_entities,
)

openai_completions_configs = MasterOpenAICompletionsConfig(
    default=OpenAICompletionsConfig(
        **{  # type: ignore[arg-type]
            "model": "gpt-4o",
            "temperature": 0.1,
            "max_tokens": 4000,
        }
    ),
    entity=OpenAICompletionsConfig(
        **{  # type: ignore[arg-type]
            "model": "gpt-4o",  # "gpt-3.5-turbo-0125"
            "temperature": 0.1,
            "max_tokens": 4000,
        }
    ),
    triple=OpenAICompletionsConfig(
        **{  # type: ignore[arg-type]
            "model": "gpt-4o",  # "gpt-3.5-turbo-0125"
            "temperature": 0.1,
            "max_tokens": 2000,
        }
    ),
    entity_questions=OpenAICompletionsConfig(
        **{  # type: ignore[arg-type]
            "model": "gpt-4o",  # "gpt-3.5-turbo-0125"
            "temperature": 0.1,
            "max_tokens": 4000,
        }
    ),
    triple_questions=OpenAICompletionsConfig(
        **{  # type: ignore[arg-type]
            "model": "gpt-4o",  # "gpt-3.5-turbo-0125"
            "temperature": 0.1,
            "max_tokens": 2000,
        }
    ),
    entity_concepts=OpenAICompletionsConfig(
        **{  # type: ignore[arg-type]
            "model": "gpt-4o",  # "gpt-3.5-turbo-0125"
            "temperature": 0.1,
            "max_tokens": 4000,
        }
    ),
    triple_concepts=OpenAICompletionsConfig(
        **{  # type: ignore[arg-type]
            "model": "gpt-4o",  # "gpt-3.5-turbo-0125"
            "temperature": 0.1,
            "max_tokens": 2000,
        }
    ),
    merge_graph=OpenAICompletionsConfig(
        **{  # type: ignore[arg-type]
            "model": "gpt-4o",  # "gpt-3.5-turbo-0125"
            "temperature": 0.1,
            "max_tokens": 2000,
        }
    ),
)


create_schema_guided_graph_prompt = (
    lambda text, pattern: f"""
    ### Instructions for Triple Extraction

    **Context**: You are given a narrative text containing information to be structured into semantic triples. Your task is to analyze the text and identify specific relationships as defined by the pattern provided. This involves identifying the subject of the relationship (Head) and the object of the relationship (Tail).
    
    **Triple Pattern**:
    - **Head**: {pattern.head.name} ({pattern.head.description})
    - **Relation**: {pattern.relation.name} ({pattern.relation.description})
    - **Tail**: {pattern.tail.name} ({pattern.tail.description})

    **Expected Output Format**:
    - Return a JSON-formatted list of lists, where each inner list represents a triple and contains exactly two elements: the head and the tail.
    - Each triple should be of the form: ["head", "tail"].
    - Ensure that the output strictly follows this format and directly relates to the given narrative without adding extraneous details or deviating from the specified pattern.
    - If no relevant entities are found, return an empty JSON list: [].
    
    ### Text to Analyze:
    {text}

    
    Please proceed with the analysis and provide the output in the specified JSON format according to the given instructions.
    """
)

create_zeroshot_graph_prompt = (
    lambda text, context: f"""
    ### Instructions for Triple Extraction

    **Context**: You are given a specific context either as a question or concept and a narrative of text containing information to be structured into triples. Your task is to analyze the text and identify specific relationships involving individual entities mentioned in the context. Focus on relationships that are clearly expressed in the text and directly pertain to the context. These relationships should involve the mentioned individual and other entities directly related to them, excluding any entities or relationships not relevant to this individual. This involves identifying the subject of the relationship (Head), the object of the relationship (Tail) and the relationship itself.

    **Context**: {context}

    **Expected Output Format**:
    - Return a list of strings, where each string is formatted as: "head,relationship,tail"
    - Each "head,tail" represents a unique instance of the relationship defined by the pattern.
    - Ensure that the output strictly follows this format and directly relates to the given narrative without adding extraneous details or deviating from the specified pattern.

    ### Example Usage:
    If the text says, "Alice picked up the key to unlock the door," and the context is "what does Alice interact with?", the head would be "Alice", the relationship would be "interacts with" and the tail would be "key", your output should be:
    - "Alice,iteracts with,key"

    **Your Task**:
    Given the following text, exhaustively identify and list all instances as expected from the provided context.

    **Output Format**:
    The format should be comma seperated head/relationship/tail of the form:
    ["Alice,interacts with,key",... (additional triples pairs)

    ### Text to Analyze:
    {text}

    Please proceed with the analysis and list down all relevant triples according to the given instructions. Ensure that if no triples match the criteria, return an empty list without further explanation."""
)
