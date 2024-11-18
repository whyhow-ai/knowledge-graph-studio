"""Provides a mechanism for setting up demo data for testing and new user purposes."""

import json
import logging
import os
from typing import Any, Dict, List, Union

from bson import ObjectId

from whyhow_api.schemas.chunks import ChunkDocumentModel
from whyhow_api.schemas.graphs import GraphDocumentModel
from whyhow_api.schemas.nodes import NodeDocumentModel
from whyhow_api.schemas.schemas import SchemaDocumentModel
from whyhow_api.schemas.triples import TripleDocumentModel
from whyhow_api.schemas.workspaces import WorkspaceDocumentModel

logger = logging.getLogger(__name__)


class DemoDataLoader:
    """Initialises demo data from JSON files for testing environments and new users."""

    def __init__(self, user_id: ObjectId):
        self.user_id = user_id
        self.base_path = os.path.dirname(__file__)
        self.resources = self.load_resources()
        self.data = self.initialize_data_models()

    def load_resources(
        self,
    ) -> Dict[str, Union[Dict[str, Any], List[Dict[str, Any]]]]:
        """Load JSON files containing the demo data."""
        resource_files: dict[str, str] = {
            "workspace": "workspace.json",
            "chunks": "chunks.json",
            "schema": "schema.json",
            "graph": "graph.json",
            "nodes": "nodes.json",
            "triples": "triples.json",
        }
        resources: Dict[str, Union[Dict[str, Any], List[Dict[str, Any]]]] = {}
        for key, filename in resource_files.items():
            with open(os.path.join(self.base_path, filename)) as file:
                resources[key] = json.load(file)
        return resources

    def initialize_data_models(self) -> dict[str, Any]:
        """Process JSON data into database models."""
        workspace = self.resources["workspace"]
        schema = self.resources["schema"]
        self.workspace_id = ObjectId()
        self.schema_id = ObjectId()

        if not isinstance(workspace, dict):
            raise TypeError("Workspace data must be dictionaries.")
        if not isinstance(schema, dict):
            raise TypeError("Schema data must be dictionaries.")

        data: dict[str, Any] = {
            "workspace": WorkspaceDocumentModel(
                id=self.workspace_id,
                created_by=self.user_id,
                name=workspace["name"],
            ),
            "chunks": self.process_chunks(),
            "schema": SchemaDocumentModel(
                id=self.schema_id,
                created_by=self.user_id,
                workspace=self.workspace_id,
                name=schema["name"],
                entities=schema["entities"],
                relations=schema["relations"],
                patterns=schema["patterns"],
            ),
            "graph": None,
            "nodes": None,
            "triples": None,
        }

        data["graph"], data["nodes"], data["triples"] = self.process_graph()

        # Serialize the data for database insertion
        for key in data:
            if isinstance(data[key], list):
                data[key] = [
                    model.model_dump(by_alias=True) for model in data[key]
                ]
            else:
                data[key] = data[key].model_dump(by_alias=True)

        return data

    def process_chunks(self) -> list[ChunkDocumentModel]:
        """Create chunk models from resource data."""
        chunks = self.resources["chunks"]

        if not isinstance(chunks, list):
            raise TypeError("Chunk data must be a list of dictionaries.")

        chunks_out = []
        chunk_id_map = {}
        for chunk in chunks:
            old_id, new_id = chunk["_id"]["$oid"], ObjectId()
            chunk_id_map[old_id] = str(new_id)

            has_valid_tags = chunk.get("tags") is not None and isinstance(
                chunk["tags"], list
            )
            if not has_valid_tags:
                logger.warning(
                    f"Chunk {old_id} has invalid tags. Defaulting to empty list."
                )
            has_valid_user_metadata = chunk.get(
                "user_metadata"
            ) is not None and isinstance(chunk["user_metadata"], dict)
            if not has_valid_user_metadata:
                logger.warning(
                    f"Chunk {old_id} has invalid user metadata. Defaulting to empty dictionary."
                )
            chunks_out.append(
                ChunkDocumentModel(
                    id=new_id,
                    created_by=self.user_id,
                    workspaces=[self.workspace_id],
                    data_type=chunk["data_type"],
                    content=chunk["content"],
                    embedding=chunk["embedding"],
                    metadata=chunk["metadata"],
                    tags={
                        str(self.workspace_id): (
                            chunk["tags"] if has_valid_tags else []
                        )
                    },
                    user_metadata={
                        str(self.user_id): (
                            chunk["user_metadata"]
                            if has_valid_user_metadata
                            else {}
                        )
                    },
                )
            )
        self.chunk_id_map = chunk_id_map
        return chunks_out

    def process_graph(
        self,
    ) -> tuple[
        GraphDocumentModel, list[NodeDocumentModel], list[TripleDocumentModel]
    ]:
        """Create graph, nodes, and triples models from resource data."""
        graph = self.resources["graph"]

        if not isinstance(graph, dict):
            raise TypeError("Graph data must be a dictionary.")

        graph_out = GraphDocumentModel(
            id=ObjectId(),
            created_by=self.user_id,
            workspace=self.workspace_id,
            schema_=self.schema_id,
            name=graph["name"],
            status="ready",
            public=False,
        )

        nodes = self.create_nodes(graph_id=ObjectId(graph_out.id))
        triples = self.create_triples(graph_id=ObjectId(graph_out.id))

        return graph_out, nodes, triples

    def create_nodes(self, graph_id: ObjectId) -> list[NodeDocumentModel]:
        """Generate node models based on the graph ID."""
        nodes = self.resources["nodes"]
        if not isinstance(nodes, list):
            raise TypeError("Node data must be a list of dictionaries.")

        nodes_out = []
        node_id_map = {}
        for node in nodes:
            old_id, new_id = node["_id"]["$oid"], ObjectId()
            node_id_map[old_id] = new_id
            nodes_out.append(
                NodeDocumentModel(
                    id=new_id,
                    created_by=self.user_id,
                    graph=graph_id,
                    name=node["name"],
                    type=node["type"],
                    properties=node["properties"],
                    chunks=[
                        ObjectId(self.chunk_id_map[chunk["$oid"]])
                        for chunk in node["chunks"]
                        if chunk["$oid"] in self.chunk_id_map
                    ],
                )
            )
        self.node_id_map = node_id_map
        return nodes_out

    def create_triples(self, graph_id: ObjectId) -> list[TripleDocumentModel]:
        """Generate triple models based on the graph ID and nodes."""
        triples = self.resources["triples"]
        if not isinstance(triples, list):
            raise TypeError("Triple data must be a list of dictionaries.")

        triples_out = []
        for triple in triples:
            triples_out.append(
                TripleDocumentModel(
                    id=ObjectId(),
                    created_by=self.user_id,
                    graph=graph_id,
                    head_node=self.node_id_map[triple["head_node"]["$oid"]],
                    tail_node=self.node_id_map[triple["tail_node"]["$oid"]],
                    type=triple["type"],
                    properties=triple["properties"],
                    chunks=[
                        ObjectId(self.chunk_id_map[chunk["$oid"]])
                        for chunk in triple["chunks"]
                        if chunk["$oid"] in self.chunk_id_map
                    ],
                    embedding=triple["embedding"],
                )
            )
        return triples_out


if __name__ == "__main__":

    import cProfile
    import io
    import pstats

    def profile_demo_data_loader(user_id: ObjectId) -> None:
        """Profile the DemoDataLoader class."""
        pr = cProfile.Profile()
        pr.enable()
        demo_loader = DemoDataLoader(user_id)
        demo_loader.data  # Assuming you need to access .data to trigger loading
        pr.disable()
        s = io.StringIO()
        sortby = "cumulative"
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())

    profile_demo_data_loader(ObjectId())
