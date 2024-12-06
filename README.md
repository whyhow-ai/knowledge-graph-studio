# Description

Welcome to the WhyHow Knowledge Graph Studio! This platform makes it easy to create and manage RAG-native knowledge graphs and offers features like rule-based entity resolution, modular graph construction, flexible data ingestion, and an API-first design with a supporting [SDK](https://pypi.org/project/whyhow/) (_check out our [code examples](https://whyhow-ai.github.io/whyhow-sdk-docs/examples/overview/)_). Whether you’re working with structured or unstructured data, building exploratory graphs or highly schema-constrained graphs, this platform is built for scalability and flexibility, enabling you to build dynamic graph-enabled AI workflows, ideal for both experimentation and large-scale use.

This platform is built on top of a NoSQL database. NoSQL data stores like MongoDB are a powerful choice for building knowledge graphs, offering a flexible, scalable storage layer that enable fast data retrieval, easy traversal of complex relationships, and a familiar interface for developers.

We are aiming to be database agnostic and also working with a number of other partners to bring similar capabilities to other relational and graph databases.

Check out our case studies and articles on the [WhyHow Blog](https://medium.com/enterprise-rag). (_Here's a link to the [open sourcing announcement](https://medium.com/enterprise-rag/open-sourcing-the-whyhow-knowledge-graph-studio-powered-by-nosql-edce283fb341)._)

We also have a parallel open-source triple creation tool, [Knowledge Table](https://github.com/whyhow-ai/knowledge-table) to check out

![os-dashboard](https://github.com/user-attachments/assets/07d7926f-547f-41b1-a9e7-e9ec31590478)

![create_graph1](https://github.com/user-attachments/assets/0471338b-3045-4f6b-90a3-51370fd80372)

**Demo**

https://github.com/user-attachments/assets/8e98626d-c531-4d6a-a9bd-c1c7ef16667a

_Check out the graph [here](https://app.whyhow.ai/public/graph/673ba7d0aa25224ee88c2406). We built this demo using an [Amazon 10k](https://d18rn0p25nwr6d.cloudfront.net/CIK-0001018724/c7c14359-36fa-40c3-b3ca-5bf7f3fa0b96.pdf)._

### Installation

To install the package you can first clone the repo

_This client requires Python version 3.10 or higher._

```shell
$ git clone git@github.com:whyhow-ai/knowledge-graph-studio.git
$ cd knowledge-graph-studio
$ pip install .
```

If you are a developer you probably want to use an editable install. Additionally,
you need to install development and documentation dependencies.

```shell
$ pip install -e .[dev,docs]
```

# Quickstart

### 1. Pre-requisites

In order to get started with the WhyHow API with this quickstart, you will need the following:

- OpenAI API key
- MongoDB account
  - _You must create a project and cluster in MongoDB Atlas_

### 2. Configuration

**Environment Variables**
Copy the `.env.example` file to `.env` and update the values per your environment. To get started with this version, you need to provide values for `mongodb`, `openai`.

```shell
$ cp .env.sample .env
```

To get started, you must configure, at minimum, the following enviroinment variables:

```shell
WHYHOW__EMBEDDING__OPENAI__API_KEY=<your openai api key>
WHYHOW__GENERATIVE__OPENAI__API_KEY=<your openai api key - can be the same>
WHYHOW__MONGODB__USERNAME=<your altas database username>
WHYHOW__MONGODB__PASSWORD=<your altas database password>
WHYHOW__MONGODB__DATABASE_NAME=main
WHYHOW__MONGODB__HOST=<your altas host i.e. 'xxx.xxx.mongodb.net'>
```

**Create Collections**

Once you have configured your environment variables, you must create the database, collections, and indexes in your Atlas cluster. To simplify this, we have included a cli script in `src/whyhow_api/cli/`. To set this up, run the following:

```shell
$ cd src/whyhow_api/cli/
$ python admin.py setup-collections --config-file collection_index_config.json
```

This script will create 11 collections: `chunk`, `document`, `graph`, `node`, `query`, `rule`, `schema`, `task`, `triple`, `user`, and `workspace`. To verify, browse your collections in your MongoDB Atlas browser, or MongoDB Compass.

**Create User**

> [!Important]
> Once you create the user below, **copy the API Key** as you will need this to communicate with the backend via the SDK.

Once the collections have been configured, you must create a user and an API key. We have included a cli script for this as well. To create the user, run the following from `src/whyhow_api/cli/`:

```shell
$ python admin.py create-user --email <your email address> --openai-key <your openai api key>

# User created with email: <email>, API Key: <whyhow api key>
```

Once the user creation completes successfully, you should see a message that includes your email address and WhyHow API key. You should copy this key and use this to configure the SDK.

### 3. Launching the API

One the configuration is complete, you can start the API server by running the following:

```shell
$ uvicorn src.whyhow_api.main:app
```

Note that there is a utility script `whyhow-locate` that will generate
the full path.

```shell
$ uvicorn $(whyhow-locate)
```

You can then navigate to `http://localhost:8000/docs` to see the Swagger UI.

### 4. Test Locally

**Install Python SDK**

```shell
$ pip install whyhow
```

**Configure and run**

> [!Important]
> Configure your WhyHow client using the API Key you created in **step 2**.


```shell
from whyhow import WhyHow, Triple, Node, Chunk, Relation

# Configure WhyHow client
client = WhyHow(api_key='<your whyhow api key>', base_url="http://localhost:8000")

# Create workspace
workspace = client.workspaces.create(name="Demo Workspace")

# Create chunk(s)
chunk = client.chunks.create(
    workspace_id=workspace.workspace_id,
    chunks=[Chunk(
        content="preneur and visionary, Sam Altman serves as the CEO of OpenAI, leading advancements in artifici"
    )]
)

# Create triple(s)
triples = [
    Triple(
        head=Node(
            name="Sam Altman",
            label="Person",
            properties={"title": "CEO"}
        ),
        relation=Relation(
            name="runs",
        ),
        tail=Node(
            name="OpenAI",
            label="Business",
            properties={"market cap": "$157 Billion"}
        ),
        chunk_ids=[c.chunk_id for c in chunk]
    )
]

# Create graph
graph = client.graphs.create_graph_from_triples(
    name="Demo Graph",
    workspace_id=workspace.workspace_id,
    triples=triples
)

# Query graph
query = client.graphs.query_unstructured(
    graph_id=graph.graph_id,
    query="Who runs OpenAI?"
)
```

# _Docker_

You can also run the server using Docker. Once you have completed steps 1 and 2 of the Quickstart, you can build and run the Knowlwedge Graph Studio backend using Docker. 

### Building the image

We assume that the image tag is `v1` (modify based on your needs)

```shell
$ docker build --platform=linux/amd64 -t kg_engine:v1 .
```

### Running the image

```shell
$ OUTSIDE_PORT=1234
$ docker run -it --rm -p $OUTSIDE_PORT:8000 kg_engine:v1
```
