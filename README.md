# Description

Welcome to the WhyHow Knowledge Graph Studio! This platform makes it easy to create and manage RAG-native knowledge graphs and offers features like rule-based entity resolution, modular graph construction, flexible data ingestion, and an API-first design with a supporting [SDK](https://pypi.org/project/whyhow/). Whether you’re working with structured or unstructured data, building exploratory graphs or highly schema-constrained graphs, this platform is built for scalability and flexibility, enabling you to build dynamic graph-enabled AI workflows, ideal for both experimentation and large-scale use.

This platform is built on top of a NoSQL database. NoSQL data stores like MongoDB are a powerful choice for building knowledge graphs, offering a flexible, scalable storage layer that enable fast data retrieval, easy traversal of complex relationships, and a familiar interface for developers.

We are aiming to be database agnostic and also working with a number of other partners to bring similar capabilities to other relational and graph databases.

Check out our case studies and articles on the [WhyHow Blog](https://medium.com/enterprise-rag). (Here's a link to the [open sourcing announcement](https://medium.com/enterprise-rag/open-sourcing-the-whyhow-knowledge-graph-studio-powered-by-nosql-edce283fb341).)

![demo](https://github.com/user-attachments/assets/57ecd4dd-6044-485c-ab9b-0582171f6b7f)

*Check out the graph [here](https://app.whyhow.ai/public/graph/673ba7d0aa25224ee88c2406). We biult this demo using an [Amazon 10k](https://d18rn0p25nwr6d.cloudfront.net/CIK-0001018724/c7c14359-36fa-40c3-b3ca-5bf7f3fa0b96.pdf).*


# Installation

To install the package you can first clone the repo

_This client requires Python version 3.10 or higher._

```shell
$ git clone git@github.com:whyhow-ai/kg-engine.git
$ cd kg-engine
$ pip install .
```

If you are a developer you probably want to use an editable install. Additionally,
you need to install development and documentation dependencies.

```shell
$ pip install -e .[dev,docs]
```

# Launching the API

```shell
$ uvicorn src.whyhow_api.main:app
```

Note that there is a utility script `whyhow-locate` that will generate
the full path.

```shell
$ uvicorn $(whyhow-locate)
```

You can then navigate to `http://localhost:8000/docs` to see the Swagger UI.

# Docker

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
