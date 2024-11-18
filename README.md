# Description

WhyHow API (backend)

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
