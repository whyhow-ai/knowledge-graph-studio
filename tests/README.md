- [Description](#description)
- [Unit tests](#unit-tests)
  - [Unit tests coverage](#unit-tests-coverage)
  - [Unit tests guide](#unit-tests-guide)
  - [Unit tests coverage](#unit-tests-coverage)
  - [Unit tests debug](#unit-tests-debug)

# Description

WhyHow tests are based on [pytest](https://docs.pytest.org/en/stable/)

Unit tests are stored in **_tests/unit_** folder

Each type of tests has it own **_conftest.py_** file which provides fixtures for an entire directory. Fixtures defined in a conftest.py can be used by any test in that package without needing to import them (pytest will automatically discover them).

Use conftest.py to avoid code duplication and manage the scope of your fixtures! See [reference](<(https://docs.pytest.org/en/stable/reference/fixtures.html#conftest-py-sharing-fixtures-across-multiple-files)>).

# Unit tests

To run all unit tests:

```shell
pytest tests/unit
```

Run only one test (for debug purposes):

```shell
pytest -k "test_get_graph_successful"
```

Run whole suite, but fail quick on the first error:

```shell
pytest tests/unit -x --ff
```

### Unit tests debug

Use breakpoint() and (Pdb) to debug your code. [Documentation](https://docs.python.org/3/library/pdb.html)

## Unit tests coverage

Unit test coverage could be found in **_htmlcov_** folder **_index.html_** file.
