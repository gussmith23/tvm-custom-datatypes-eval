This is the evaluation repo for the TVM Bring Your Own Datatypes project.

# Installation
Build the Dockerfile:
```bash
docker build . -t datatypes-eval
```

Run the resulting Docker image:
```bash
docker run datatypes-eval
```

# What's What?
- `datatypes`: contains the datatype implementations themselves, and any scripts needed to build the datatype implementations.
- `tests`: contains datatype tests, each of which is a Python script. Additionally contains some utilities needed by the tests.
