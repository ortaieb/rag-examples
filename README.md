# rag-examples - Ollama Based solution

##

## How to run?


### Setting up

<Assumption> we'll be using uv to manage the project

#### Initialise the project
If you already in the project directory the following will be suffecient:

```bash
$ uv init
```

(With uv, all you need is to specificy the directory, if you wish to create it: `uv init <my-directory`>)

#### Create a virtual environment

A virtual environment will allow you to isolate dependencies from your general use Python

```bash
$ uv venv
```

To use the venv
```bash
$ source .venv/bin/activate
```

#### Setting up dependencies
Project dependencies are managed inside `pyproject.toml` descriptor.
To install the dependencies, use `sync` command:

```bash
$ uv sync
```
