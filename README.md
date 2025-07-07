# rag-examples - Ollama Based solution

##

## How to run?

### Where's the engine?

Ollama allows you to run LLMs on your local machine. In this usecase, we avoid any dependencies from outside out machine. This will require two models available for the operation.
1. General LLM functioning as a chat bot (Let's start with `llama3.2`)
1. Embeddinbg model, assisting with ingesting the reviews and keep them in the vector store (`mxbai-embed-large`).

The assumption is that you run `ollama` on your machine. If not, or you need some help with setting it up, use [Ollama website](https://ollama.com/).

For this example we'd need the following
```bash
$ ollama pull llama3.2:latest
$ ollama pull mxbai-embed-large:latest
```

### Running the application

#### DEV

While developing, you will need to use the full path to start the application:

```bash
$ uv run python src/rag_examples/main.py
```

#### Using as application
You can use `uv` to package and install rag_examples in your local python reepository by pip the project:
```bash
$ uv pip install -e .
```
After that, you'll be able to run the installed package
```bash
$ uv run rag-examples
```

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
