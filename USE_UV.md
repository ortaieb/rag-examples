# Using uv with rag-examples

This guide explains how to use [uv](https://github.com/astral-sh/uv) for managing, building, and running the rag-examples project.

---

## 1. Project Initialization

If you are starting a new project or want to initialize uv in this directory:

```bash
uv init
```

(You can also specify a directory: `uv init <my-directory>`)

---

## 2. Creating a Virtual Environment

To create a virtual environment for isolating dependencies:

```bash
uv venv
```

Activate the environment:

```bash
source .venv/bin/activate
```

---

## 3. Installing Dependencies

Dependencies are managed in `pyproject.toml`. To install them:

```bash
uv sync
```

Or, to install in editable mode (for development):

```bash
uv pip install -e .
```

---

## 4. Running the Application

### Development Mode

Run the main script directly:

```bash
uv run python src/rag_examples/main.py
```

### Installed CLI Mode

After installing with pip, you can run the CLI entry point:

```bash
uv run rag-examples
```

---

For more details, see the main README or source code.
