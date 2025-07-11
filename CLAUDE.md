# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a RAG (Retrieval-Augmented Generation) application built with Python using LangChain and Ollama. It creates a pizza restaurant review chatbot that uses vector search to find relevant reviews and generates responses based on them.

## Architecture

The application follows a clean, modular architecture:

- **config.py**: Immutable configuration management using frozen dataclasses, loads from environment variables with CLI overrides
- **vector_store.py**: Handles Chroma vector database operations with Ollama embeddings for persistent storage
- **main.py**: Entry point that orchestrates the LLM chain (prompt template + Ollama model) and interactive chat loop
- **knowledge/reviews.csv**: Data source containing pizza restaurant reviews (Title, Date, Rating, Review columns)

### Key Components

1. **Configuration System**: Uses immutable dataclasses with environment variable loading and command-line overrides
2. **Vector Store**: Chroma database with persistent storage, automatically populated from CSV data on first run
3. **LLM Chain**: LangChain pipeline combining retrieval and generation with Ollama models
4. **Interactive Loop**: Command-line interface for querying the restaurant reviews

## Development Commands

### Prerequisites
- Install Ollama and required models:
  ```bash
  ollama pull llama3.2:latest
  ollama pull mxbai-embed-large:latest
  ```

### Environment Setup
```bash
# Initialize project and virtual environment
uv init
uv venv
source .venv/bin/activate

# Install dependencies
uv sync

# Set up environment configuration
cp env.example .env  # if env.example exists
# Edit .env with your settings
```

### Running the Application

**Development mode:**
```bash
uv run python src/rag_examples/main.py
```

**As installed package:**
```bash
uv pip install -e .
uv run rag-examples
```

**With CLI overrides:**
```bash
uv run python src/rag_examples/main.py --model llama3.2 --temperature 0.5 --max-tokens 2000
uv run python src/rag_examples/main.py --store-location ./custom_store --n-reviews 10
```

### Testing

**Running Tests:**
```bash
# Install test dependencies
uv sync --extra test

# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src/rag_examples

# Run only unit tests
uv run pytest -m unit

# Run only integration tests
uv run pytest -m integration

# Run tests excluding slow tests
uv run pytest -m "not slow"

# Run specific test file
uv run pytest tests/rag_examples/test_config.py

# Run with verbose output
uv run pytest -v
```

**Test Structure:**
- `tests/` - Test directory (mirrors src structure)
- `tests/conftest.py` - Test configuration and fixtures
- `tests/rag_examples/test_config.py` - Configuration module tests
- `tests/rag_examples/test_vector_store.py` - Vector store tests
- `tests/rag_examples/test_main.py` - Main module tests

**Test Coverage:**
- Unit tests for all modules with 80% minimum coverage requirement
- Integration tests for end-to-end workflows
- Mocked external dependencies (Ollama, Chroma)
- Test fixtures for sample data and configurations

## Configuration

The application uses environment variables with sensible defaults:

- `VECTOR_DB_LOCATION`: Chroma database location (default: `./chroma_store_db`)
- `VECTOR_K_VALUE`: Number of reviews to retrieve (default: `5`)
- `OLLAMA_EMBEDDING_MODEL`: Embedding model (default: `mxbai-embed-large`)
- `OLLAMA_LLM_MODEL`: LLM model (default: `llama3.2`)
- `TEMPERATURE`: Model temperature (default: `0.1`)
- `MAX_TOKENS`: Maximum response tokens (default: `1000`)

See CONFIG.md for detailed configuration options and CLI overrides.

## Data Source

The application expects `knowledge/reviews.csv` with columns: Title, Date, Rating, Review. The vector store automatically processes this data on first run and persists it for subsequent runs.

## Entry Points

- Script execution: `python src/rag_examples/main.py`
- Package entry point: `rag-examples` (defined in pyproject.toml)

## Dependencies

Core dependencies include:
- langchain==0.3.26
- langchain-ollama (Ollama integration)
- langchain-chroma (vector store)
- pandas (CSV processing)
- dotenv (environment variables)