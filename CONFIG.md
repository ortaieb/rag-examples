# Configuration

This project uses environment variables for configuration. Create a `.env` file in the root directory with the following variables:

## Quick Setup

1. Copy the example environment file:
   ```bash
   cp env.example .env
   ```

2. Modify the `.env` file with your desired settings

3. Run the application:
   ```bash
   uv run python src/rag_examples/main.py
   ```

## Vector Store Configuration

```bash
# Database location for Chroma vector store
VECTOR_DB_LOCATION=./chroma_store_db

# Number of documents to retrieve (k parameter)
VECTOR_K_VALUE=5
```

## Ollama Configuration

```bash
# Embedding model for vector operations
OLLAMA_EMBEDDING_MODEL=mxbai-embed-large

# LLM model for chat operations
OLLAMA_LLM_MODEL=llama3.2
```

## LLM Configuration

```bash
# Model temperature (0.0 to 1.0)
TEMPERATURE=0.1

# Maximum tokens for responses
MAX_TOKENS=1000
```

### Configuration Options

#### Vector Store
- `VECTOR_DB_LOCATION`: Path where the Chroma vector database will be stored (default: `./chroma_store_db`)
- `VECTOR_K_VALUE`: Number of documents to retrieve when searching the vector store (default: `5`)

#### Ollama Models
- `OLLAMA_EMBEDDING_MODEL`: Ollama model to use for embeddings (default: `mxbai-embed-large`)
- `OLLAMA_LLM_MODEL`: Ollama model to use for chat/LLM operations (default: `llama3.2`)

#### LLM Settings
- `TEMPERATURE`: Controls randomness in model responses, 0.0 = deterministic, 1.0 = very random (default: `0.1`)
- `MAX_TOKENS`: Maximum number of tokens in the model's response (default: `1000`)

### Example .env file

```bash
# Copy this to .env and modify as needed
VECTOR_DB_LOCATION=./my_custom_chroma_store
VECTOR_K_VALUE=10
OLLAMA_EMBEDDING_MODEL=mxbai-embed-large
OLLAMA_LLM_MODEL=llama3.2
TEMPERATURE=0.1
MAX_TOKENS=1000
```

### Command Line Overrides

You can override configuration values using command line arguments:

```bash
# Override model and temperature
uv run python src/rag_examples/main.py --model llama3.2 --temperature 0.5

# Override max tokens
uv run python src/rag_examples/main.py --max-tokens 2000

# Override vector store settings
uv run python src/rag_examples/main.py --store-location ./chroma1 --n-reviews 10

# Override embedding model
uv run python src/rag_examples/main.py --emb-model nomic-embed-text
```

### Architecture

The configuration is managed through the `config.py` module which:
- Uses immutable dataclasses for type safety
- Loads environment variables with sensible defaults
- Provides a single source of truth for all configuration
- Follows functional programming principles with frozen dataclasses
- Supports command line overrides while maintaining immutable base configuration
