# rag-examples (Claude/Anthropic Edition)

## 1. What does it showcase?

This project demonstrates a Retrieval-Augmented Generation (RAG) pipeline using a local vector store and an external Large Language Model (LLM) provider (Anthropic Claude). It showcases:
- How to ingest and embed domain-specific data (e.g., reviews) for retrieval.
- How to use a vector store to fetch relevant context for user queries.
- How to integrate with Anthropic's Claude API for advanced LLM completions.
- Immutable, functional configuration and argument parsing.

## 2. How to operate it

### Prerequisites
- Python 3.8+
- [uv](https://github.com/astral-sh/uv) for dependency management and running
- An Anthropic API key (get one from https://console.anthropic.com/)

For more details about use of uv, try [use_uv](./USE_UV.md).

### Setup

1. **Clone the repository**
2. **Install dependencies and build the application:**
   ```bash
   uv pip install -e .
   ```
3. **Set your Anthropic API key:**
   - Create a `.env` file or export the variable in your shell:
     ```bash
     export ANTHROPIC_API_KEY=sk-ant-...your-key...
     ```
   - (Optional) Set the Claude model:
     ```bash
     export CLAUDE_MODEL=claude-3-opus-20240229
     ```

### Running the Application

- **Development mode:**
  ```bash
  uv run python src/rag_examples/main.py
  ```
- **Installed CLI mode:**
  ```bash
  uv run rag-examples
  ```

### Command-line Arguments

You can override configuration using CLI arguments:

| Argument              | Description                                      | Default                      |
|-----------------------|--------------------------------------------------|------------------------------|
| --model               | Claude model to use                              | $CLAUDE_MODEL                |
| --emb-model           | Embedding model for vector store                 | mxbai-embed-large            |
| --temperature         | LLM temperature                                  | 0.1                          |
| --max-tokens          | Maximum tokens for LLM response                  | 1000                         |
| --store-location      | Path to vector store database                    | ./chroma_store_db            |
| --n-reviews           | Number of reviews to retrieve                    | 5                            |
| --claude-model        | Claude model to use (overrides --model)          | $CLAUDE_MODEL                |
| --anthropic-api-key   | Anthropic API key (overrides env variable)       | $ANTHROPIC_API_KEY           |

### Example: Running with Arguments

```bash
uv run rag-examples \
  --claude-model claude-3-opus-20240229 \
  --anthropic-api-key sk-ant-...your-key... \
  --emb-model mxbai-embed-large \
  --temperature 0.2 \
  --max-tokens 800 \
  --store-location ./my_store \
  --n-reviews 3
```

---
For more details, see the main README or source code.
