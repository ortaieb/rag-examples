"""
Configuration module for rag-examples.
Loads environment variables and provides immutable configuration values.
"""
from dataclasses import dataclass
from dotenv import load_dotenv
import os
import argparse
from typing import Final

# Load environment variables
load_dotenv()


@dataclass(frozen=True)
class VectorStoreConfig:
    """Immutable configuration for vector store settings."""
    embedding_model: str
    db_location: str
    k_value: int

@dataclass(frozen=True)
class LLMConfig:
    """Immutable configuration for LLM settings."""
    temperature: float
    max_tokens: int

@dataclass(frozen=True)
class ClaudeConfig:
    """Immutable configuration for Claude (Anthropic) settings."""
    model: str
    api_key: str

@dataclass(frozen=True)
class AppConfig:
    """Immutable application configuration."""
    vector_store: VectorStoreConfig
    llm: LLMConfig
    claude: ClaudeConfig


def load_config() -> AppConfig:
    """
    Load configuration from environment variables with defaults.

    Returns:
        AppConfig: Immutable configuration object
    """
    vector_config = VectorStoreConfig(
        embedding_model=os.getenv("VECTOR_EMBEDDING_MODEL", "mxbai-embed-large"),
        db_location=os.getenv("VECTOR_DB_LOCATION", "./chroma_store_db"),
        k_value=int(os.getenv("VECTOR_K_VALUE", "5"))
    )

    llm_config = LLMConfig(
        temperature=float(os.getenv("TEMPERATURE", "0.1")),
        max_tokens=int(os.getenv("MAX_TOKENS", "1000"))
    )

    claude_config = ClaudeConfig(
        model=os.getenv("CLAUDE_MODEL", "claude-3-opus-20240229"),
        api_key=os.getenv("ANTHROPIC_API_KEY", "")
    )

    return AppConfig(
        vector_store=vector_config,
        llm=llm_config,
        claude=claude_config
    )


def get_cli_overrides() -> argparse.Namespace:
    """
    Get command line overrides for configuration.

    Returns:
        argparse.Namespace: Command line arguments with CONFIG defaults
    """
    parser = argparse.ArgumentParser(description="RAG Examples CLI")
    parser.add_argument('--model',
                        default=CONFIG.claude.model,
                        help=f'Model to use (default: {CONFIG.claude.model})')

    parser.add_argument('--anthropic-api-key',
                        default=CONFIG.claude.api_key,
                        help=f'Model provider api key (default: {CONFIG.claude.api_key})')

    parser.add_argument('--temperature',
                       type=float,
                       default=CONFIG.llm.temperature,
                       help=f'Model temperature (default: {CONFIG.llm.temperature})')

    parser.add_argument('--emb-model',
                        default=CONFIG.vector_store.embedding_model,
                        help=f'Embedding modek model to use (default: {CONFIG.vector_store.embedding_model})')

    parser.add_argument('--max-tokens',
                       type=int,
                       default=CONFIG.llm.max_tokens,
                       help=f'Maximum tokens (default: {CONFIG.llm.max_tokens})')

    parser.add_argument('--store-location',
                        default = CONFIG.vector_store.db_location,
                        help=f"Location of the vector store persistent copy (default: {CONFIG.vector_store.db_location})")

    parser.add_argument('--n-reviews',
                        default = CONFIG.vector_store.k_value,
                        help=f"number of reviews provided to the model (default: {CONFIG.vector_store.k_value})"
    )

    parsed_args = parser.parse_args()

    if not parsed_args.anthropic_api_key.startswith("sk-ant-"):
        err_msg = f"api_key was not provided: [found: {parsed_args.anthropic_api_key}]"
        raise RuntimeError(err_msg)

    return parsed_args


# Global configuration instance
CONFIG: Final[AppConfig] = load_config()
