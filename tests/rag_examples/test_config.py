"""
Unit tests for the configuration module.
"""
import os
import pytest
from unittest.mock import patch, MagicMock
import argparse

from rag_examples.config import (
    VectorStoreConfig,
    OllamaConfig,
    LLMConfig,
    AppConfig,
    load_config,
    get_cli_overrides
)


class TestVectorStoreConfig:
    """Test VectorStoreConfig dataclass."""
    
    @pytest.mark.unit
    def test_vector_store_config_creation(self):
        """Test creating VectorStoreConfig with valid parameters."""
        config = VectorStoreConfig(
            db_location="./test_db",
            k_value=10
        )
        
        assert config.db_location == "./test_db"
        assert config.k_value == 10
    
    @pytest.mark.unit
    def test_vector_store_config_immutable(self):
        """Test that VectorStoreConfig is immutable."""
        config = VectorStoreConfig(db_location="./test_db", k_value=5)
        
        with pytest.raises(AttributeError):
            config.db_location = "./new_db"
        
        with pytest.raises(AttributeError):
            config.k_value = 10


class TestOllamaConfig:
    """Test OllamaConfig dataclass."""
    
    @pytest.mark.unit
    def test_ollama_config_creation(self):
        """Test creating OllamaConfig with valid parameters."""
        config = OllamaConfig(
            embedding_model="test-embed",
            llm_model="test-llm"
        )
        
        assert config.embedding_model == "test-embed"
        assert config.llm_model == "test-llm"
    
    @pytest.mark.unit
    def test_ollama_config_immutable(self):
        """Test that OllamaConfig is immutable."""
        config = OllamaConfig(embedding_model="test-embed", llm_model="test-llm")
        
        with pytest.raises(AttributeError):
            config.embedding_model = "new-embed"


class TestLLMConfig:
    """Test LLMConfig dataclass."""
    
    @pytest.mark.unit
    def test_llm_config_creation(self):
        """Test creating LLMConfig with valid parameters."""
        config = LLMConfig(temperature=0.5, max_tokens=2000)
        
        assert config.temperature == 0.5
        assert config.max_tokens == 2000
    
    @pytest.mark.unit
    def test_llm_config_immutable(self):
        """Test that LLMConfig is immutable."""
        config = LLMConfig(temperature=0.5, max_tokens=2000)
        
        with pytest.raises(AttributeError):
            config.temperature = 0.8


class TestAppConfig:
    """Test AppConfig dataclass."""
    
    @pytest.mark.unit
    def test_app_config_creation(self):
        """Test creating AppConfig with all sub-configs."""
        vector_config = VectorStoreConfig(db_location="./test_db", k_value=5)
        ollama_config = OllamaConfig(embedding_model="test-embed", llm_model="test-llm")
        llm_config = LLMConfig(temperature=0.1, max_tokens=1000)
        
        app_config = AppConfig(
            vector_store=vector_config,
            ollama=ollama_config,
            llm=llm_config
        )
        
        assert app_config.vector_store == vector_config
        assert app_config.ollama == ollama_config
        assert app_config.llm == llm_config


class TestLoadConfig:
    """Test load_config function."""
    
    @pytest.mark.unit
    def test_load_config_with_defaults(self):
        """Test loading config with default values."""
        with patch.dict(os.environ, {}, clear=True):
            config = load_config()
            
            assert config.vector_store.db_location == "./chroma_store_db"
            assert config.vector_store.k_value == 5
            assert config.ollama.embedding_model == "mxbai-embed-large"
            assert config.ollama.llm_model == "llama3.2"
            assert config.llm.temperature == 0.1
            assert config.llm.max_tokens == 1000
    
    @pytest.mark.unit
    def test_load_config_with_env_vars(self, mock_env_vars):
        """Test loading config with environment variables."""
        config = load_config()
        
        assert config.vector_store.db_location == "./test_chroma_db"
        assert config.vector_store.k_value == 3
        assert config.ollama.embedding_model == "test-embed-model"
        assert config.ollama.llm_model == "test-llm-model"
        assert config.llm.temperature == 0.5
        assert config.llm.max_tokens == 500
    
    @pytest.mark.unit
    def test_load_config_type_conversion(self):
        """Test that environment variables are properly converted to correct types."""
        with patch.dict(os.environ, {
            'VECTOR_K_VALUE': '10',
            'TEMPERATURE': '0.8',
            'MAX_TOKENS': '2000'
        }):
            config = load_config()
            
            assert isinstance(config.vector_store.k_value, int)
            assert config.vector_store.k_value == 10
            assert isinstance(config.llm.temperature, float)
            assert config.llm.temperature == 0.8
            assert isinstance(config.llm.max_tokens, int)
            assert config.llm.max_tokens == 2000


class TestGetCliOverrides:
    """Test get_cli_overrides function."""
    
    @pytest.mark.unit
    def test_get_cli_overrides_defaults(self):
        """Test CLI overrides with default values."""
        with patch('sys.argv', ['script.py']):
            with patch('rag_examples.config.CONFIG') as mock_config:
                mock_config.ollama.llm_model = "llama3.2"
                mock_config.ollama.embedding_model = "mxbai-embed-large"
                mock_config.llm.temperature = 0.1
                mock_config.llm.max_tokens = 1000
                mock_config.vector_store.db_location = "./chroma_store_db"
                mock_config.vector_store.k_value = 5
                
                args = get_cli_overrides()
                
                assert args.model == "llama3.2"
                assert args.emb_model == "mxbai-embed-large"
                assert args.temperature == 0.1
                assert args.max_tokens == 1000
                assert args.store_location == "./chroma_store_db"
                assert args.n_reviews == 5
    
    @pytest.mark.unit
    def test_get_cli_overrides_with_args(self):
        """Test CLI overrides with provided arguments."""
        test_args = [
            'script.py',
            '--model', 'custom-llm',
            '--emb-model', 'custom-embed',
            '--temperature', '0.7',
            '--max-tokens', '1500',
            '--store-location', './custom_store',
            '--n-reviews', '8'
        ]
        
        with patch('sys.argv', test_args):
            with patch('rag_examples.config.CONFIG') as mock_config:
                mock_config.ollama.llm_model = "llama3.2"
                mock_config.ollama.embedding_model = "mxbai-embed-large"
                mock_config.llm.temperature = 0.1
                mock_config.llm.max_tokens = 1000
                mock_config.vector_store.db_location = "./chroma_store_db"
                mock_config.vector_store.k_value = 5
                
                args = get_cli_overrides()
                
                assert args.model == "custom-llm"
                assert args.emb_model == "custom-embed"
                assert args.temperature == 0.7
                assert args.max_tokens == 1500
                assert args.store_location == "./custom_store"
                assert args.n_reviews == "8"  # Note: n_reviews is not converted to int by argparse
    
    @pytest.mark.unit
    def test_get_cli_overrides_type_conversion(self):
        """Test that CLI arguments are properly type converted."""
        test_args = [
            'script.py',
            '--temperature', '0.9',
            '--max-tokens', '2500'
        ]
        
        with patch('sys.argv', test_args):
            with patch('rag_examples.config.CONFIG') as mock_config:
                mock_config.ollama.llm_model = "llama3.2"
                mock_config.ollama.embedding_model = "mxbai-embed-large"
                mock_config.llm.temperature = 0.1
                mock_config.llm.max_tokens = 1000
                mock_config.vector_store.db_location = "./chroma_store_db"
                mock_config.vector_store.k_value = 5
                
                args = get_cli_overrides()
                
                assert isinstance(args.temperature, float)
                assert args.temperature == 0.9
                assert isinstance(args.max_tokens, int)
                assert args.max_tokens == 2500