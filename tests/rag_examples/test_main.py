"""
Unit tests for the main module.
"""
import pytest
from unittest.mock import patch, Mock, MagicMock, call
from io import StringIO
import sys

from rag_examples.main import create_llm_chain, main


class TestCreateLLMChain:
    """Test create_llm_chain function."""
    
    @pytest.mark.unit
    @patch('rag_examples.main.OllamaLLM')
    @patch('rag_examples.main.ChatPromptTemplate')
    def test_create_llm_chain_basic(self, mock_prompt_template, mock_ollama_llm):
        """Test basic LLM chain creation."""
        mock_llm_instance = Mock()
        mock_ollama_llm.return_value = mock_llm_instance
        
        mock_prompt_instance = Mock()
        mock_prompt_template.from_template.return_value = mock_prompt_instance
        
        model, prompt = create_llm_chain("test-model", 0.5)
        
        # Verify LLM was created with correct parameters
        mock_ollama_llm.assert_called_once_with(
            model="test-model",
            temperature=0.5
        )
        
        # Verify prompt template was created
        mock_prompt_template.from_template.assert_called_once()
        template_arg = mock_prompt_template.from_template.call_args[0][0]
        assert "pizza restaurant" in template_arg
        assert "{reviews}" in template_arg
        assert "{question}" in template_arg
        
        assert model == mock_llm_instance
        assert prompt == mock_prompt_instance
    
    @pytest.mark.unit
    @patch('rag_examples.main.OllamaLLM')
    @patch('rag_examples.main.ChatPromptTemplate')
    def test_create_llm_chain_different_parameters(self, mock_prompt_template, mock_ollama_llm):
        """Test LLM chain creation with different parameters."""
        mock_llm_instance = Mock()
        mock_ollama_llm.return_value = mock_llm_instance
        
        mock_prompt_instance = Mock()
        mock_prompt_template.from_template.return_value = mock_prompt_instance
        
        model, prompt = create_llm_chain("llama3.2", 0.8)
        
        mock_ollama_llm.assert_called_once_with(
            model="llama3.2",
            temperature=0.8
        )
        
        assert model == mock_llm_instance
        assert prompt == mock_prompt_instance
    
    @pytest.mark.unit
    @patch('rag_examples.main.OllamaLLM')
    @patch('rag_examples.main.ChatPromptTemplate')
    def test_create_llm_chain_prompt_template_content(self, mock_prompt_template, mock_ollama_llm):
        """Test that prompt template contains expected content."""
        mock_ollama_llm.return_value = Mock()
        mock_prompt_template.from_template.return_value = Mock()
        
        create_llm_chain("test-model", 0.1)
        
        # Verify the template content
        template_arg = mock_prompt_template.from_template.call_args[0][0]
        assert "exeprt in answering questions about a pizza restaurant" in template_arg
        assert "Base answers only on reviews provided" in template_arg
        assert "Here are some relevant reviews: {reviews}" in template_arg
        assert "Here is the question to answer: {question}" in template_arg


class TestMain:
    """Test main function."""
    
    @pytest.mark.unit
    @patch('rag_examples.main.get_cli_overrides')
    @patch('rag_examples.main.create_vector_store')
    @patch('rag_examples.main.create_llm_chain')
    @patch('builtins.input')
    @patch('builtins.print')
    def test_main_single_question_and_quit(self, mock_print, mock_input, mock_create_llm_chain, mock_create_vector_store, mock_get_cli_overrides):
        """Test main function with single question and quit."""
        # Setup CLI config mock
        mock_cli_config = Mock()
        mock_cli_config.store_location = "./test_store"
        mock_cli_config.emb_model = "test-embed"
        mock_cli_config.n_reviews = 5
        mock_cli_config.model = "test-llm"
        mock_cli_config.temperature = 0.1
        mock_get_cli_overrides.return_value = mock_cli_config
        
        # Setup vector store mock
        mock_retriever = Mock()
        mock_retriever.invoke.return_value = ["Review 1", "Review 2"]
        mock_create_vector_store.return_value = mock_retriever
        
        # Setup LLM chain mock
        mock_model = Mock()
        mock_prompt = Mock()
        mock_chain = Mock()
        mock_chain.invoke.return_value = "This is the LLM response"
        mock_prompt.__or__ = Mock(return_value=mock_chain)
        mock_create_llm_chain.return_value = (mock_model, mock_prompt)
        
        # Setup input sequence: question, then quit
        mock_input.side_effect = ["What's the best pizza?", "q"]
        
        # Run main
        main()
        
        # Verify CLI config was loaded
        mock_get_cli_overrides.assert_called_once()
        
        # Verify vector store was created
        mock_create_vector_store.assert_called_once_with(
            db_location="./test_store",
            embedding_model="test-embed",
            k_value=5
        )
        
        # Verify LLM chain was created
        mock_create_llm_chain.assert_called_once_with("test-llm", 0.1)
        
        # Verify retriever was called
        mock_retriever.invoke.assert_called_once_with("What's the best pizza?")
        
        # Verify chain was invoked
        mock_chain.invoke.assert_called_once_with({
            "reviews": ["Review 1", "Review 2"],
            "question": "What's the best pizza?"
        })
        
        # Verify output was printed
        assert mock_print.called
    
    @pytest.mark.unit
    @patch('rag_examples.main.get_cli_overrides')
    @patch('rag_examples.main.create_vector_store')
    @patch('rag_examples.main.create_llm_chain')
    @patch('builtins.input')
    @patch('builtins.print')
    def test_main_multiple_questions(self, mock_print, mock_input, mock_create_llm_chain, mock_create_vector_store, mock_get_cli_overrides):
        """Test main function with multiple questions."""
        # Setup CLI config mock
        mock_cli_config = Mock()
        mock_cli_config.store_location = "./test_store"
        mock_cli_config.emb_model = "test-embed"
        mock_cli_config.n_reviews = 3
        mock_cli_config.model = "test-llm"
        mock_cli_config.temperature = 0.5
        mock_get_cli_overrides.return_value = mock_cli_config
        
        # Setup vector store mock
        mock_retriever = Mock()
        mock_retriever.invoke.side_effect = [
            ["Pizza review 1", "Pizza review 2"],
            ["Service review 1", "Service review 2"]
        ]
        mock_create_vector_store.return_value = mock_retriever
        
        # Setup LLM chain mock
        mock_model = Mock()
        mock_prompt = Mock()
        mock_chain = Mock()
        mock_chain.invoke.side_effect = ["Pizza answer", "Service answer"]
        mock_prompt.__or__ = Mock(return_value=mock_chain)
        mock_create_llm_chain.return_value = (mock_model, mock_prompt)
        
        # Setup input sequence: two questions, then quit
        mock_input.side_effect = ["How's the pizza?", "How's the service?", "q"]
        
        # Run main
        main()
        
        # Verify retriever was called twice
        assert mock_retriever.invoke.call_count == 2
        mock_retriever.invoke.assert_any_call("How's the pizza?")
        mock_retriever.invoke.assert_any_call("How's the service?")
        
        # Verify chain was invoked twice
        assert mock_chain.invoke.call_count == 2
        mock_chain.invoke.assert_any_call({
            "reviews": ["Pizza review 1", "Pizza review 2"],
            "question": "How's the pizza?"
        })
        mock_chain.invoke.assert_any_call({
            "reviews": ["Service review 1", "Service review 2"],
            "question": "How's the service?"
        })
    
    @pytest.mark.unit
    @patch('rag_examples.main.get_cli_overrides')
    @patch('rag_examples.main.create_vector_store')
    @patch('rag_examples.main.create_llm_chain')
    @patch('builtins.input')
    @patch('builtins.print')
    def test_main_immediate_quit(self, mock_print, mock_input, mock_create_llm_chain, mock_create_vector_store, mock_get_cli_overrides):
        """Test main function with immediate quit."""
        # Setup CLI config mock
        mock_cli_config = Mock()
        mock_cli_config.store_location = "./test_store"
        mock_cli_config.emb_model = "test-embed"
        mock_cli_config.n_reviews = 5
        mock_cli_config.model = "test-llm"
        mock_cli_config.temperature = 0.1
        mock_get_cli_overrides.return_value = mock_cli_config
        
        # Setup vector store mock
        mock_retriever = Mock()
        mock_create_vector_store.return_value = mock_retriever
        
        # Setup LLM chain mock
        mock_model = Mock()
        mock_prompt = Mock()
        mock_chain = Mock()
        mock_prompt.__or__ = Mock(return_value=mock_chain)
        mock_create_llm_chain.return_value = (mock_model, mock_prompt)
        
        # Setup input sequence: immediate quit
        mock_input.side_effect = ["q"]
        
        # Run main
        main()
        
        # Verify setup was done
        mock_get_cli_overrides.assert_called_once()
        mock_create_vector_store.assert_called_once()
        mock_create_llm_chain.assert_called_once()
        
        # Verify no retrieval or chain invocation happened
        mock_retriever.invoke.assert_not_called()
        mock_chain.invoke.assert_not_called()
    
    @pytest.mark.unit
    @patch('rag_examples.main.get_cli_overrides')
    @patch('rag_examples.main.create_vector_store')
    @patch('rag_examples.main.create_llm_chain')
    @patch('builtins.input')
    @patch('builtins.print')
    def test_main_config_display(self, mock_print, mock_input, mock_create_llm_chain, mock_create_vector_store, mock_get_cli_overrides):
        """Test that main function displays configuration information."""
        # Setup CLI config mock
        mock_cli_config = Mock()
        mock_cli_config.store_location = "./custom_store"
        mock_cli_config.emb_model = "custom-embed"
        mock_cli_config.n_reviews = 10
        mock_cli_config.model = "custom-llm"
        mock_cli_config.temperature = 0.8
        mock_get_cli_overrides.return_value = mock_cli_config
        
        # Setup mocks
        mock_retriever = Mock()
        mock_create_vector_store.return_value = mock_retriever
        mock_model = Mock()
        mock_prompt = Mock()
        mock_chain = Mock()
        mock_prompt.__or__ = Mock(return_value=mock_chain)
        mock_create_llm_chain.return_value = (mock_model, mock_prompt)
        
        # Setup input sequence: immediate quit
        mock_input.side_effect = ["q"]
        
        # Run main
        main()
        
        # Verify configuration was printed
        print_calls = [call.args[0] for call in mock_print.call_args_list]
        config_output = " ".join(print_calls)
        
        assert "Using model: custom-llm" in config_output
        assert "Temperature: 0.8" in config_output
        assert "Store location: ./custom_store" in config_output
        assert "Number of reviews: 10" in config_output
    
    @pytest.mark.unit
    @patch('rag_examples.main.get_cli_overrides')
    @patch('rag_examples.main.create_vector_store')
    @patch('rag_examples.main.create_llm_chain')
    @patch('builtins.input')
    @patch('builtins.print')
    def test_main_n_reviews_type_conversion(self, mock_print, mock_input, mock_create_llm_chain, mock_create_vector_store, mock_get_cli_overrides):
        """Test that n_reviews is properly converted to int."""
        # Setup CLI config mock with string n_reviews (as would come from argparse)
        mock_cli_config = Mock()
        mock_cli_config.store_location = "./test_store"
        mock_cli_config.emb_model = "test-embed"
        mock_cli_config.n_reviews = "7"  # String value
        mock_cli_config.model = "test-llm"
        mock_cli_config.temperature = 0.1
        mock_get_cli_overrides.return_value = mock_cli_config
        
        # Setup mocks
        mock_retriever = Mock()
        mock_create_vector_store.return_value = mock_retriever
        mock_model = Mock()
        mock_prompt = Mock()
        mock_chain = Mock()
        mock_prompt.__or__ = Mock(return_value=mock_chain)
        mock_create_llm_chain.return_value = (mock_model, mock_prompt)
        
        # Setup input sequence: immediate quit
        mock_input.side_effect = ["q"]
        
        # Run main
        main()
        
        # Verify that n_reviews was converted to int when passed to create_vector_store
        mock_create_vector_store.assert_called_once_with(
            db_location="./test_store",
            embedding_model="test-embed",
            k_value=7  # Should be converted to int
        )


class TestMainIntegration:
    """Integration tests for main function."""
    
    @pytest.mark.integration
    @patch('rag_examples.main.get_cli_overrides')
    @patch('rag_examples.main.create_vector_store')
    @patch('rag_examples.main.create_llm_chain')
    @patch('builtins.input')
    @patch('builtins.print')
    def test_main_full_workflow(self, mock_print, mock_input, mock_create_llm_chain, mock_create_vector_store, mock_get_cli_overrides):
        """Test complete main workflow integration."""
        # Setup CLI config
        mock_cli_config = Mock()
        mock_cli_config.store_location = "./integration_store"
        mock_cli_config.emb_model = "integration-embed"
        mock_cli_config.n_reviews = 5
        mock_cli_config.model = "integration-llm"
        mock_cli_config.temperature = 0.3
        mock_get_cli_overrides.return_value = mock_cli_config
        
        # Setup vector store
        mock_retriever = Mock()
        mock_retriever.invoke.return_value = [
            "Great pizza with perfect crust",
            "Amazing service and friendly staff"
        ]
        mock_create_vector_store.return_value = mock_retriever
        
        # Setup LLM chain
        mock_model = Mock()
        mock_prompt = Mock()
        mock_chain = Mock()
        mock_chain.invoke.return_value = "Based on the reviews, this restaurant has excellent pizza and great service."
        mock_prompt.__or__ = Mock(return_value=mock_chain)
        mock_create_llm_chain.return_value = (mock_model, mock_prompt)
        
        # Setup interaction
        mock_input.side_effect = ["What do you think about this restaurant?", "q"]
        
        # Run main
        main()
        
        # Verify the complete workflow
        mock_get_cli_overrides.assert_called_once()
        mock_create_vector_store.assert_called_once()
        mock_create_llm_chain.assert_called_once()
        mock_retriever.invoke.assert_called_once_with("What do you think about this restaurant?")
        mock_chain.invoke.assert_called_once_with({
            "reviews": ["Great pizza with perfect crust", "Amazing service and friendly staff"],
            "question": "What do you think about this restaurant?"
        })
        
        # Verify output includes the LLM response
        print_calls = [call.args[0] for call in mock_print.call_args_list]
        output = " ".join(print_calls)
        assert "Based on the reviews, this restaurant has excellent pizza and great service." in output