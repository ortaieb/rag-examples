"""
Unit tests for the vector store module.
"""
import os
import pytest
from unittest.mock import patch, Mock, MagicMock, call
import pandas as pd
from pathlib import Path

from rag_examples.vector_store import create_vector_store


class TestCreateVectorStore:
    """Test create_vector_store function."""
    
    @pytest.mark.unit
    @patch('rag_examples.vector_store.pd.read_csv')
    @patch('rag_examples.vector_store.OllamaEmbeddings')
    @patch('rag_examples.vector_store.Chroma')
    @patch('rag_examples.vector_store.os.path.exists')
    def test_create_vector_store_new_db(self, mock_exists, mock_chroma, mock_embeddings, mock_read_csv, sample_reviews_df):
        """Test creating vector store when database doesn't exist."""
        # Setup mocks
        mock_exists.return_value = False
        mock_read_csv.return_value = sample_reviews_df
        mock_embedding_instance = Mock()
        mock_embeddings.return_value = mock_embedding_instance
        
        mock_vector_store = Mock()
        mock_retriever = Mock()
        mock_vector_store.as_retriever.return_value = mock_retriever
        mock_chroma.return_value = mock_vector_store
        
        # Call function
        result = create_vector_store(
            db_location="./test_db",
            embedding_model="test-embed-model",
            k_value=5
        )
        
        # Verify CSV was read
        mock_read_csv.assert_called_once_with("knowledge/reviews.csv")
        
        # Verify embeddings were created
        mock_embeddings.assert_called_once_with(model="test-embed-model")
        
        # Verify Chroma was initialized
        mock_chroma.assert_called_once_with(
            collection_name="Restaurant_Reviews",
            persist_directory="./test_db",
            embedding_function=mock_embedding_instance
        )
        
        # Verify documents were added
        mock_vector_store.add_documents.assert_called_once()
        call_args = mock_vector_store.add_documents.call_args
        documents = call_args[1]['documents']
        ids = call_args[1]['ids']
        
        # Check that documents were created correctly
        assert len(documents) == 3
        assert documents[0].page_content == "Great pizza The pizza was absolutely delicious with perfect crust."
        assert documents[0].metadata == {"rating": 5, "date": "2024-01-01"}
        assert documents[0].id == "0"
        
        assert ids == ["0", "1", "2"]
        
        # Verify retriever was created
        mock_vector_store.as_retriever.assert_called_once_with(search_kwargs={"k": 5})
        assert result == mock_retriever
    
    @pytest.mark.unit
    @patch('rag_examples.vector_store.pd.read_csv')
    @patch('rag_examples.vector_store.OllamaEmbeddings')
    @patch('rag_examples.vector_store.Chroma')
    @patch('rag_examples.vector_store.os.path.exists')
    def test_create_vector_store_existing_db(self, mock_exists, mock_chroma, mock_embeddings, mock_read_csv, sample_reviews_df):
        """Test creating vector store when database already exists."""
        # Setup mocks
        mock_exists.return_value = True
        mock_read_csv.return_value = sample_reviews_df
        mock_embedding_instance = Mock()
        mock_embeddings.return_value = mock_embedding_instance
        
        mock_vector_store = Mock()
        mock_retriever = Mock()
        mock_vector_store.as_retriever.return_value = mock_retriever
        mock_chroma.return_value = mock_vector_store
        
        # Call function
        result = create_vector_store(
            db_location="./existing_db",
            embedding_model="test-embed-model",
            k_value=3
        )
        
        # Verify CSV was still read (needed for document creation logic)
        mock_read_csv.assert_called_once_with("knowledge/reviews.csv")
        
        # Verify embeddings were created
        mock_embeddings.assert_called_once_with(model="test-embed-model")
        
        # Verify Chroma was initialized
        mock_chroma.assert_called_once_with(
            collection_name="Restaurant_Reviews",
            persist_directory="./existing_db",
            embedding_function=mock_embedding_instance
        )
        
        # Verify documents were NOT added (since db exists)
        mock_vector_store.add_documents.assert_not_called()
        
        # Verify retriever was created with correct k value
        mock_vector_store.as_retriever.assert_called_once_with(search_kwargs={"k": 3})
        assert result == mock_retriever
    
    @pytest.mark.unit
    @patch('rag_examples.vector_store.pd.read_csv')
    @patch('rag_examples.vector_store.OllamaEmbeddings')
    @patch('rag_examples.vector_store.Chroma')
    @patch('rag_examples.vector_store.os.path.exists')
    def test_create_vector_store_document_creation(self, mock_exists, mock_chroma, mock_embeddings, mock_read_csv):
        """Test document creation from CSV data."""
        # Setup test data
        test_df = pd.DataFrame({
            'Title': ['Pizza Review', 'Service Review'],
            'Date': ['2024-01-01', '2024-01-02'],
            'Rating': [4, 3],
            'Review': ['Great pizza taste', 'Service was okay']
        })
        
        mock_exists.return_value = False
        mock_read_csv.return_value = test_df
        mock_embedding_instance = Mock()
        mock_embeddings.return_value = mock_embedding_instance
        
        mock_vector_store = Mock()
        mock_retriever = Mock()
        mock_vector_store.as_retriever.return_value = mock_retriever
        mock_chroma.return_value = mock_vector_store
        
        # Call function
        create_vector_store(
            db_location="./test_db",
            embedding_model="test-embed-model",
            k_value=2
        )
        
        # Verify documents were created and added
        mock_vector_store.add_documents.assert_called_once()
        call_args = mock_vector_store.add_documents.call_args
        documents = call_args[1]['documents']
        ids = call_args[1]['ids']
        
        # Check first document
        assert documents[0].page_content == "Pizza Review Great pizza taste"
        assert documents[0].metadata == {"rating": 4, "date": "2024-01-01"}
        assert documents[0].id == "0"
        
        # Check second document
        assert documents[1].page_content == "Service Review Service was okay"
        assert documents[1].metadata == {"rating": 3, "date": "2024-01-02"}
        assert documents[1].id == "1"
        
        # Check IDs
        assert ids == ["0", "1"]
    
    @pytest.mark.unit
    @patch('rag_examples.vector_store.pd.read_csv')
    @patch('rag_examples.vector_store.OllamaEmbeddings')
    @patch('rag_examples.vector_store.Chroma')
    @patch('rag_examples.vector_store.os.path.exists')
    def test_create_vector_store_with_different_k_values(self, mock_exists, mock_chroma, mock_embeddings, mock_read_csv, sample_reviews_df):
        """Test vector store creation with different k values."""
        mock_exists.return_value = True
        mock_read_csv.return_value = sample_reviews_df
        mock_embedding_instance = Mock()
        mock_embeddings.return_value = mock_embedding_instance
        
        mock_vector_store = Mock()
        mock_retriever = Mock()
        mock_vector_store.as_retriever.return_value = mock_retriever
        mock_chroma.return_value = mock_vector_store
        
        # Test with k=1
        create_vector_store("./test_db", "test-embed-model", 1)
        mock_vector_store.as_retriever.assert_called_with(search_kwargs={"k": 1})
        
        # Test with k=10
        create_vector_store("./test_db", "test-embed-model", 10)
        mock_vector_store.as_retriever.assert_called_with(search_kwargs={"k": 10})
    
    @pytest.mark.unit
    @patch('src.rag_examples.vector_store.pd.read_csv')
    def test_create_vector_store_csv_read_error(self, mock_read_csv):
        """Test handling of CSV read errors."""
        mock_read_csv.side_effect = FileNotFoundError("CSV file not found")
        
        with pytest.raises(FileNotFoundError):
            create_vector_store(
                db_location="./test_db",
                embedding_model="test-embed-model",
                k_value=5
            )
    
    @pytest.mark.unit
    @patch('src.rag_examples.vector_store.pd.read_csv')
    @patch('src.rag_examples.vector_store.OllamaEmbeddings')
    def test_create_vector_store_embedding_initialization(self, mock_embeddings, mock_read_csv, sample_reviews_df):
        """Test that embeddings are initialized with correct model."""
        mock_read_csv.return_value = sample_reviews_df
        mock_embedding_instance = Mock()
        mock_embeddings.return_value = mock_embedding_instance
        
        with patch('src.rag_examples.vector_store.Chroma') as mock_chroma:
            with patch('src.rag_examples.vector_store.os.path.exists', return_value=True):
                mock_vector_store = Mock()
                mock_retriever = Mock()
                mock_vector_store.as_retriever.return_value = mock_retriever
                mock_chroma.return_value = mock_vector_store
                
                create_vector_store(
                    db_location="./test_db",
                    embedding_model="custom-embedding-model",
                    k_value=5
                )
                
                mock_embeddings.assert_called_once_with(model="custom-embedding-model")
                mock_chroma.assert_called_once_with(
                    collection_name="Restaurant_Reviews",
                    persist_directory="./test_db",
                    embedding_function=mock_embedding_instance
                )


class TestVectorStoreIntegration:
    """Integration tests for vector store functionality."""
    
    @pytest.mark.integration
    @patch('rag_examples.vector_store.pd.read_csv')
    @patch('rag_examples.vector_store.OllamaEmbeddings')
    @patch('rag_examples.vector_store.Chroma')
    @patch('rag_examples.vector_store.os.path.exists')
    def test_vector_store_end_to_end_workflow(self, mock_exists, mock_chroma, mock_embeddings, mock_read_csv, sample_reviews_df):
        """Test complete vector store workflow from creation to retrieval."""
        # Setup mocks for new database
        mock_exists.return_value = False
        mock_read_csv.return_value = sample_reviews_df
        
        mock_embedding_instance = Mock()
        mock_embeddings.return_value = mock_embedding_instance
        
        mock_vector_store = Mock()
        mock_retriever = Mock()
        mock_retriever.invoke.return_value = [
            "Great pizza content",
            "Amazing food content"
        ]
        mock_vector_store.as_retriever.return_value = mock_retriever
        mock_chroma.return_value = mock_vector_store
        
        # Create vector store
        retriever = create_vector_store(
            db_location="./integration_test_db",
            embedding_model="test-embed-model",
            k_value=2
        )
        
        # Test retrieval
        results = retriever.invoke("pizza quality")
        
        # Verify the workflow
        assert mock_read_csv.called
        assert mock_embeddings.called
        assert mock_chroma.called
        assert mock_vector_store.add_documents.called
        assert mock_vector_store.as_retriever.called
        assert results == ["Great pizza content", "Amazing food content"]