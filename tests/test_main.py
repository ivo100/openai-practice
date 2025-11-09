"""Unit tests for simple.main module."""
import pytest
from unittest.mock import Mock, MagicMock, patch, mock_open
from pathlib import Path
import os

from simple.main import Simple


class TestSimple:
    """Test cases for Simple class."""

    @pytest.fixture
    def simple_instance(self, mocker):
        """Create a Simple instance with mocked OpenAI client."""
        # Mock environment variables
        mocker.patch.dict(os.environ, {"OPENAI_API_KEY": "test-api-key"})
        mocker.patch("simple.main.load_dotenv")
        
        # Create instance
        instance = Simple()
        
        # Mock the OpenAI client
        instance.client = Mock()
        
        return instance

    def test_upload_pdf_file_not_found(self, simple_instance, tmp_path):
        """Test upload_pdf raises FileNotFoundError when file doesn't exist."""
        non_existent_file = tmp_path / "nonexistent.pdf"
        
        with pytest.raises(FileNotFoundError, match="PDF file not found"):
            simple_instance.upload_pdf(str(non_existent_file))

    def test_upload_pdf_success(self, simple_instance, tmp_path, mocker):
        """Test successful PDF upload and vector store creation."""
        # Create a test PDF file
        test_pdf = tmp_path / "test.pdf"
        test_pdf.write_bytes(b"fake pdf content")
        
        # Mock OpenAI API responses
        mock_file = Mock()
        mock_file.id = "file-123"
        
        mock_vector_store = Mock()
        mock_vector_store.id = "vs-456"
        
        mock_file_status = Mock()
        mock_file_status.status = "processed"
        
        simple_instance.client.files.create.return_value = mock_file
        simple_instance.client.vector_stores.create.return_value = mock_vector_store
        simple_instance.client.files.retrieve.return_value = mock_file_status
        
        # Mock time.sleep to avoid waiting
        mocker.patch("time.sleep")
        
        # Execute
        file_id, vector_store_id = simple_instance.upload_pdf(str(test_pdf))
        
        # Assertions
        assert file_id == "file-123"
        assert vector_store_id == "vs-456"
        simple_instance.client.files.create.assert_called_once()
        simple_instance.client.vector_stores.create.assert_called_once()
        assert simple_instance.client.vector_stores.create.call_args[1]["file_ids"] == ["file-123"]

    def test_upload_pdf_file_processing_error(self, simple_instance, tmp_path, mocker):
        """Test upload_pdf raises exception when file processing fails."""
        # Create a test PDF file
        test_pdf = tmp_path / "test.pdf"
        test_pdf.write_bytes(b"fake pdf content")
        
        # Mock OpenAI API responses
        mock_file = Mock()
        mock_file.id = "file-123"
        
        mock_vector_store = Mock()
        mock_vector_store.id = "vs-456"
        
        mock_file_status = Mock()
        mock_file_status.status = "error"
        mock_file_status.last_error = "Processing failed"
        
        simple_instance.client.files.create.return_value = mock_file
        simple_instance.client.vector_stores.create.return_value = mock_vector_store
        simple_instance.client.files.retrieve.return_value = mock_file_status
        
        # Mock time.sleep to avoid waiting
        mocker.patch("time.sleep")
        
        # Execute and assert
        with pytest.raises(Exception, match="File processing failed"):
            simple_instance.upload_pdf(str(test_pdf))

    def test_upload_pdf_custom_vector_store_name(self, simple_instance, tmp_path, mocker):
        """Test upload_pdf with custom vector store name."""
        # Create a test PDF file
        test_pdf = tmp_path / "test.pdf"
        test_pdf.write_bytes(b"fake pdf content")
        
        # Mock OpenAI API responses
        mock_file = Mock()
        mock_file.id = "file-123"
        
        mock_vector_store = Mock()
        mock_vector_store.id = "vs-456"
        
        mock_file_status = Mock()
        mock_file_status.status = "processed"
        
        simple_instance.client.files.create.return_value = mock_file
        simple_instance.client.vector_stores.create.return_value = mock_vector_store
        simple_instance.client.files.retrieve.return_value = mock_file_status
        
        # Mock time.sleep to avoid waiting
        mocker.patch("time.sleep")
        
        # Execute
        file_id, vector_store_id = simple_instance.upload_pdf(
            str(test_pdf), 
            vector_store_name="custom_store"
        )
        
        # Assertions
        assert file_id == "file-123"
        assert vector_store_id == "vs-456"
        assert simple_instance.client.vector_stores.create.call_args[1]["name"] == "custom_store"

    def test_search_pdf_success(self, simple_instance, mocker):
        """Test successful PDF search."""
        vector_store_id = "vs-456"
        query = "What is the main topic?"
        
        # Mock response
        mock_response = Mock()
        mock_response.output_text = "The main topic is AI."
        
        simple_instance.client.responses.create.return_value = mock_response
        
        # Mock time.time to control timing
        mocker.patch("time.time", side_effect=[0.0, 1.5])
        
        # Execute
        response = simple_instance.search_pdf(vector_store_id, query)
        
        # Assertions
        assert response == mock_response
        simple_instance.client.responses.create.assert_called_once()
        call_args = simple_instance.client.responses.create.call_args[1]
        assert call_args["model"] == "gpt-4.1"
        assert call_args["input"] == query
        assert call_args["tools"] == [{"type": "file_search"}]
        assert call_args["file_search"]["vector_store_ids"] == [vector_store_id]

    def test_search_pdf_custom_model_and_instructions(self, simple_instance, mocker):
        """Test search_pdf with custom model and instructions."""
        vector_store_id = "vs-456"
        query = "What is the main topic?"
        custom_model = "gpt-4o"
        custom_instructions = "You are a technical expert."
        
        # Mock response
        mock_response = Mock()
        simple_instance.client.responses.create.return_value = mock_response
        
        # Mock time.time to control timing
        mocker.patch("time.time", side_effect=[0.0, 1.5])
        
        # Execute
        response = simple_instance.search_pdf(
            vector_store_id, 
            query, 
            model=custom_model,
            instructions=custom_instructions
        )
        
        # Assertions
        assert response == mock_response
        call_args = simple_instance.client.responses.create.call_args[1]
        assert call_args["model"] == custom_model
        assert call_args["instructions"] == custom_instructions

    def test_upload_pdf_and_search(self, simple_instance, tmp_path, mocker):
        """Test the convenience method that combines upload and search."""
        # Create a test PDF file
        test_pdf = tmp_path / "test.pdf"
        test_pdf.write_bytes(b"fake pdf content")
        
        query = "What is the main topic?"
        
        # Mock OpenAI API responses
        mock_file = Mock()
        mock_file.id = "file-123"
        
        mock_vector_store = Mock()
        mock_vector_store.id = "vs-456"
        
        mock_file_status = Mock()
        mock_file_status.status = "processed"
        
        mock_response = Mock()
        mock_response.output_text = "The main topic is AI."
        
        simple_instance.client.files.create.return_value = mock_file
        simple_instance.client.vector_stores.create.return_value = mock_vector_store
        simple_instance.client.files.retrieve.return_value = mock_file_status
        simple_instance.client.responses.create.return_value = mock_response
        
        # Mock time.sleep and time.time
        mocker.patch("time.sleep")
        mocker.patch("time.time", side_effect=[0.0, 1.5])
        
        # Execute
        response, vector_store_id = simple_instance.upload_pdf_and_search(
            str(test_pdf), 
            query
        )
        
        # Assertions
        assert response == mock_response
        assert vector_store_id == "vs-456"
        simple_instance.client.files.create.assert_called_once()
        simple_instance.client.vector_stores.create.assert_called_once()
        simple_instance.client.responses.create.assert_called_once()

