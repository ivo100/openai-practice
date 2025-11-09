from openai import OpenAI
from dotenv import load_dotenv
import os
import time
from pathlib import Path


class Simple:
    def __init__(self):
        # Load environment variables from .env file
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")
        assert self.api_key is not None, "OPENAI_API_KEY is not set"
        self.client = OpenAI()
        #self.model = "gpt-5-nano"
        #self.model = "gpt-5-mini"
        #self.model = "gpt-5"
        self.model = "gpt-4.1-mini"
        #self.model = "gpt-4o-mini"
        #self.model = "gpt-4o"
        print(f"Using model: {self.model}")

    def _create_response_with_timing(self, model: str, input_data, label: str = "Response"):
        """Create a response with timing and print the elapsed time."""
        start_time = time.time()
        response = self.client.responses.create(
            model=model,
            instructions="Talk like a pirate.",
            #reasoning={"effort": "low"},   # only for gpt-5
            input=input_data,
        )
        elapsed_time = time.time() - start_time
        print(f"{label} time: {elapsed_time:.2f} sec")
        return response

    def run(self, input: str, input2: str) :
                # response = self.client.responses.create(model=self.model, input=input)
                # return response.output_text
        context = [
            { "role": "user", "content": input }
        ]
        
        res1 = self._create_response_with_timing(self.model, context, "First response")

        # Append the first response’s output to context
        context += res1.output
        #print(res1.output_text)

        # Add the next user message
        context += [
            { "role": "user", "content": input2 }
        ]

        res2 = self._create_response_with_timing(self.model, context, "Second response")

        return res1, res2

    def web_search(self):
        resp = self.client.responses.create(
            model="gpt-4.1",
            instructions="You are a financial assistant.",
            input="Find $ORCL current stock price and summarize its recent trend.",
            tools=[{"type": "web_search"}]           
        )        
        return resp

    # def web_search(self):
    #     resp = self.client.responses.create(
    #         model="gpt-4.1",
    #         instructions="You are a financial assistant. Use both the web and local docs if helpful.",
    #         input="Find the current ORCL stock price and summarize its recent trend.",
    #         tools=[{"type": "web_search"}, {"type": "file_search"}],
    #         file_search={"vector_store_ids": ["vs_123"]},
    #     )        
    #     return resp

    def upload_pdf(self, pdf_path: str, vector_store_name: str = None):
        """
        Upload a local PDF file and create a vector store.
        
        Args:
            pdf_path: Path to the local PDF file
            vector_store_name: Optional name for the vector store. If not provided, 
                             a name will be generated from the file ID.
        
        Returns:
            Tuple of (file_id, vector_store_id)
        """
        # Step 1: Upload the PDF file
        print(f"Uploading PDF file: {pdf_path}")
        pdf_file = Path(pdf_path)
        if not pdf_file.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        with open(pdf_file, "rb") as f:
            uploaded_file = self.client.files.create(
                file=f,
                purpose="assistants"
            )
        print(f"File uploaded with ID: {uploaded_file.id}")
        
        # Step 2: Create a vector store
        print("Creating vector store...")
        store_name = vector_store_name or f"vector_store_{uploaded_file.id[:8]}"
        vector_store = self.client.vector_stores.create(
            name=store_name,
            file_ids=[uploaded_file.id]
        )
        vector_store_id = vector_store.id
        print(f"Vector store created with ID: {vector_store_id}")
        
        # Step 3: Wait for file processing to complete
        print("Waiting for file processing...")
        while True:
            file_status = self.client.files.retrieve(uploaded_file.id)
            if file_status.status == "processed":
                print("File processing completed")
                break
            elif file_status.status == "error":
                raise Exception(f"File processing failed: {file_status.last_error}")
            time.sleep(1)
        
        return uploaded_file.id, vector_store_id

    def search_pdf(self, vector_store_id: str, query: str, model: str = "gpt-4.1", 
                   instructions: str = "You are a helpful assistant that answers questions based on the provided documents."):
        """
        Perform a file search query against a vector store.
        
        Args:
            vector_store_id: The ID of the vector store to search
            query: The search query
            model: The model to use for the search (default: "gpt-4.1")
            instructions: Instructions for the assistant (default: helpful assistant)
        
        Returns:
            Response object with the search results
        """
        print(f"Searching for: {query}")
        start_time = time.time()
        response = self.client.responses.create(
            model=model,
            instructions=instructions,
            input=query,
            tools=[{"type": "file_search"}],
            file_search={"vector_store_ids": [vector_store_id]}
        )
        elapsed_time = time.time() - start_time
        print(f"Search time: {elapsed_time:.2f} sec")
        
        return response

    def upload_pdf_and_search(self, pdf_path: str, query: str):
        """
        Convenience method that uploads a PDF and performs a search in one call.
        
        Args:
            pdf_path: Path to the local PDF file
            query: The search query to search in the PDF
        
        Returns:
            Tuple of (response, vector_store_id)
        """
        file_id, vector_store_id = self.upload_pdf(pdf_path)
        response = self.search_pdf(vector_store_id, query)
        return response, vector_store_id

if __name__ == "__main__":
    simple = Simple()
    #resp = simple.run("tell me a joke") 
    # resp1, resp2 = simple.run("What is the capital of France?", "And it's population (proper)?") 
    # print(resp1.output_text)
    # print(resp2.output_text)
    # resp = simple.web_search()
    # print(resp.output_text)
    
    # Example: Upload PDF and search (split methods)
    # file_id, vector_store_id = simple.upload_pdf("path/to/your/document.pdf")
    # resp = simple.search_pdf(vector_store_id, "What is the main topic of this document?")
    # print(resp.output_text)
    
    # Example: Upload PDF and search (convenience method)
    # resp, vector_store_id = simple.upload_pdf_and_search(
    #     pdf_path="path/to/your/document.pdf",
    #     query="What is the main topic of this document?"
    # )
    # print(resp.output_text)