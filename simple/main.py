from openai import OpenAI
from dotenv import load_dotenv
import os
import time


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
            input=input_data,
        )
        elapsed_time = time.time() - start_time
        print(f"{label} time: {elapsed_time:.2f} sec")
        return response

    def run(self, input: str, input2: str) -> None:
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


if __name__ == "__main__":
    simple = Simple()
    #resp = simple.run("tell me a joke") 
    resp1, resp2 = simple.run("What is the capital of France?", "And it's population (proper)?") 
    print(resp1.output_text)
    print(resp2.output_text)
