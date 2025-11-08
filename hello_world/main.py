from openai import OpenAI
from dotenv import load_dotenv
import os

MODEL = "gpt-4o-mini"


def main() -> None:
    # Load environment variables from .env file
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    print(f"API Key: {api_key}")

    client = OpenAI()

    response = client.responses.create(
        model=MODEL,
        input="why sky is blue?"
    )

    print(response.output_text)


if __name__ == "__main__":
    main()
