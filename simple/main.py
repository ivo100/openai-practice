from openai import OpenAI
from dotenv import load_dotenv
import os

MODEL = "gpt-4o-mini"


def main() -> None:
    # Load environment variables from .env file
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    #print(f"API Key: {api_key}")

    client = OpenAI()

    response = client.responses.create(
        model=MODEL,
        input="tell me a joke"
    )
    #print(response.output)
    #print(response)
    print(response.output_text)
"""
Response(id='resp_0804ca6536c1056400690fa6d171c48193b8cecb21edda4a3d', created_at=1762633425.0, error=None, 
incomplete_details=None, instructions=None, metadata={}, model='gpt-4o-mini-2024-07-18', 
object='response',
 
output=[ResponseOutputMessage(id='msg_0804ca6536c1056400690fa6d2537c8193964c5a7b4a2178b5', 
content=[ResponseOutputText(annotations=[], 
text='Why did the scarecrow win an award?\n\nBecause he was outstanding in his field!', 
type='output_text', logprobs=[])], 
role='assistant', status='completed', 
type='message')],
 
parallel_tool_calls=True, temperature=1.0, tool_choice='auto', tools=[], top_p=1.0, background=False, conversation=None, max_output_tokens=None, max_tool_calls=None, 
previous_response_id=None, prompt=None, prompt_cache_key=None, reasoning=Reasoning(effort=None, generate_summary=None, summary=None), 
safety_identifier=None, service_tier='default', status='completed', text=ResponseTextConfig(format=ResponseFormatText(type='text'), verbosity='medium'), 
top_logprobs=0, truncation='disabled', 

usage=ResponseUsage(input_tokens=11, input_tokens_details=InputTokensDetails(cached_tokens=0), 
output_tokens=18, output_tokens_details=OutputTokensDetails(reasoning_tokens=0), total_tokens=29), 
user=None, billing={'payer': 'openai'}, 
prompt_cache_retention=None, store=True)

"""

if __name__ == "__main__":
    main()
