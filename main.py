# write python code for GOLD DAG construction here

# TEST CODE FOR API KEY
# main.py - auth test only, not final code

# Load env vars from .env (api key)
import os
from dotenv import load_dotenv
from anthropic import Anthropic

# Pull the api key from .env into the environment
load_dotenv()

# Init the client (reads ANTHROPIC_API_KEY env var automatically)
client = Anthropic()

# Minimal test call - sonnet is cheap, good for auth check
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=50,
    messages=[
        {"role": "user", "content": "Reply with exactly the word: authenticated"}
    ],
)

# Print reply + cost breakdown
print("Response:", response.content[0].text)
print("Input tokens:", response.usage.input_tokens)
print("Output tokens:", response.usage.output_tokens)
