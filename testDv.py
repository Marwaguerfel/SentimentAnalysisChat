from openai import OpenAI
import os
from key import OPENAI_API_KEY

client = OpenAI(
    api_key=OPENAI_API_KEY
)

completion = client.completions.create(  # Changed from chat.completions to completions
    model="text-davinci-003",
    prompt="Write a haiku about recursion in programming.",  # Changed from messages to prompt
    max_tokens=100  # Adding max_tokens parameter
)

print(completion.choices[0].text)  # Changed from message.content to text
