import os
from dotenv import load_dotenv
load_dotenv()

from google import genai

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

print("Fetching available models...")
for m in client.models.list():
    print(m.name)