import os
from google import genai
from dotenv import load_dotenv

load_dotenv(override=True)
key = os.getenv("GOOGLE_API_KEY")

if not key:
    print("No API key found")
    exit(1)

print(f"Key loaded: {key[:15]}...")

try:
    client = genai.Client(api_key=key)
    resp = client.models.generate_content(
        model="gemini-2.0-flash-lite",
        contents="Say hello",
    )
    if resp and resp.text:
        print(f"SUCCESS: {resp.text.strip()}")
    else:
        print("Empty response")
except Exception as e:
    err = str(e).lower()
    if "quota" in err or "429" in err:
        print("Quota exceeded - wait 60 seconds")
    elif "api_key" in err or "401" in err:
        print("Invalid API key")
    else:
        print(f"Error: {e}")
