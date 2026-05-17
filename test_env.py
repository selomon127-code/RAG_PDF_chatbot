import os
import sys
from dotenv import load_dotenv

# Get the directory where this script runs
script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
env_path = os.path.join(script_dir, ".env")

print(f"Looking for .env at: {env_path}")
print(f"File exists: {os.path.exists(env_path)}")

# Load with explicit path
if os.path.exists(env_path):
    result = load_dotenv(dotenv_path=env_path, override=True)
    print(f"load_dotenv result: {result}")
else:
    # Fallback: try current directory
    result = load_dotenv(override=True)
    print(f"load_dotenv (fallback) result: {result}")

# Check values
key = os.getenv("GOOGLE_API_KEY")
debug = os.getenv("DEBUG_MODE")

print(f"\nGOOGLE_API_KEY: {'SET' if key else 'MISSING'}")
if key:
    print(f"  Preview: {key[:20]}...")
print(f"DEBUG_MODE: {debug}")

# Test Gemini if key is loaded
if key:
    try:
        from google import genai
        client = genai.Client(api_key=key)
        resp = client.models.generate_content(
            model="gemini-2.0-flash-lite",
            contents="Hi",
        )
        if resp and resp.text:
            print(f"\n✅ Gemini works: '{resp.text.strip()}'")
        else:
            print("\n⚠️ Empty response from Gemini")
    except Exception as e:
        err = str(e).lower()
        if "quota" in err or "429" in err:
            print("\n⏳ Quota exceeded - wait 60 seconds")
        elif "api_key" in err or "401" in err:
            print("\n🔑 Invalid API key")
        else:
            print(f"\n❌ Error: {e}")
