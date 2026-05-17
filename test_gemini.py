import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("❌ GOOGLE_API_KEY not found in .env")
    exit(1)

print(f"✅ API Key loaded (first 10 chars): {api_key[:10]}...")

genai.configure(api_key=api_key)
print("✅ API configured")

# List available models
print("\n📋 Available models with generateContent:")
found = False
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(f"  ✅ {m.name}")
        found = True
if not found:
    print("  ⚠️ No models found - check API key permissions")

# Test generation with CORRECT model name (no 'models/' prefix)
print("\n🧪 Testing generation with 'gemini-1.5-flash'...")
try:
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content("Say hello in one word")
    if response and response.text:
        print(f"✅ Success! Response: '{response.text.strip()}'")
    else:
        print("⚠️ Empty response (check safety settings)")
except Exception as e:
    print(f"❌ Error: {e}")
    # Try fallback model
    print("\n🔄 Trying fallback model 'gemini-1.0-pro'...")
    try:
        model = genai.GenerativeModel("gemini-1.0-pro")
        response = model.generate_content("Say hello in one word")
        if response and response.text:
            print(f"✅ Fallback success! Response: '{response.text.strip()}'")
    except Exception as e2:
        print(f"❌ Fallback also failed: {e2}")
