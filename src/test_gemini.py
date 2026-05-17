# test_gemini.py
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("❌ GOOGLE_API_KEY not found in .env")
    exit(1)

genai.configure(api_key=api_key)

print("✅ API configured")

# List available models
print("\n📋 Available models with generateContent:")
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(f"  • {m.name}")

# Test generation
print("\n🧪 Testing generation...")
try:
    model = genai.GenerativeModel("gemini-1.5-flash")  # ✅ No 'models/' prefix
    response = model.generate_content("Say hello in one word")
    print(f"✅ Response: {response.text}")
except Exception as e:
    print(f"❌ Error: {e}")