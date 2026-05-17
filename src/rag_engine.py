# src/rag_engine.py
# RAG Engine — Optimized for ChromaDB & Gemini (google-genai SDK) + Debug Mode
import os
import sys
import time
import hashlib
import requests
import json

# ============================================================================
# ✅ FIXED: Reliable .env loading for Windows
# ============================================================================
def _load_env_reliably():
    """Load .env with explicit path resolution for Windows compatibility."""
    try:
        from dotenv import load_dotenv
        # Project root is 2 levels up from src/rag_engine.py
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        env_path = os.path.join(project_root, ".env")
        
        if os.path.exists(env_path):
            result = load_dotenv(dotenv_path=env_path, override=True)
            print(f"[rag_engine] ✅ Loaded .env from: {env_path} | Success: {result}")
            return True
        # Fallback: try current directory
        result = load_dotenv(override=True)
        print(f"[rag_engine] ✅ Loaded .env from current directory | Success: {result}")
        return result
    except Exception as e:
        print(f"[rag_engine] ⚠️ Could not load .env: {e}")
        return False

# Load environment variables BEFORE any other imports that depend on them
_load_env_reliably()

# Now import Google SDK (after env vars are loaded)
from google import genai
from google.genai import types

# ============================================================================
# CONFIGURATION
# ============================================================================
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "").strip()
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"

if not GOOGLE_API_KEY and not DEBUG_MODE:
    raise ValueError(
        "[rag_engine] ❌ GOOGLE_API_KEY not found.\n"
        "Add to .env: GOOGLE_API_KEY=AIzaSy...\n"
        "Or set DEBUG_MODE=true for offline testing"
    )

# Initialize client only if we have a key
client = genai.Client(api_key=GOOGLE_API_KEY) if GOOGLE_API_KEY else None
if client:
    print(f"[rag_engine] ✅ Gemini client initialized | Key: {GOOGLE_API_KEY[:8]}...")

DEFAULT_MODEL = "gemini-2.0-flash-lite"
AVAILABLE_MODELS = [
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.5-flash",
    "gemini-flash-latest",
]

_response_cache: dict = {}
CACHE_TTL_SECONDS = 300

# ============================================================================
# HELPERS
# ============================================================================
def _get_cache_key(question: str, context: str, model: str, backend: str) -> str:
    key_data = f"{backend}|||{model}|||{question}|||{context[:500]}"
    return hashlib.md5(key_data.encode("utf-8")).hexdigest()

def _is_cache_valid(timestamp: float) -> bool:
    return (time.time() - timestamp) < CACHE_TTL_SECONDS

def _clean_model_name(model_name: str) -> str:
    return model_name.replace("models/", "").strip()

def _generate_debug_response(question: str, context: str) -> str:
    """Fallback response for DEBUG_MODE — simulates RAG answer."""
    q_lower = question.lower()
    
    if "artificial intelligence" in q_lower or "ai" in q_lower:
        return (
            "Based on the uploaded document:\n\n"
            "**Artificial Intelligence (AI)** refers to the simulation of human intelligence "
            "processes by machines, especially computer systems. These processes include:\n\n"
            "• **Learning**: Acquiring information and rules for using it\n"
            "• **Reasoning**: Using rules to reach conclusions\n"
            "• **Self-correction**: Improving performance over time\n\n"
            "AI applications include expert systems, natural language processing, "
            "speech recognition, and machine vision.\n\n"
            "*(This is a simulated response for DEBUG_MODE. Set DEBUG_MODE=false for real answers.)*"
        )
    else:
        return (
            f"Based on the context provided for: \"{question}\"\n\n"
            "The document contains relevant information on this topic. "
            "Key points include definitions, examples, and applications "
            "discussed across multiple pages.\n\n"
            "*(Simulated response — set DEBUG_MODE=false for real API answers)*"
        )

def generate_with_retry(
    model_name: str,
    prompt: str,
    max_retries: int = 3,
    base_delay: float = 2.0,
) -> str:
    """Call Gemini API with exponential backoff for quota errors."""
    
    # DEBUG MODE: Skip API entirely
    if DEBUG_MODE:
        print("[rag_engine] 🧪 DEBUG_MODE: Using simulated response")
        return _generate_debug_response("", prompt)
    
    if not client:
        return "❌ Gemini client not initialized. Check API key or enable DEBUG_MODE."
    
    clean_name = _clean_model_name(model_name)
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            print(f"[rag_engine] 📡 Calling Gemini: {clean_name} (attempt {attempt+1})")
            
            response = client.models.generate_content(
                model=clean_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    max_output_tokens=1024,
                    safety_settings=[
                        types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_MEDIUM_AND_ABOVE"),
                        types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_MEDIUM_AND_ABOVE"),
                        types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_MEDIUM_AND_ABOVE"),
                        types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_MEDIUM_AND_ABOVE"),
                    ],
                ),
            )

            if response and response.text:
                print(f"[rag_engine] ✅ Got response ({len(response.text)} chars)")
                return response.text.strip()

            if response and hasattr(response, "prompt_feedback"):
                reason = getattr(response.prompt_feedback, "block_reason", "unknown")
                return f"⚠️ Response blocked by safety filter ({reason}). Please rephrase."

            return "⚠️ Model returned an empty response."

        except Exception as e:
            last_error = e
            error_str = str(e).lower()
            print(f"[rag_engine] ❌ Attempt {attempt+1} failed: {type(e).__name__}: {e}")

            is_quota = any(kw in error_str for kw in ["quota", "429", "rate limit", "resource exhausted", "too many requests"])

            if is_quota and attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                print(f"[rag_engine] ⏳ Quota hit. Retrying in {delay:.0f}s...")
                time.sleep(delay)
                continue

            # If all retries failed, try DEBUG_MODE fallback
            if DEBUG_MODE and attempt == max_retries:
                print("[rag_engine] 🔄 Falling back to DEBUG_MODE response")
                return _generate_debug_response("", prompt)
            
            raise

    raise last_error

# ============================================================================
# MAIN FUNCTION
# ============================================================================
def answer_question(
    question: str,
    model,  # Kept for compatibility
    index,  # ChromaDB VectorStore
    chunks: list,
    llm_backend: str = "gemini",
    llm_model: str = "gemini-2.0-flash-lite",
) -> dict:
    """Answer a question using RAG with ChromaDB + Gemini/Ollama."""
    
    print(f"\n[rag_engine] 🔍 Question: '{question}'")
    print(f"[rag_engine] 🔧 Backend: {llm_backend} | Model: {llm_model} | DEBUG: {DEBUG_MODE}")

    # STEP 1: Search ChromaDB
    try:
        retrieved = index.search(query=question, top_k=5)
        print(f"[rag_engine] 📊 Retrieved {len(retrieved) if retrieved else 0} chunks")
        
        if retrieved and len(retrieved) > 0:
            print(f"[rag_engine] 🎯 Best match score: {retrieved[0].get('score', 0):.4f}")
            print(f"[rag_engine] 📄 Sample chunk: {retrieved[0].get('text', '')[:100]}...")
            
    except Exception as e:
        print(f"[rag_engine] ❌ Search error: {e}")
        return {
            "answer": f"⚠️ Vector search error: {str(e)[:300]}", 
            "sources": [], 
            "context": ""
        }

    # STEP 2: Handle empty results
    if not retrieved or len(retrieved) == 0:
        fallback_msg = "🔍 No relevant information found."
        if DEBUG_MODE:
            fallback_msg += " (DEBUG_MODE: Try a different question or check your PDF text extraction)"
        return {
            "answer": fallback_msg,
            "sources": [],
            "context": "",
        }

    # STEP 3: Build context + sources
    context_parts = []
    sources = []

    for item in retrieved:
        text = item.get("text", "").strip()
        if text:
            page_num = item.get("page", "Unknown")
            context_parts.append(f"[Page {page_num}]\n{text}")
            sources.append({
                "page": page_num,
                "chunk_id": item.get("chunk_id", 0),
                "text": text[:200] + ("..." if len(text) > 200 else ""),
                "score": item.get("score", 0),
            })

    context = "\n\n".join(context_parts)
    print(f"[rag_engine] 📝 Context built: {len(context)} chars from {len(sources)} sources")

    # STEP 4: Build prompt
    prompt = f"""You are a helpful academic assistant. Answer using ONLY the provided context.

CONTEXT:
{context}

QUESTION: {question}

INSTRUCTIONS:
1. Answer ONLY using information from the CONTEXT above.
2. If the answer is not in the context, say: "I cannot find this information in the provided document."
3. Cite page numbers when referencing content.
4. Be concise and accurate.

ANSWER:"""

    # STEP 5: Cache check
    cache_key = _get_cache_key(question, context, llm_model, llm_backend)
    cached = _response_cache.get(cache_key)
    if cached and _is_cache_valid(cached["timestamp"]):
        print("[rag_engine] 🎯 Cache hit")
        return {"answer": cached["answer"], "sources": sources, "context": context}

    # STEP 6: Generate answer
    if llm_backend.lower() == "ollama":
        try:
            url = "http://localhost:11434/api/generate"
            payload = {"model": llm_model, "prompt": prompt, "stream": False}
            response = requests.post(url, json=payload, timeout=60)
            if response.status_code == 200:
                answer = response.json().get("response", "").strip()
            else:
                answer = f"⚠️ Ollama error (Status: {response.status_code})."
        except requests.exceptions.ConnectionError:
            answer = "⚠️ Ollama not running. Start with: `ollama serve`"
        except Exception as e:
            answer = f"❌ Ollama Error: {str(e)}"
            
    else:  # Gemini
        effective_model = llm_model if llm_model else DEFAULT_MODEL
        try:
            answer = generate_with_retry(effective_model, prompt)
        except Exception as e:
            error_str = str(e).lower()
            print(f"[rag_engine] ❌ Final error: {type(e).__name__}: {repr(e)}")

            if any(kw in error_str for kw in ["quota", "429", "rate limit", "resource exhausted"]):
                answer = (
                    "⏳ **API quota exceeded.**\n\n"
                    "Free tier: ~15 requests/minute, ~1,500/day\n\n"
                    "**Quick fixes**:\n"
                    "1. Wait 60 seconds and retry\n"
                    "2. Set `DEBUG_MODE=true` in `.env` for offline testing\n"
                    "3. Use Ollama backend for unlimited local inference\n"
                    "4. Upgrade at [Google AI Studio](https://aistudio.google.com/)"
                )
            elif any(kw in error_str for kw in ["api_key", "authentication", "401", "invalid key"]):
                answer = "🔑 **API key error**. Check your `.env` file or set `DEBUG_MODE=true`."
            else:
                answer = f"❌ Error: {str(e)[:250]}"

    # STEP 7: Cache successful answers
    is_error = any(c in answer for c in ["⚠️", "❌", "🔑", "🚫", "🌐", "⏳"])
    if not is_error and not DEBUG_MODE:
        _response_cache[cache_key] = {"answer": answer, "timestamp": time.time()}
        if len(_response_cache) > 100:
            oldest = min(_response_cache, key=lambda k: _response_cache[k]["timestamp"])
            del _response_cache[oldest]

    return {"answer": answer, "sources": sources, "context": context}


def clear_cache():
    """Clear the in-memory response cache."""
    global _response_cache
    count = len(_response_cache)
    _response_cache = {}
    print(f"[rag_engine] 🗑️ Cleared {count} cached responses")