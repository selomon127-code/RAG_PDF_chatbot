# src/gemini_rag.py - Gemini-powered RAG engine
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# Configure Gemini
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file")

genai.configure(api_key=GOOGLE_API_KEY)

# ማስተካከያ 1፡ ይፋዊውን የሞዴል መለያ ስም (models/prefix) መጠቀም
MODEL_NAME = "models/gemini-1.5-flash"

class GeminiRAG:
    def __init__(self):
        # ማስተካከያ 2፡ ሎጂካዊ እና እውነተኛ መልስ ብቻ እንዲሰጥ temperature ዝቅ ማድረግ
        self.model = genai.GenerativeModel(
            model_name=MODEL_NAME,
            generation_config={"temperature": 0.2}
        )
        print(f"[gemini_rag] Gemini engine successfully loaded with {MODEL_NAME}")
        
    def generate_answer(self, context: str, question: str) -> str:
        """Generate answer using Gemini with safety guards"""
        prompt = f"""You are a helpful study assistant answering questions based ONLY on the provided context.

CONTEXT:
{context}

QUESTION: {question}

INSTRUCTIONS:
1. Answer ONLY using information from the context above. Do not guess.
2. If the answer is not in the context, say exactly: "I cannot find this information in the provided document."
3. Cite the relevant information or page numbers from the context.
4. Be concise, professional, and helpful.

ANSWER:"""
        
        try:
            response = self.model.generate_content(prompt)
            # ማስተካከያ 3፡ የምላሹን መኖር ማረጋገጥ (Safety block ከተፈጠረ ባዶ እንዳይሆን)
            if response.text:
                return response.text.strip()
            else:
                return "I could not find the answer in the provided document (Response was empty)."
        except Exception as e:
            return f"Error generating response: {e}"
    
    def answer_question(self, question: str, retrieved_chunks: list) -> dict:
        """Complete RAG pipeline with Gemini"""
        # ጽሁፍ መኖሩን ማረጋገጥ (Guard Clause)
        if not retrieved_chunks:
            return {
                "answer": "I could not find any relevant text chunks in the document to answer this.",
                "sources": [],
                "context": ""
            }

        # Build context from chunks
        context_parts = []
        for chunk in retrieved_chunks:
            context_parts.append(f"[Page {chunk['page']}]\n{chunk['text']}")
        context = "\n\n".join(context_parts)
        
        # Generate answer
        answer = self.generate_answer(context, question)
        
        return {
            "answer": answer,
            "sources": retrieved_chunks,
            "context": context
        }