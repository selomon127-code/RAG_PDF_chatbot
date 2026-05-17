# test_pipeline.py - Test the complete end-to-end RAG pipeline
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import modules from all parts of the system
from src.pdf_loader import load_pdf, get_full_text, save_extracted_text
from src.chunker import chunk_pages, print_chunk_stats, save_chunks
from src.embedder import load_embedding_model, build_vectorstore, load_index
from src.rag_engine import answer_question

# Test PDF path
PDF_PATH = r"data/uploads/test.pdf"

print("=" * 60)
print("TESTING COMPLETE END-TO-END RAG PIPELINE")
print("=" * 60)

# Step 1: Check if PDF exists or create a test one
if not os.path.exists(PDF_PATH):
    print(f"❌ PDF not found: {PDF_PATH}")
    print("\nCreating a test PDF for you...")
    try:
        import fitz
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 50), "Machine Learning Basics")
        page.insert_text((50, 80), "This is a test document for my RAG PDF chatbot.")
        page.insert_text((50, 110), "Key concepts include: supervised learning, unsupervised learning, and reinforcement learning.")
        page.insert_text((50, 140), "Supervised learning uses labeled data for training.")
        page.insert_text((50, 170), "Unsupervised learning finds patterns in unlabeled data.")
        page.insert_text((50, 200), "Reinforcement learning learns through rewards and punishments.")
        
        os.makedirs("data/uploads", exist_ok=True)
        doc.save(PDF_PATH)
        doc.close()
        print(f"✅ Created test PDF at: {PDF_PATH}")
    except Exception as e:
        print(f"❌ Could not create test PDF: {e}")
        sys.exit(1)

print(f"\n📄 Using PDF: {PDF_PATH}\n")

# -------------------------------------------------------------------
# STEP 2: LOAD & EXTRACT TEXT
# -------------------------------------------------------------------
print("STEP 1: Loading PDF...")
try:
    pages = load_pdf(PDF_PATH)
    print(f"✅ Loaded {len(pages)} pages")
except Exception as e:
    print(f"❌ Error loading PDF: {e}")
    sys.exit(1)

# Save raw text for debugging
save_extracted_text(pages, "data/uploads/debug_extracted.txt")

# -------------------------------------------------------------------
# STEP 3: CHUNKING
# -------------------------------------------------------------------
print("\nSTEP 2: Creating chunks...")
try:
    chunks = chunk_pages(pages, chunk_size=500, chunk_overlap=50)
    print(f"✅ Created {len(chunks)} chunks")
except Exception as e:
    print(f"❌ Error creating chunks: {e}")
    sys.exit(1)

print_chunk_stats(chunks)
save_chunks(chunks, "data/uploads/debug_chunks.txt")

# -------------------------------------------------------------------
# STEP 4: VECTOR EMBEDDING & INDEXING (ChromaDB/FAISS)
# -------------------------------------------------------------------
print("\nSTEP 3: Loading embedding model & building vector store...")
try:
    embed_model = load_embedding_model()
    # Build or load vectorstore safely
    embed_model, index, chunks_meta = build_vectorstore(chunks)
    print("✅ Vector store successfully built and saved to data/vectorstore/")
except Exception as e:
    print(f"❌ Error during Vector Indexing: {e}")
    sys.exit(1)

# -------------------------------------------------------------------
# STEP 5: GENERATE ANSWER VIA GEMINI RAG
# -------------------------------------------------------------------
print("\nSTEP 4: Testing RAG Query Engine with Google Gemini...")

test_query = "What is supervised learning?"
print(f"\n💬 Question: '{test_query}'")

try:
    result = answer_question(
        question=test_query,
        model=embed_model,
        index=index,
        chunks=chunks_meta,
        llm_backend="gemini" # የ እኛን ነፃ Gemini ሞተር እዚህ እንጠራዋለን
    )
    
    print("\n🤖 Gemini Answer:")
    print("-" * 50)
    print(result["answer"])
    print("-" * 50)
    
    print("\n📌 Sources Used:")
    for src in result["sources"]:
        print(f"   - Page {src['page']} (Similarity Score: {src['score']:.3f})")

except Exception as e:
    print(f"❌ Error during RAG Generation: {e}")
    sys.exit(1)

# Full text statistics
full_text = get_full_text(pages)
print(f"\n📊 Total characters extracted: {len(full_text):,}")
print("\n🎉 End-to-End Pipeline test complete and successful!")