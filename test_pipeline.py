# test_pipeline.py - Test the complete pipeline
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.pdf_loader import load_pdf, get_full_text, save_extracted_text
from src.chunker import chunk_pages, print_chunk_stats, save_chunks

PDF_PATH = r"data/uploads/test.pdf"

print("=" * 60)
print("TESTING COMPLETE RAG PIPELINE")
print("=" * 60)

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

print("STEP 1: Loading PDF...")
try:
    pages = load_pdf(PDF_PATH)
    print(f"✅ Loaded {len(pages)} pages")
except Exception as e:
    print(f"❌ Error loading PDF: {e}")
    sys.exit(1)

os.makedirs("data/uploads", exist_ok=True)
save_extracted_text(pages, "data/uploads/debug_extracted.txt")

print("\nSTEP 2: Creating chunks...")
try:
    chunks = chunk_pages(pages, chunk_size=500, chunk_overlap=50)
    print(f"✅ Created {len(chunks)} chunks")
except Exception as e:
    print(f"❌ Error creating chunks: {e}")
    sys.exit(1)

print_chunk_stats(chunks)
save_chunks(chunks, "data/uploads/debug_chunks.txt")

print("\nSTEP 3: Sample chunk:")
if chunks:
    sample = chunks[0]
    print(f"Chunk {sample['chunk_id']} (Page {sample['page']}):")
    print("-" * 40)
    print(sample['text'][:300])
    print("-" * 40)

full_text = get_full_text(pages)
print(f"\n📊 Total characters extracted: {len(full_text):,}")

print("\n✅ Pipeline test complete!")
print("📁 Check: data/uploads/debug_extracted.txt and debug_chunks.txt")
