# src/chunker.py
# This module splits the extracted PDF text into smaller overlapping chunks.
# Why? Because LLMs have a context limit — you can't feed 50 pages at once.
# Instead, we find only the RELEVANT chunks and send those to the LLM.
# Smaller chunks = more precise retrieval = better answers.

# FIXED IMPORTS for LangChain 0.2.5
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:
    # Provide a lightweight fallback splitter if langchain_text_splitters
    # is not installed. This fallback splits by character windows and
    # preserves a simple chunk_overlap behaviour.
    class RecursiveCharacterTextSplitter:
        def __init__(self, chunk_size=500, chunk_overlap=50, length_function=len, separators=None):
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap

        def split_text(self, text: str) -> list[str]:
            if not text:
                return []
            chunks = []
            i = 0
            L = len(text)
            while i < L:
                end = min(i + self.chunk_size, L)
                chunk = text[i:end]
                chunks.append(chunk)
                if end == L:
                    break
                i = max(end - self.chunk_overlap, end)
            return chunks

import os    # for saving chunks to disk


# -------------------------------------------------------------------
# WHY chunk_size=500 and chunk_overlap=50?
# -------------------------------------------------------------------
#
#  chunk_size=500:
#    - 500 characters ≈ 80–120 words ≈ 3–5 sentences
#    - Small enough to be specific (you retrieve exactly what's needed)
#    - Large enough to contain a complete thought or concept
#    - Too small (e.g. 100): chunks lose context, answers are incomplete
#    - Too large (e.g. 2000): retrieval becomes vague, LLM gets overloaded
#
#  chunk_overlap=50:
#    - The last 50 characters of chunk N become the first 50 of chunk N+1
#    - This prevents answers from being CUT OFF at a chunk boundary
#    - Example without overlap: "The answer is 42" split as "The answer is" | "42"
#      → searching for "42" misses the context "The answer is"
#    - Example with overlap:    "The answer is" | "answer is 42"
#      → both chunks contain the key information
#    - Rule of thumb: overlap = 10% of chunk_size
# -------------------------------------------------------------------


def chunk_text(full_text: str,
               chunk_size: int = 500,
               chunk_overlap: int = 50) -> list[str]:
    """
    Split a long string of text into overlapping chunks.

    Args:
        full_text:     the complete extracted PDF text (from get_full_text())
        chunk_size:    max characters per chunk (default 500)
        chunk_overlap: characters shared between consecutive chunks (default 50)

    Returns:
        A list of strings, each one a chunk ready for embedding.
        e.g. ["Chapter 1 introduces...", "introduces the concept of...", ...]
    """

    # --- guard: make sure we actually received text ---
    if not full_text or not full_text.strip():
        raise ValueError("full_text is empty — did pdf_loader run successfully?")

    # --- create the splitter ---
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    # --- split the text ---
    chunks = splitter.split_text(full_text)

    print(f"[chunker] Input text length : {len(full_text):,} characters")
    print(f"[chunker] Chunk size        : {chunk_size} characters")
    print(f"[chunker] Chunk overlap     : {chunk_overlap} characters")
    print(f"[chunker] Total chunks made : {len(chunks)}")

    return chunks


def chunk_pages(pages_data: list[dict],
                chunk_size: int = 500,
                chunk_overlap: int = 50) -> list[dict]:
    """
    Advanced version — chunks page by page and keeps page number metadata.
    This lets you tell the user WHICH PAGE the answer came from.

    Args:
        pages_data:    list of dicts from load_pdf() —
                       each dict has {"page": int, "text": str}
        chunk_size:    max characters per chunk
        chunk_overlap: overlap between consecutive chunks

    Returns:
        A list of dicts with chunk text AND source page:
        [
          {"chunk_id": 0, "page": 1, "text": "Chapter 1 introduces..."},
          {"chunk_id": 1, "page": 1, "text": "introduces the concept..."},
          {"chunk_id": 2, "page": 2, "text": "Section 2 discusses..."},
          ...
        ]
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks = []   # collect all chunks across all pages
    chunk_id = 0      # global chunk index across the whole document

    for page in pages_data:
        # split this page's text into chunks
        page_chunks = splitter.split_text(page["text"])

        for chunk_text in page_chunks:
            # skip any chunk that is effectively empty after stripping
            if len(chunk_text.strip()) < 20:
                continue

            all_chunks.append({
                "chunk_id" : chunk_id,
                "page"     : page["page"],   # which PDF page this came from
                "text"     : chunk_text.strip()
            })
            chunk_id += 1

    print(f"[chunker] Pages processed  : {len(pages_data)}")
    print(f"[chunker] Total chunks made: {len(all_chunks)}")
    return all_chunks


def print_chunk_stats(chunks: list) -> None:
    """
    Print useful statistics about your chunks.
    Run this to check your chunk_size setting is working well.
    Good chunks: avg 300–500 chars, no chunks under 50 or over 600.
    """

    # handle both plain strings and dicts
    texts = [c["text"] if isinstance(c, dict) else c for c in chunks]
    lengths = [len(t) for t in texts]

    print(f"\n--- Chunk statistics ---")
    print(f"Total chunks : {len(lengths)}")
    print(f"Avg length   : {sum(lengths) // len(lengths)} characters")
    print(f"Min length   : {min(lengths)} characters")
    print(f"Max length   : {max(lengths)} characters")
    print(f"------------------------\n")

    # warn if chunks look bad
    tiny = sum(1 for l in lengths if l < 50)
    huge = sum(1 for l in lengths if l > 800)
    if tiny:
        print(f"[chunker] WARNING: {tiny} chunks under 50 chars — consider raising chunk_size")
    if huge:
        print(f"[chunker] WARNING: {huge} chunks over 800 chars — consider lowering chunk_size")


def save_chunks(chunks: list[dict], output_path: str) -> None:
    """
    Save chunks to a .txt file for debugging.
    Open this file to visually check your chunks look clean and complete.

    Args:
        chunks:      list of dicts from chunk_pages()
        output_path: where to save, e.g. "data/uploads/debug_chunks.txt"
    """

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(f"\n{'='*60}\n")
            f.write(f"  CHUNK {c['chunk_id']}  |  Page {c['page']}\n")
            f.write(f"{'='*60}\n\n")
            f.write(c["text"])
            f.write("\n")

    print(f"[chunker] Saved {len(chunks)} chunks to: {output_path}")


# -------------------------------------------------------------------
# QUICK TEST — run this file directly:
#   python src/chunker.py data/uploads/your_file.pdf
# -------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import os
    
    # Add parent directory to path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Import pdf_loader
    try:
        from src.pdf_loader import load_pdf
    except ImportError:
        print("❌ Could not import pdf_loader. Make sure src/pdf_loader.py exists")
        sys.exit(1)

    # Get PDF path from command line or use default
    if len(sys.argv) > 1:
        test_path = sys.argv[1]
    else:
        test_path = "data/uploads/test.pdf"

    print(f"\n--- Testing chunker with: {test_path} ---\n")

    # Check if file exists
    if not os.path.exists(test_path):
        print(f"❌ File not found: {test_path}")
        print("\nUsage: python src/chunker.py <path_to_pdf>")
        print("Example: python src/chunker.py C:/Users/hp/Downloads/python.pdf")
        sys.exit(1)

    # Load the PDF
    pages = load_pdf(test_path)

    # Chunk with page metadata
    chunks = chunk_pages(pages, chunk_size=500, chunk_overlap=50)

    # Print stats
    print_chunk_stats(chunks)

    # Preview first 3 chunks
    print("--- First 3 chunks preview ---")
    for c in chunks[:3]:
        print(f"\n[Chunk {c['chunk_id']} | Page {c['page']}]")
        print(c["text"][:200] + "..." if len(c["text"]) > 200 else c["text"])
        print()

    # Save debug file
    os.makedirs("data/uploads", exist_ok=True)
    save_chunks(chunks, "data/uploads/debug_chunks.txt")