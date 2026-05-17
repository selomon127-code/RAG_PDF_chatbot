# src/embedder.py
# ChromaDB vector store with SentenceTransformer embeddings
# Handles embedding, indexing, and semantic search for RAG pipeline

import os
import json
import shutil  # ✅ ADDED: Required for reset() method
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
PERSIST_DIR = "data/vectorstore"


class VectorStore:
    """
    ChromaDB wrapper for persistent vector storage and semantic search.
    Uses SentenceTransformer for local, offline embedding generation.
    """
    
    def __init__(self, persist_directory: str = PERSIST_DIR):
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize ChromaDB persistent client with reset support
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=chromadb.Settings(anonymized_telemetry=False, allow_reset=True)
        )
        
        # Create embedding function using SentenceTransformer
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=MODEL_NAME,
            device="cpu"  # Use "cuda" if GPU available
        )
        
        self.collection = None
        print(f"[embedder] ✓ ChromaDB initialized at: {persist_directory}")
    
    def create_collection(self, chunks: list[dict], collection_name: str = "pdf_chunks") -> chromadb.Collection:
        """
        Create a new collection and add chunks with embeddings.
        
        Args:
            chunks: List of dicts with keys: chunk_id, page, text
            collection_name: Name for the ChromaDB collection
            
        Returns:
            The created ChromaDB collection object
        """
        # Delete existing collection if present (for fresh indexing)
        try:
            self.client.delete_collection(collection_name)
            print(f"[embedder] 🗑️ Deleted existing collection: {collection_name}")
        except Exception:
            pass
        
        # Create new collection with cosine similarity and embedding function
        self.collection = self.client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=self.embedding_function
        )
        
        # Prepare data for batch insertion
        texts = [chunk["text"].strip() for chunk in chunks if chunk.get("text", "").strip()]
        ids = [str(chunk["chunk_id"]) for chunk in chunks if chunk.get("text", "").strip()]
        metadatas = [
            {"page": chunk["page"], "chunk_id": chunk["chunk_id"]} 
            for chunk in chunks if chunk.get("text", "").strip()
        ]
        
        if not texts:
            raise ValueError("No valid text chunks found for embedding")
        
        # Add in batches to avoid memory issues
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            self.collection.add(
                documents=texts[i:i+batch_size],
                ids=ids[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size]
            )
            print(f"[embedder] 📦 Indexed batch {i//batch_size + 1}/{(len(texts)+batch_size-1)//batch_size}")
        
        print(f"[embedder] ✅ Added {len(texts)} chunks to ChromaDB collection '{collection_name}'")
        return self.collection
    
    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Perform semantic search using the query text.
        
        Args:
            query: The search query string
            top_k: Number of results to return
            
        Returns:
            List of dicts with keys: chunk_id, page, text, score
        """
        if not self.collection:
            raise ValueError("Collection not created or loaded. Call create_collection() or load_collection() first.")
        
        if not query or not query.strip():
            return []
        
        # ChromaDB handles embedding internally via embedding_function
        results = self.collection.query(
            query_texts=[query.strip()],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results for RAG pipeline
        formatted_results = []
        if results and results.get("ids") and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                distance = results["distances"][0][i] if results.get("distances") else 0
                formatted_results.append({
                    "chunk_id": int(results["ids"][0][i]),
                    "page": results["metadatas"][0][i].get("page", "Unknown"),
                    "text": results["documents"][0][i],
                    "score": round(1 - distance, 4) if distance is not None else 0.0
                })
        
        return formatted_results
    
    def load_collection(self, collection_name: str = "pdf_chunks") -> chromadb.Collection | None:
        """
        Load an existing collection from disk.
        
        Args:
            collection_name: Name of the collection to load
            
        Returns:
            The loaded ChromaDB collection, or None if not found
        """
        try:
            # Check if collection exists first
            existing = self.client.list_collections()
            if not any(c.name == collection_name for c in existing):
                print(f"[embedder] ⚠️ Collection '{collection_name}' not found in: {[c.name for c in existing]}")
                return None
                
            self.collection = self.client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
            count = self.collection.count()
            print(f"[embedder] ✅ Loaded collection '{collection_name}' with {count} chunks")
            return self.collection
        except Exception as e:
            print(f"[embedder] ⚠️ Collection '{collection_name}' not found: {e}")
            return None
    
    def get_stats(self) -> dict:
        """Return collection statistics for debugging."""
        if not self.collection:
            return {"status": "not_loaded"}
        return {
            "name": self.collection.name,
            "count": self.collection.count(),
            "metadata": getattr(self.collection, "metadata", {})
        }
    
    def reset(self):
        """
        ✅ ADDED: Clear the entire vector store directory and reset client state.
        Called by app.py during recovery from indexing errors.
        """
        print(f"[embedder] 🔄 Resetting vector store at: {self.persist_directory}")
        
        # Delete the entire persistence directory
        if os.path.exists(self.persist_directory):
            try:
                shutil.rmtree(self.persist_directory)
                print(f"[embedder] 🗑️ Deleted: {self.persist_directory}")
            except Exception as e:
                print(f"[embedder] ⚠️ Could not delete directory: {e}")
        
        # Recreate empty directory
        os.makedirs(self.persist_directory, exist_ok=True)
        print(f"[embedder] 📁 Recreated: {self.persist_directory}")
        
        # Reinitialize ChromaDB client with fresh state
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=chromadb.Settings(anonymized_telemetry=False, allow_reset=True)
        )
        self.collection = None
        print(f"[embedder] ✅ Vector store reset complete")


def load_embedding_model(model_name: str = MODEL_NAME) -> SentenceTransformer:
    """
    Load the SentenceTransformer embedding model.
    
    Args:
        model_name: HuggingFace model identifier
        
    Returns:
        Loaded SentenceTransformer model
    """
    print(f"[embedder] 🤖 Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name, device="cpu", trust_remote_code=True)
    
    # ✅ FIXED: Use correct method name for newer sentence-transformers versions
    # get_sentence_embedding_dimension() is the correct method (not get_embedding_dimension)
    try:
        dim = model.get_sentence_embedding_dimension()
    except AttributeError:
        # Fallback for older versions
        dim = model.get_embedding_dimension()
    
    print(f"[embedder] ✅ Model loaded | Dimensions: {dim}")
    return model


def build_vectorstore(chunks: list[dict], model_name: str = MODEL_NAME, persist_dir: str = PERSIST_DIR):
    """
    Complete pipeline: load model → create ChromaDB collection → save metadata.
    
    Args:
        chunks: List of chunk dicts from chunker
        model_name: Embedding model to use
        persist_dir: Directory for ChromaDB persistence
        
    Returns:
        Tuple: (embedding_model, vector_store, chunks_metadata)
    """
    # Ensure persist directory exists
    os.makedirs(persist_dir, exist_ok=True)
    
    # Load embedding model (kept for compatibility, ChromaDB handles embedding internally)
    model = load_embedding_model(model_name)
    
    # Initialize and populate vector store
    vector_store = VectorStore(persist_directory=persist_dir)
    vector_store.create_collection(chunks)
    
    # Save chunk metadata for source citation
    metadata_path = os.path.join(persist_dir, "chunks_metadata.json")
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    
    print(f"[embedder] 💾 Saved metadata to: {metadata_path}")
    print(f"[embedder] 🎉 Vector store built successfully!")
    
    return model, vector_store, chunks


def load_index(persist_dir: str = PERSIST_DIR):
    """
    Load existing vector store and metadata from disk.
    
    Args:
        persist_dir: Directory where ChromaDB data is stored
        
    Returns:
        Tuple: (embedding_model, vector_store, chunks_metadata) or (None, None, None) if not found
    """
    os.makedirs(persist_dir, exist_ok=True)
    
    vector_store = VectorStore(persist_directory=persist_dir)
    collection = vector_store.load_collection()
    
    if not collection:
        print("[embedder] ⚠️ No existing index found. Call build_vectorstore() first.")
        return None, None, None
    
    # Load saved metadata
    metadata_path = os.path.join(persist_dir, "chunks_metadata.json")
    if not os.path.exists(metadata_path):
        print(f"[embedder] ⚠️ Metadata file not found: {metadata_path}")
        return None, vector_store, None
    
    with open(metadata_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    
    # Load model for potential future use
    model = load_embedding_model()
    
    print(f"[embedder] ✅ Index loaded: {vector_store.get_stats()}")
    return model, vector_store, chunks


def search_index(query: str, vector_store: VectorStore, top_k: int = 5) -> list[dict]:
    """
    Convenience function for semantic search.
    
    Args:
        query: Search query string
        vector_store: Initialized VectorStore object
        top_k: Number of results
        
    Returns:
        List of relevant chunk dicts
    """
    return vector_store.search(query=query, top_k=top_k)


# ============================================================================
# TESTING
# ============================================================================
if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from src.pdf_loader import load_pdf
    from src.chunker import chunk_pages
    
    test_path = sys.argv[1] if len(sys.argv) > 1 else "data/uploads/test.pdf"
    print(f"\n🧪 Testing embedder with: {test_path}\n")
    
    if not os.path.exists(test_path):
        print(f"❌ File not found: {test_path}")
        sys.exit(1)
    
    # Process PDF
    pages = load_pdf(test_path)
    chunks = chunk_pages(pages, chunk_size=500, chunk_overlap=50)
    
    # Build vector store
    model, store, saved_chunks = build_vectorstore(chunks)
    
    # Test search
    print("\n🔍 Testing semantic search...")
    test_queries = [
        "What is the main topic?",
        "Explain key concepts",
        "What are the conclusions?"
    ]
    
    for query in test_queries:
        print(f"\n💬 Query: '{query}'")
        results = store.search(query=query, top_k=2)
        for r in results:
            print(f"   📄 Page {r['page']} | Score: {r['score']:.3f}")
            print(f"   📝 {r['text'][:120]}...")
    
    print("\n✅ Embedder test complete!")