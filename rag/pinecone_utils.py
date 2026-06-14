import os
import time
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

from config import (
    PINECONE_API_KEY,
    INDEX_NAME,
    DIMENSION,
    METRIC,
    CLOUD,
    REGION,
    EMBED_MODEL_NAME,
    TOP_K,
)


# ─────────────────────────────────────────────
# 1. Connect to Pinecone
# ─────────────────────────────────────────────

def init_pinecone() -> Pinecone:
    """
    Initialize and return a Pinecone client.
    Reads API key from config (which reads from environment).
    """
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY is not set. Check your .env or Colab Secrets.")

    pc = Pinecone(api_key=PINECONE_API_KEY)
    print(f"[Pinecone] Connected successfully.")
    return pc


# ─────────────────────────────────────────────
# 2. Create index if it does not exist
# ─────────────────────────────────────────────

def get_or_create_index(pc: Pinecone):
    """
    Create a serverless Pinecone index if it does not already exist.
    Returns the index object ready for upsert/query.

    Index settings:
      dimension = 384  (all-MiniLM-L6-v2 output size)
      metric    = cosine
      cloud     = aws / us-east-1  (free tier)
    """
    if not pc.has_index(INDEX_NAME):
        print(f"[Pinecone] Index '{INDEX_NAME}' not found. Creating...")

        pc.create_index(
            name=INDEX_NAME,
            dimension=DIMENSION,
            metric=METRIC,
            spec=ServerlessSpec(cloud=CLOUD, region=REGION),
        )

        # Wait until the index is ready before returning
        while not pc.describe_index(INDEX_NAME).status["ready"]:
            print("[Pinecone] Waiting for index to be ready...")
            time.sleep(2)

        print(f"[Pinecone] Index '{INDEX_NAME}' created and ready.")
    else:
        print(f"[Pinecone] Index '{INDEX_NAME}' already exists. Skipping creation.")

    return pc.Index(INDEX_NAME)


# ─────────────────────────────────────────────
# 3. Load embedding model
# ─────────────────────────────────────────────

def build_embedding_model() -> SentenceTransformer:
    """
    Load the sentence embedding model used to convert text → 384-dim vectors.
    Must be the same model for both indexing and querying.
    """
    print(f"[Embedding] Loading model: {EMBED_MODEL_NAME}")
    model = SentenceTransformer(EMBED_MODEL_NAME)
    print(f"[Embedding] Model loaded. Output dimension: {model.get_sentence_embedding_dimension()}")
    return model


# ─────────────────────────────────────────────
# 4. Index documents
# ─────────────────────────────────────────────

def index_documents(texts: list[str], index, embed_model: SentenceTransformer, batch_size: int = 100):
    """
    Embed a list of text strings and upsert them into the Pinecone index.

    Skips indexing if vectors already exist in the index (idempotent).
    Uses batched upsert to avoid memory issues with large datasets.

    Args:
        texts:       list of financial sentences/news to index
        index:       Pinecone index object
        embed_model: SentenceTransformer model
        batch_size:  number of vectors to upsert per API call (max 100)
    """
    # Check if index already has data — skip if so
    stats = index.describe_index_stats()
    existing_count = stats.get("total_vector_count", 0)

    if existing_count > 0:
        print(f"[Pinecone] Index already contains {existing_count} vectors. Skipping indexing.")
        return

    print(f"[Pinecone] Embedding {len(texts)} documents...")
    embeddings = embed_model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    # Build records: list of (id, vector, metadata)
    records = [
        {
            "id":       f"doc_{i}",
            "values":   embeddings[i].tolist(),   # Pinecone requires plain list, not numpy
            "metadata": {"text": text},
        }
        for i, text in enumerate(texts)
    ]

    # Upsert in batches of batch_size
    total_batches = (len(records) + batch_size - 1) // batch_size

    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end   = start + batch_size
        batch = records[start:end]
        index.upsert(vectors=batch)
        print(f"[Pinecone] Upserted batch {batch_idx + 1}/{total_batches} ({len(batch)} vectors)")

    # Verify
    final_count = index.describe_index_stats().get("total_vector_count", 0)
    print(f"[Pinecone] Indexing complete. Total vectors in index: {final_count}")


# ─────────────────────────────────────────────
# 5. Query index
# ─────────────────────────────────────────────

def query_index(query: str, index, embed_model: SentenceTransformer, top_k: int = TOP_K) -> list[str]:
    """
    Embed a query string and retrieve the top-k most semantically similar texts.

    Args:
        query:       input sentence to search for
        index:       Pinecone index object
        embed_model: same SentenceTransformer used during indexing
        top_k:       number of results to return

    Returns:
        list of text strings ranked by cosine similarity (highest first)
    """
    query_vector = embed_model.encode([query], convert_to_numpy=True)[0].tolist()

    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
    )

    retrieved = [match.metadata["text"] for match in results.matches]
    return retrieved


# ─────────────────────────────────────────────
# 6. Main initializer — call this from other files
# ─────────────────────────────────────────────

def initialize_rag(texts: list[str]):
    """
    Full RAG setup in one call:
      1. Connect to Pinecone
      2. Get or create index
      3. Load embedding model
      4. Index documents (skipped if already indexed)

    Args:
        texts: list of financial sentences to use as RAG knowledge base

    Returns:
        (index, embed_model) — pass these to retriever.py
    """
    pc          = init_pinecone()
    index       = get_or_create_index(pc)
    embed_model = build_embedding_model()
    index_documents(texts, index, embed_model)

    return index, embed_model