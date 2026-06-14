from sentence_transformers import SentenceTransformer
from config import TOP_K


# ─────────────────────────────────────────────
# 1. Retrieve relevant context from Pinecone
# ─────────────────────────────────────────────

def retrieve_context(
    query: str,
    index,
    embed_model: SentenceTransformer,
    top_k: int = TOP_K,
) -> list[str]:

    query_vector = embed_model.encode([query], convert_to_numpy=True)[0].tolist()

    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
    )

    retrieved = [match.metadata["text"] for match in results.matches]

    print(f"[Retriever] Query: '{query[:60]}...' " if len(query) > 60 else f"[Retriever] Query: '{query}'")
    print(f"[Retriever] Retrieved {len(retrieved)} context chunks.")

    return retrieved


# ─────────────────────────────────────────────
# 2. Build RAG prompt
# ─────────────────────────────────────────────

def build_rag_prompt(sentence: str, context_list: list[str]) -> str:
   
    # Join all context chunks with separator
    context_block = "\n\n".join(
        f"[{i + 1}] {text}" for i, text in enumerate(context_list)
    )

    prompt = (
        "You are a financial sentiment analysis expert.\n\n"
        "Use the following context from financial news to help you classify the sentiment:\n\n"
        f"{context_block}\n\n"
        "---\n\n"
        "Based on the context above, classify the sentiment of this sentence as "
        "positive, neutral, or negative:\n\n"
        f"Sentence: {sentence}\n\n"
        "Answer:"
    )

    return prompt


# ─────────────────────────────────────────────
# 3. Build plain prompt (no RAG — used as baseline)
# ─────────────────────────────────────────────

def build_plain_prompt(sentence: str) -> str:
    """
    Build a simple prompt without any retrieved context.
    Used when RAG is disabled or for baseline comparison.

    Args:
        sentence: the financial sentence to classify

    Returns:
        formatted prompt string
    """
    prompt = (
        "You are a financial sentiment analysis expert.\n\n"
        "Classify the sentiment of this sentence as positive, neutral, or negative:\n\n"
        f"Sentence: {sentence}\n\n"
        "Answer:"
    )

    return prompt


# ─────────────────────────────────────────────
# 4. Full RAG pipeline — retrieve + build prompt
# ─────────────────────────────────────────────

def get_rag_prompt(
    sentence: str,
    index,
    embed_model: SentenceTransformer,
    top_k: int = TOP_K,
) -> tuple[str, list[str]]:
    """
    Full RAG pipeline in one call:
      1. Retrieve top-k relevant context from Pinecone
      2. Build augmented prompt with context

    Args:
        sentence:    the financial sentence to analyze
        index:       Pinecone index object
        embed_model: SentenceTransformer model
        top_k:       number of context chunks to retrieve

    Returns:
        (prompt, context_list)
          prompt:       augmented prompt string → pass to model tokenizer
          context_list: raw retrieved texts → display in Gradio UI
    """
    context_list = retrieve_context(sentence, index, embed_model, top_k)
    prompt       = build_rag_prompt(sentence, context_list)

    return prompt, context_list