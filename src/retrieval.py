import chromadb
import ollama

from config import (
    CHROMA_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    LLM_MODEL,
    DISTANCE_THRESHOLD,
    QUERY_REWRITE_COUNT,
)


# Connect to Chroma
client = chromadb.PersistentClient(
    path=CHROMA_PATH
)

collection = client.get_collection(
    name=COLLECTION_NAME
)


def rewrite_query(question):
    """
    Generate alternative search queries using the LLM.
    """

    prompt = f"""
Rewrite the following user question into
{QUERY_REWRITE_COUNT} different search queries.

The queries should preserve the original meaning
while using different natural wording.

Return ONLY the queries, one per line.
Do not number them.
Do not explain anything.

User question:
{question}
"""

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    rewritten_queries = [
        line.strip()
        for line in response["message"]["content"].splitlines()
        if line.strip()
    ]

    # Always include the original question
    queries = [question]

    for query in rewritten_queries:
        if query not in queries:
            queries.append(query)

    return queries[:QUERY_REWRITE_COUNT + 1]


def retrieve_context(question, n_results=5):
    """
    Retrieve relevant document chunks using
    the original question and rewritten queries.
    """

    queries = rewrite_query(question)

    # Store the best distance found for each chunk
    retrieved = {}

    for query in queries:

        # Create embedding for the search query
        response = ollama.embed(
            model=EMBEDDING_MODEL,
            input=query
        )

        query_embedding = response["embeddings"][0]

        # Search Chroma
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )

        for document, metadata, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):

            # Use source + chunk number as unique identifier
            chunk_id = (
                f"{metadata['source']}_"
                f"{metadata['chunk_number']}"
            )

            # Keep only the best distance for this chunk
            if (
                chunk_id not in retrieved
                or distance < retrieved[chunk_id]["distance"]
            ):
                retrieved[chunk_id] = {
                    "document": document,
                    "source": metadata["source"],
                    "distance": distance
                }

    # Sort by similarity
    relevant_results = [
        result
        for result in retrieved.values()
        if result["distance"] <= DISTANCE_THRESHOLD
    ]

    relevant_results.sort(
        key=lambda result: result["distance"]
    )

    # Return best 3 chunks
    return relevant_results[:3]