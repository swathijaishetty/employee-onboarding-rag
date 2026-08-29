import chromadb
import ollama

# Connect to our Chroma database
client = chromadb.PersistentClient(path="chroma_db")

# Get our collection
collection = client.get_collection(name="employee_policies")

# Employee's question
question = "What benefits do I get here?"
# question = "How can I check my salary details?"
# question = "When will I receive my laptop?"
# question = "Can I work from home?"
# question = "How many annual leave days do I get?"

# Convert question into an embedding
response = ollama.embed(
    model="nomic-embed-text",
    input=question
)

question_embedding = response["embeddings"][0]

# Retrieve the 3 most relevant chunks
results = collection.query(
    query_embeddings=[question_embedding],
    n_results=3
)

print("\nQuestion:", question)

for i in range(3):
    print(f"\n--- Result {i + 1} ---")
    print("Source:", results["metadatas"][0][i]["source"])
    print("Chunk:", results["metadatas"][0][i]["chunk_number"])
    print("Distance:", results["distances"][0][i])
    print("Text:", results["documents"][0][i])



# import chromadb
# import ollama

# # Connect to our existing Chroma database
# client = chromadb.PersistentClient(path="chroma_db")

# # Get our existing collection
# collection = client.get_collection(name="employee_policies")

# # The employee's question
# question = "How many vacation days do I get?"

# # Convert the question into an embedding
# response = ollama.embed(
#     model="nomic-embed-text",
#     input=question
# )

# question_embedding = response["embeddings"][0]

# # Search Chroma for the most relevant chunk
# results = collection.query(
#     query_embeddings=[question_embedding],
#     n_results=1
# )

# print("Question:", question)
# print("\nRetrieved document:")
# print(results["documents"][0][0])

# print("\nDistance:")
# print(results["distances"][0][0])