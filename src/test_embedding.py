import ollama

response = ollama.embed(
    model="nomic-embed-text",
    input="Employees are entitled to 18 days of annual leave."
)

embedding = response["embeddings"][0]

print("Number of dimensions:", len(embedding))
print("First 10 values:", embedding[:10])