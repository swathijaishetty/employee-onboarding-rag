import chromadb
import ollama

# Create a local Chroma database
client = chromadb.PersistentClient(path="chroma_db")

# Create a collection
collection = client.get_or_create_collection(name="employee_policies")

# Our first document chunk
text = "Employees are entitled to 18 days of annual leave."

# Create embedding
response = ollama.embed(
    model="nomic-embed-text",
    input=text
)

embedding = response["embeddings"][0]

# Store it in Chroma
collection.add(
    ids=["leave_annual_1"],
    documents=[text],
    embeddings=[embedding]
)

print("Document stored successfully!")
print("Number of documents:", collection.count())