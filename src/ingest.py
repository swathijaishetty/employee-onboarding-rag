from pathlib import Path

import chromadb
import ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter


# -----------------------------
# 1. Configuration
# -----------------------------

DOCUMENTS_DIR = Path("data/documents")
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "employee_policies"

EMBEDDING_MODEL = "nomic-embed-text"


# -----------------------------
# 2. Connect to Chroma
# -----------------------------

client = chromadb.PersistentClient(path=CHROMA_PATH)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME
)


# -----------------------------
# 3. Create text splitter
# -----------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)


# -----------------------------
# 4. Read and process documents
# -----------------------------

documents = []
metadatas = []
ids = []

for file_path in DOCUMENTS_DIR.glob("*.txt"):

    print(f"Processing: {file_path.name}")

    content = file_path.read_text(encoding="utf-8")

    # Split document into smaller, overlapping chunks
    chunks = splitter.split_text(content)

    for chunk_number, chunk in enumerate(chunks):

        chunk = chunk.strip()

        if not chunk:
            continue

        documents.append(chunk)

        metadatas.append({
            "source": file_path.name,
            "chunk_number": chunk_number
        })

        ids.append(
            f"{file_path.stem}_{chunk_number}"
        )


# -----------------------------
# 5. Create embeddings
# -----------------------------

embeddings = []

for i, document in enumerate(documents):

    print(f"Creating embedding {i + 1}/{len(documents)}")

    response = ollama.embed(
        model=EMBEDDING_MODEL,
        input=document
    )

    embeddings.append(
        response["embeddings"][0]
    )


# -----------------------------
# 6. Store everything in Chroma
# -----------------------------

collection.add(
    ids=ids,
    documents=documents,
    embeddings=embeddings,
    metadatas=metadatas
)


# -----------------------------
# 7. Show result
# -----------------------------

print("\nIngestion complete!")
print("Documents/chunks stored:", collection.count())