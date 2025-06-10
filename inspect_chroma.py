import chromadb
from chromadb.config import Settings

# Connect to existing ChromaDB
chroma_client = chromadb.Client(Settings(
    persist_directory="./chroma_store",
    is_persistent=True
))

# Load the collection
collection = chroma_client.get_or_create_collection("powerbi")

# Count how many items are stored
print(f"Total documents in collection: {collection.count()}")

# (Optional) Fetch first 3 documents
results = collection.get(limit=3)
for i, doc in enumerate(results['documents']):
    print(f"\n--- Document {i+1} ---")
    print("ID:", results['ids'][i])
    print("Metadata:", results['metadatas'][i])
    print("Document:", doc)
