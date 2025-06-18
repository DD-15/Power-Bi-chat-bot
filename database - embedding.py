import pyodbc
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import uuid
from datetime import datetime
from itertools import islice

# Helper function to split into batches
def batched(iterable, batch_size):
    it = iter(iterable)
    while True:
        batch = list(islice(it, batch_size))
        if not batch:
            break
        yield batch

# 1. Connect to SQL Server
conn = pyodbc.connect(
    "DRIVER={ODBC Driver 18 for SQL Server};"
    "SERVER=111.93.162.34;"
    "DATABASE=testscoredb;"
    "UID=pursuit;"
    "PWD=Score@1234;"
    "TrustServerCertificate=yes;"
)

cursor = conn.cursor()

# 2. Query Your Data
cursor.execute("SELECT subReferenceID, datetime FROM ALMTable")
rows = cursor.fetchall()

# 3. Prepare text chunks
texts = []
metadatas = []

for row in rows:
    dt_value = row.datetime
    dt_str = dt_value.isoformat() if isinstance(dt_value, datetime) else str(dt_value)
    summary = f"Reference ID: {row.subReferenceID}, Datetime: {dt_str}"
    texts.append(summary)
    metadatas.append({
        "Reference ID": row.subReferenceID,
        "Datetime": dt_str
    })

# 4. Use a 768-dimensional model
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')  # ✅ 768D model
embeddings = model.encode(texts).tolist()

# 5. Initialize ChromaDB client
chroma_client = chromadb.Client(Settings(
    persist_directory="./chroma_store",
    is_persistent=True
))

# ✅ Delete previous collection to avoid embedding dimension conflicts
try:
    chroma_client.delete_collection("powerbi")
    print("🧹 Old 'powerbi' collection deleted.")
except Exception as e:
    print(f"⚠️ Could not delete collection (it might not exist): {e}")

# 6. Create a new clean collection
collection = chroma_client.get_or_create_collection("powerbi")

# 7. Prepare UUIDs
ids = [str(uuid.uuid4()) for _ in texts]

# 8. Add in batches
BATCH_SIZE = 5000

for batch_texts, batch_embs, batch_metas, batch_ids in zip(
    batched(texts, BATCH_SIZE),
    batched(embeddings, BATCH_SIZE),
    batched(metadatas, BATCH_SIZE),
    batched(ids, BATCH_SIZE)
):
    collection.add(
        documents=batch_texts,
        embeddings=batch_embs,
        metadatas=batch_metas,
        ids=batch_ids
    )

print("✅ Data embedded and stored in ChromaDB successfully with 768D vectors.")
