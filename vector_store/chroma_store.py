import chromadb
from chromadb.config import Settings

client = chromadb.PersistentClient(path="data")


collection = client.get_or_create_collection("local_knowledge")

def save_embeddings(texts: list[str], embeddings: list[list[float]]):
    ids = [f"id_{i}" for i in range(len(texts))]
    collection.add(documents=texts, embeddings=embeddings, ids=ids)
