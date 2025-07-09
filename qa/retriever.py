from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# 加载和配置向量模型（与 embedder.py 一致）
embedding_model = SentenceTransformer("BAAI/bge-small-zh-v1.5")

# 初始化 Chroma 本地客户端
client = chromadb.PersistentClient(path="data")


collection = client.get_or_create_collection("local_knowledge")


def query_knowledge(question: str, top_k: int = 3) -> str:
    # 1. 将问题向量化
    query_vector = embedding_model.encode(question).tolist()

    # 2. 检索最相似向量
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k
    )

    # 3. 返回原文片段（列表）
    docs = results.get("documents", [[]])[0]

    # 4. 拼接成统一回答返回
    answer = "\n---\n".join(docs)
    return answer or "未找到相关内容。"
