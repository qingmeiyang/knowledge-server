from sentence_transformers import SentenceTransformer
import chromadb

# 加载中文向量模型（与 ingest 一致）
embedding_model = SentenceTransformer("BAAI/bge-small-zh-v1.5")

# 初始化持久化向量数据库客户端
client = chromadb.PersistentClient(path="data")
collection = client.get_or_create_collection("local_knowledge")


def query_knowledge(question: str, top_k: int = 3) -> dict:
    # 向量化查询问题
    query_vector = embedding_model.encode(question).tolist()

    # 从向量库检索相关文段
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "distances"]
    )

    documents = results.get("documents", [[]])[0]
    distances = results.get("distances", [[]])[0]

    # 格式化结果
    chunks = []
    for i, doc in enumerate(documents):
        chunks.append({
            "content": doc,
            "score": round(1 - distances[i], 4)  # 将距离转为相似度
        })

    # 拼接片段构成答案
    answer = "\n".join([c["content"] for c in chunks])

    return {
        "question": question,
        "matched_chunks": chunks,
        "answer": answer
    }
