from sentence_transformers import SentenceTransformer

# 使用中文向量模型（BAAI/bge-small-zh-v1.5）
model = SentenceTransformer('BAAI/bge-small-zh-v1.5')

def embed_texts(chunks: list[str]) -> list[list[float]]:
    return model.encode(chunks, show_progress_bar=True, convert_to_numpy=True).tolist()
