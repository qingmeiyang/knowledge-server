import os
from docx import Document
from pdfminer.high_level import extract_text as extract_pdf_text

def extract_text_from_docx(content: bytes) -> str:
    from io import BytesIO
    doc = Document(BytesIO(content))
    return '\n'.join([p.text for p in doc.paragraphs if p.text.strip() != ''])

def extract_text_from_pdf(content: bytes) -> str:
    from io import BytesIO
    return extract_pdf_text(BytesIO(content))

def extract_text_from_txt(content: bytes) -> str:
    return content.decode('utf-8', errors='ignore')

def load_file_and_ingest(content: bytes, filename: str) -> dict:
    if filename.endswith('.docx'):
        text = extract_text_from_docx(content)
    elif filename.endswith('.pdf'):
        text = extract_text_from_pdf(content)
    elif filename.endswith('.txt'):
        text = extract_text_from_txt(content)
    else:
        return {"status": "error", "message": f"Unsupported file type: {filename}"}

    from ingest.splitter import split_text
    from ingest.embedder import embed_texts
    from vector_store.chroma_store import save_embeddings

    chunks = split_text(text)
    embeddings = embed_texts(chunks)
    save_embeddings(chunks, embeddings)

    return {"status": "success", "chunks": len(chunks)}
