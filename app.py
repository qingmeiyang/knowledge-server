from fastapi import FastAPI, UploadFile, File, Form
from qa.retriever import query_knowledge
from ingest.file_loader import load_file_and_ingest

app = FastAPI()

@app.post("/ingest")
async def ingest_knowledge(file: UploadFile = File(...)):
    file_content = await file.read()
    filename = file.filename
    return load_file_and_ingest(file_content, filename)

@app.post("/query")
def query_qa(question: str = Form(...)):
    answer = query_knowledge(question)
    return {"answer": answer}
