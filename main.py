# cd "C:\Users\ayush\OneDrive\Desktop\llm\backend"
# uvicorn main:app --reload

from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel

from rag_store import RAGStore

app = FastAPI()
store = RAGStore()  # global store loaded at startup


class IngestRequest(BaseModel):
    doc_id: str
    text: str


class AskRequest(BaseModel):
    question: str
    k: int = 3


@app.get("/")
def read_root():
    return {
        "status": "ok",
        "docs_indexed": len(store.metadata),
        "vectors": int(store.index.ntotal),
    }


# Old paste-text endpoint (keep it if you want)
@app.post("/ingest_text")
def ingest_text(req: IngestRequest):
    n = store.ingest_text(req.doc_id, req.text)
    return {"status": "ok", "doc_id": req.doc_id, "chunks_added": n}


# New upload-file endpoint
@app.post("/ingest_file")
async def ingest_file(doc_id: str = Form(...), file: UploadFile = File(...)):
    raw = await file.read()  # bytes from uploaded file [web:386]
    text = raw.decode("utf-8", errors="ignore")
    n = store.ingest_text(doc_id, text)
    return {
        "status": "ok",
        "doc_id": doc_id,
        "chunks_added": n,
        "filename": file.filename,
    }


@app.post("/ask")
def ask(req: AskRequest):
    results = store.query(req.question, k=req.k)
    if not results:
        return {
            "answer": "Index is empty; please ingest documents first.",
            "results": [],
        }

    max_chars = 600
    joined = "\n\n---\n\n".join(r["text"] for r in results)
    answer_text = joined[:max_chars]

    return {
        "answer": answer_text,
        "results": results,
    }
