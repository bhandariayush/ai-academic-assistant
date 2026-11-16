from pathlib import Path
import faiss
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter  # [web:258]

# 1) Load and split text
PDF_TEXT_PATH = Path(r"C:\Users\ayush\OneDrive\Desktop\llm\data\100.txt")

with open(PDF_TEXT_PATH, "r", encoding="utf-8") as f:
    full_text = f.read()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=200,
)
chunks = text_splitter.split_text(full_text)  # list[str]

# 2) Build embeddings + FAISS index
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")  # 384-dim embeddings [web:267][web:272]

embeddings = model.encode(chunks)  # shape: (n_chunks, 384)

dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)  # simple L2 index [web:268][web:271]
index.add(embeddings)

def answer_from_pdf(question: str, k: int = 3, max_chars: int = 600) -> str:
    """Return a short answer based on the top retrieved chunk(s)."""
    # 1) encode question and search index
    q_emb = model.encode([question])  # shape (1, dim)
    D, I = index.search(q_emb, k)     # nearest neighbors

    # 2) pick best chunks
    best_chunks = [chunks[i] for i in I[0] if i != -1]
    if not best_chunks:
        return "I could not find anything relevant in the document."

    # 3) join and truncate to keep it readable for the demo
    context = "\n\n---\n\n".join(best_chunks)
    context = context[:max_chars]
    return (
        "Here is the most relevant part from the book for your question:\n\n"
        + context
        + "\n\n(Prototype: this is retrieved context; next step is to add a summarizer.)"
    )
