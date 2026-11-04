from embedder import Embedder
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
import os

import indexer

pdf_files = ["BCASYL.pdf", "Ordinance11.pdf", "BCA_5th_Sem.pdf", "BCA_1st_year.pdf", "BCA_2nd_year.pdf", "N_Queen_Problem_AI_Detailed_Notes.pdf"]

splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
model = Embedder()

client = chromadb.PersistentClient(path="./chroma_db")
try:
    client.delete_collection(name="campus_docs")
except Exception:
    pass
collection = client.get_or_create_collection(name="campus_docs")

total_chunks = 0
for pdf_file in pdf_files:
    if not os.path.exists(pdf_file):
        print(f"Skipping {pdf_file} — file not found.")
        continue

    count, courses = indexer.index_pdf_into_collection(pdf_file, pdf_file, collection, model, splitter)
    total_chunks += count
    print(f"{pdf_file}: {count} chunks ({len(courses)} course codes found)")

print(f"\nTotal: {total_chunks} chunks stored from {len(pdf_files)} documents.")