from embedder import Embedder
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
import os

import indexer

# Index every PDF in the project folder (or pass file names: python3 build_index.py a.pdf b.pdf)
import sys
pdf_files = sys.argv[1:] or sorted(f for f in os.listdir(".") if f.lower().endswith(".pdf"))

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

    count, _ = indexer.index_pdf_into_collection(pdf_file, pdf_file, collection, model, splitter)
    total_chunks += count
    print(f"{pdf_file}: {count} chunks ")

print(f"\nTotal: {total_chunks} chunks stored from {len(pdf_files)} documents.")