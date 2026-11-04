from sentence_transformers import SentenceTransformer
import chromadb

model = SentenceTransformer('all-MiniLM-L6-v2')

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="campus_docs")

question = "What are the credits for first semester?"
question_embedding = model.encode(question).tolist()

results = collection.query(
    query_embeddings=[question_embedding],
    n_results=3
)

for i, doc in enumerate(results['documents'][0]):
    print(f"--- Match {i+1} ---")
    print(doc)
    print()