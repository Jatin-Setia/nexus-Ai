import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb
from google import genai

load_dotenv()
client_genai = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

model = SentenceTransformer('all-MiniLM-L6-v2')
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="campus_docs")

question = input("Ask a question about the BCA syllabus: ")

question_embedding = model.encode(question).tolist()
results = collection.query(
    query_embeddings=[question_embedding],
    n_results=5,
    where_document={"$contains": "THIRD SEMESTER"}
)
for i, doc in enumerate(results['documents'][0]):
    print(f"\n=== Retrieved chunk {i+1} ===")
    print(doc[:300])
context = "\n\n".join(results['documents'][0])

prompt = f"""Answer the question using only the context below. If the answer isn't in the context, say so.

Context:
{context}

Question: {question}

Answer:"""

response = client_genai.models.generate_content(
    model='gemini-3.6-flash',
    contents=prompt
)

print("\n--- Answer ---")
print(response.text)