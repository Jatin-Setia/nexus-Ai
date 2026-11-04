from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

sentence = "What is the credit structure for BCA first semester?"
embedding = model.encode(sentence)

print(f"Embedding length: {len(embedding)}")
print(f"First 10 values: {embedding[:10]}")