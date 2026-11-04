from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

reader = PdfReader("BCASYL.pdf")
text = ""
for page in reader.pages:
    text += page.extract_text()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_text(text)

print(f"Total chunks: {len(chunks)}")
print(f"First chunk:\n{chunks[0]}")
print(f"\nSecond chunk:\n{chunks[1]}")