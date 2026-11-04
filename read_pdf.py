from pypdf import PdfReader

reader = PdfReader("BCASYL.pdf")
text = ""
for page in reader.pages:
    text += page.extract_text()

print(f"Total pages: {len(reader.pages)}")
print(f"First 500 characters:\n{text[:500]}")