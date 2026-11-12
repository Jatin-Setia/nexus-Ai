"""
Shared PDF chunking + embedding logic used by both:
- build_index.py   (full rebuild of all PDFs)
- app.py's /admin/upload endpoint (adding a single new PDF live)
"""

import re
from bisect import bisect_right
from pypdf import PdfReader

# Matches "Course Code: BCA 101T" / "CourseCode: BCA103T" etc.
COURSE_CODE_RE = re.compile(r'Course\s*Code:\s*([A-Z]{2,5})\s*(\d{2,3})\s*([A-Z]?)')


def extract_text(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


def extract_text_with_pages(pdf_path):
    """Return (full_text, page_offsets, page_numbers).
    page_offsets[i] is the character offset in full_text where page
    page_numbers[i] (1-based PDF page) begins."""
    reader = PdfReader(pdf_path)
    text = ""
    offsets, numbers = [], []
    for number, page in enumerate(reader.pages, start=1):
        offsets.append(len(text))
        numbers.append(number)
        text += (page.extract_text() or "") + "\n"
    return text, offsets, numbers


def split_by_course(text):
    """Split text at each 'Course Code:' header so every section stays
    attached to the code it belongs to. Text before the first header
    (if any) is kept with course_code=None."""
    matches = list(COURSE_CODE_RE.finditer(text))
    if not matches:
        return [(None, text)]

    sections = []
    if matches[0].start() > 0:
        sections.append((None, text[:matches[0].start()]))

    for i, m in enumerate(matches):
        code = f"{m.group(1)}{m.group(2)}{m.group(3)}"
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append((code, text[start:end]))

    return sections


def index_pdf_into_collection(pdf_path, filename, collection, embed_model, splitter, collect=None):
    """Chunk + embed a single PDF and add it to an existing Chroma collection.
    Does NOT delete or touch any existing documents in the collection.
    Each chunk is stored with its PDF page number(s).
    If `collect` is a list, a dict describing every chunk (id, source, page, page_end,
    course_code, text, embedding) is appended to it, so the caller can also save the
    chunks in the database.
    Returns (chunk_count, set_of_course_codes_found)."""
    text, page_offsets, page_numbers = extract_text_with_pages(pdf_path)
    sections = split_by_course(text)

    def page_at(offset):
        return page_numbers[max(0, bisect_right(page_offsets, offset) - 1)]

    chunk_id = 0
    chunk_count = 0
    courses_found = set()
    section_offset = 0  # sections are consecutive slices of `text`

    for course_code, section_text in sections:
        cursor = 0
        for chunk in splitter.split_text(section_text):
            pos = section_text.find(chunk, cursor)
            if pos == -1:
                pos = section_text.find(chunk[:60], cursor)
            if pos == -1:
                pos = cursor
            cursor = pos

            start = section_offset + pos
            end = start + max(len(chunk) - 1, 0)

            embedding = embed_model.encode(chunk).tolist()
            metadata = {
                "source": filename,
                "page": page_at(start),
                "page_end": page_at(end),
            }
            if course_code:
                metadata["course_code"] = course_code
                courses_found.add(course_code)

            chunk_key = f"{filename}_chunk_{chunk_id}"
            collection.add(
                ids=[chunk_key],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[metadata]
            )
            if collect is not None:
                collect.append({
                    "id": chunk_key,
                    "source": filename,
                    "page": metadata["page"],
                    "page_end": metadata["page_end"],
                    "course_code": course_code,
                    "text": chunk,
                    "embedding": embedding,
                })
            chunk_id += 1
            chunk_count += 1

        section_offset += len(section_text)

    return chunk_count, courses_found