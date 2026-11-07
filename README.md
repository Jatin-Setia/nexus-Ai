# NEXUS AI — Document Assistant

A RAG (retrieval-augmented generation) chatbot. Upload any PDFs and ask questions; it answers straight from their content and tells you **which file and page** each answer came from.

**Live demo:** https://nexus-ai-vxpd.onrender.com/app

> The demo runs on a free hosting tier. If it has been idle, the first load can take about a minute while it wakes up. Accounts reset whenever the server restarts, so just sign up again.

<!-- Add a screenshot at docs/screenshot.png, then uncomment the next line -->
<!-- ![NEXUS AI screenshot](docs/screenshot.png) -->

## What it does

- Answers questions such as *"Summarize section 3 of the report"* or *"What does the handbook say about refunds?"* from your PDFs.
- **Cites its sources inline**, e.g. `[report.pdf, p. 3]`, and returns a structured `citations` list from the API.
- Understands follow-up questions by rewriting them into standalone questions using the chat history.
- Study tools: notes, quizzes, flashcards and explanations generated from the retrieved documents.
- Accounts with login and signup, per-user rate limits, and an admin panel to upload or delete PDFs and change branding.
- Gemini-powered image generation, plus voice input and read-aloud in the browser.

## How retrieval works

1. **Indexing:** each PDF is split into overlapping chunks (documents with "Course Code:" headers are split at those headers first) (1500 characters, 200 overlap). Every chunk is stored with its `source`, `page`, `page_end` and `course_code`.
2. **Hybrid search:** the question is embedded and searched in ChromaDB (vector search), and also searched with BM25 (keyword search). The BM25 tokenizer also recognises structured codes (for example `BCA 101T`, `bca101t` and `BCA101` all match), which helps with documents that use course or part codes.
3. **Fusion:** both ranked lists are merged with Reciprocal Rank Fusion.
4. **Reranking (optional):** a cross-encoder re-scores the merged candidates and keeps the best 6. This is switched off on the free demo to save memory (`RERANK=0`).
5. **Exact lookups (optional):** if a document uses course codes (such as a syllabus), questions that name a code or semester also pull in the matching chunks directly from the metadata.
6. **Answer:** the chunks go to Gemini with `Source | Page` headers, and the model is told to cite only pages it was given.

## Tech stack

| Part | Tools |
|------|-------|
| Backend | Python, Flask, Gunicorn |
| Vector database | ChromaDB |
| Embeddings | `all-MiniLM-L6-v2` via fastembed (ONNX, no PyTorch) |
| Keyword search | BM25, implemented from scratch in `retrieval.py` |
| LLM | Google Gemini (automatic fallback across models) |
| PDF parsing | pypdf |
| Frontend | Single-page HTML, CSS and vanilla JavaScript |
| Hosting | Render (free tier) |

## Run it locally

```bash
git clone https://github.com/Jatin-Setia/nexus-Ai.git
cd nexus-Ai
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project folder:

```env
GEMINI_API_KEY=your_key_here
SECRET_KEY=a_long_random_string
ADMIN_USERNAME=admin
ADMIN_PASSWORD=choose_a_strong_password
```

Put your PDFs in the project folder, build the index, then start the app (you can also upload PDFs later from the admin panel):

```bash
python3 build_index.py        # indexes every PDF in the folder
python3 app.py
```

Open http://127.0.0.1:5001/app

To turn on the optional reranker locally: `pip install sentence-transformers`.

## Configuration

| Variable | Purpose |
|----------|---------|
| `GEMINI_API_KEY` | Gemini API key (required) |
| `SECRET_KEY` | Signs login tokens; set it so sessions survive restarts |
| `ADMIN_USERNAME`, `ADMIN_PASSWORD` | Admin account created on first start |
| `RERANK` | Set to `0` to disable the reranker (saves memory) |
| `ALLOW_SIGNUP` | Set to `false` to disable new signups |
| `AI_PER_MINUTE`, `AI_PER_DAY` | Per-user AI request limits |

## Project structure

```
app.py          Flask API: auth, /ask, /study-session, admin, image generation
retrieval.py    Hybrid search (vector + BM25 + RRF), reranking, citations
indexer.py      PDF chunking with page numbers and course codes
embedder.py     fastembed wrapper used for all embeddings
build_index.py  Rebuilds the vector index from the PDFs
index.html      Frontend
```

## Deployment notes

Deployed on Render's free tier (512 MB RAM), so the app is trimmed to fit: fastembed instead of PyTorch, the reranker disabled, and the index built during deploy. The free tier has no persistent disk, so accounts and admin uploads reset when the server restarts.

## Author

**Jatin Setia** — BCA student at DBIT (GGSIPU), Delhi.