"""
Hybrid retrieval for NEXUS AI:
  1. vector search (Chroma) + keyword search (BM25) over the same chunks
  2. merge both rankings with Reciprocal Rank Fusion (RRF)
  3. rerank the merged candidates with a cross-encoder
  4. helpers that turn chunk metadata into page-number citations
"""
import math
import os
import re
from collections import Counter, defaultdict
import math
import re
from collections import Counter, defaultdict

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RRF_K = 60  # standard RRF constant

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "of", "in", "on", "to", "for",
    "and", "or", "what", "which", "who", "how", "do", "does", "me", "my", "i",
    "tell", "about", "give", "show", "please", "with", "at", "by", "from", "it",
}

# "BCA 101T" / "bca101 t" / "BCA101" -> "bca101t" / "bca101t" / "bca101"
_CODE_RE = re.compile(r"\bbca\s?(\d{3})\s?([tp])?\b")


def tokenize(text):
    text = text.lower()
    text = _CODE_RE.sub(lambda m: f"bca{m.group(1)}{m.group(2) or ''}", text)
    return [t for t in re.findall(r"[a-z0-9]+", text) if t not in STOPWORDS]


def _query_tokens(query):
    tokens = tokenize(query)
    # a bare "bca101" should also match the T and P versions in the index
    for t in list(tokens):
        if re.fullmatch(r"bca\d{3}", t):
            tokens += [t + "t", t + "p"]
    return tokens


class BM25:
    def __init__(self, tokenized_docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in tokenized_docs]
        self.doc_len = [len(d) for d in tokenized_docs]
        self.avg_len = (sum(self.doc_len) / len(self.doc_len)) if self.doc_len else 1.0
        self.postings = defaultdict(list)
        for i, counts in enumerate(self.tf):
            for term in counts:
                self.postings[term].append(i)
        n = len(tokenized_docs)
        self.idf = {
            term: math.log(1 + (n - len(ids) + 0.5) / (len(ids) + 0.5))
            for term, ids in self.postings.items()
        }

    def top(self, query_tokens, k):
        scores = defaultdict(float)
        for term in set(query_tokens):
            if term not in self.idf:
                continue
            for i in self.postings[term]:
                f = self.tf[i][term]
                norm = 1 - self.b + self.b * self.doc_len[i] / self.avg_len
                scores[i] += self.idf[term] * f * (self.k1 + 1) / (f + self.k1 * norm)
        return sorted(scores, key=scores.get, reverse=True)[:k]


# ---- corpus cache (rebuilt automatically when chunks are added/deleted) ----
_corpus = {"key": None, "ids": [], "docs": [], "metas": [], "bm25": None}


def _load_corpus(collection):
    ids = collection.get(include=[])["ids"]
    key = (len(ids), hash(tuple(ids)))
    if key != _corpus["key"]:
        data = collection.get(include=["documents", "metadatas"])
        _corpus.update(
            key=key,
            ids=data["ids"],
            docs=data["documents"],
            metas=data["metadatas"],
            # prepend the course code so every chunk of a course matches code queries
            bm25=BM25([
                tokenize(f"{m.get('course_code', '')} {d}")
                for d, m in zip(data["documents"], data["metadatas"])
            ]),
        )
    return _corpus


# ---- reranker (loaded once, on first use) ----
_reranker = {"model": None, "failed": False}


def _get_reranker():
    if os.environ.get("RERANK", "1") == "0":
        return None
    if _reranker["model"] is None and not _reranker["failed"]:
        try:
            from sentence_transformers import CrossEncoder
            _reranker["model"] = CrossEncoder(RERANK_MODEL)
        except Exception as e:
            print(f"[rerank] model unavailable, skipping reranking: {e}")
            _reranker["failed"] = True
    return _reranker["model"]


def hybrid_search(collection, query, query_embedding, top_k=6, candidates=20):
    """Return (docs, metas) for the best `top_k` chunks."""
    print("[search] loading corpus", flush=True)
    corpus = _load_corpus(collection)
    print(f"[search] corpus ready, {len(corpus['ids'])} chunks", flush=True)
    if not corpus["ids"]:
        return [], []

    index_of = {cid: i for i, cid in enumerate(corpus["ids"])}

    vec = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(candidates, len(corpus["ids"])),
    )
    vector_ids = vec["ids"][0]
    print("[search] vector query done", flush=True)
    keyword_ids = [corpus["ids"][i] for i in corpus["bm25"].top(_query_tokens(query), candidates)]

    print("[search] keyword search done", flush=True)
    fused = defaultdict(float)
    for ranking in (vector_ids, keyword_ids):
        for rank, cid in enumerate(ranking):
            fused[cid] += 1.0 / (RRF_K + rank + 1)
    merged = sorted(fused, key=fused.get, reverse=True)[:candidates]

    reranker = _get_reranker()
    if reranker is not None and len(merged) > 1:
        scores = reranker.predict([(query, corpus["docs"][index_of[cid]]) for cid in merged])
        merged = [cid for cid, _ in sorted(zip(merged, scores), key=lambda p: p[1], reverse=True)]

    best = merged[:top_k]
    return (
        [corpus["docs"][index_of[cid]] for cid in best],
        [corpus["metas"][index_of[cid]] for cid in best],
    )


# ---- citations ----
def _pages(meta):
    page, page_end = meta.get("page"), meta.get("page_end")
    if not page:
        return ""
    return str(page) if not page_end or page_end == page else f"{page}-{page_end}"


def page_label(meta):
    pages = _pages(meta)
    if not pages:
        return ""
    return f"pp. {pages}" if "-" in pages else f"p. {pages}"


def chunk_header(meta):
    """Header line placed above each chunk in the LLM prompt."""
    header = f"Source: {meta.get('source', 'Unknown source')}"
    if meta.get("course_code"):
        header += f" | Course Code: {meta['course_code']}"
    pages = _pages(meta)
    if pages:
        header += f" | Page: {pages}"
    return header


def build_citations(metas):
    """De-duplicated [{source, page, page_end, course_code, label}] for the API response."""
    seen, out = set(), []
    for m in metas:
        key = (m.get("source"), m.get("page"), m.get("page_end"), m.get("course_code"))
        if key in seen:
            continue
        seen.add(key)
        label = m.get("source", "Unknown source")
        pl = page_label(m)
        out.append({
            "source": m.get("source", "Unknown source"),
            "page": m.get("page"),
            "page_end": m.get("page_end"),
            "course_code": m.get("course_code"),
            "label": f"{label}, {pl}" if pl else label,
        })
    return out