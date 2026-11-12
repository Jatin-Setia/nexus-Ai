"""
Retrieval evaluation: vector-only vs BM25-only vs hybrid (vector + BM25 + RRF).

For every question in eval/questions.json the "correct" pages are found automatically:
any page, in any PDF in this folder, whose text matches the question's evidence pattern.
A retrieved chunk counts as a HIT if it comes from one of those PDFs and its page range
(page .. page_end) overlaps one of those pages.

Metrics (top 6 chunks, the same number the app sends to the LLM):
  Hit@1 / Hit@3 / Hit@6  - share of questions with a correct chunk in the top 1 / 3 / 6
  MRR                    - mean of 1/rank of the first correct chunk (0 if none in top 6)

Usage:
  python3 eval_retrieval.py                 # vector, BM25, hybrid (needs chromadb + fastembed, i.e. your normal venv)
  python3 eval_retrieval.py --rerank        # also evaluate hybrid + cross-encoder rerank (needs sentence-transformers)
  python3 eval_retrieval.py --bm25-only     # keyword search only; needs no embedding model
  python3 eval_retrieval.py --check-gold    # just show which pages each question's answer was found on
"""
import argparse
import contextlib
import glob
import io
import json
import os
import re
import sqlite3
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
TOP_K = 6

ap = argparse.ArgumentParser()
ap.add_argument("--bm25-only", action="store_true")
ap.add_argument("--rerank", action="store_true")
ap.add_argument("--check-gold", action="store_true")
ap.add_argument("--questions", default="eval/questions.json")
ap.add_argument("--out", default="eval/results")
args = ap.parse_args()

if not args.rerank:
    os.environ["RERANK"] = "0"  # must be set before importing retrieval
import retrieval  # noqa: E402


# ---------------------------------------------------------------- gold pages
def load_pdf_pages():
    from pypdf import PdfReader
    pages = {}
    for f in sorted(glob.glob("*.pdf")):
        reader = PdfReader(f)
        pages[f] = [(p.extract_text() or "") for p in reader.pages]
    return pages


def find_gold(questions, pdf_pages):
    gold = {}
    for q in questions:
        hits = {}
        for pattern in q["evidence"]:
            rx = re.compile(pattern, re.I)
            for src, texts in pdf_pages.items():
                for i, t in enumerate(texts, 1):
                    if rx.search(t):
                        hits.setdefault(src, set()).add(i)
        gold[q["id"]] = hits
    return gold


# ---------------------------------------------------------------- corpus access
class SqliteCollection:
    """Minimal read-only stand-in for a Chroma collection (enough for BM25)."""
    def __init__(self, path):
        self.db = sqlite3.connect(path)

    def get(self, include=None):
        rows = {}
        for rid, key, sv, iv, fv in self.db.execute(
                "select id,key,string_value,int_value,float_value from embedding_metadata"):
            rows.setdefault(rid, {})[key] = sv if sv is not None else (iv if iv is not None else fv)
        eid = dict(self.db.execute("select id, embedding_id from embeddings"))
        ids = sorted(rows)
        out = {"ids": [eid[i] for i in ids]}
        if include:
            out["documents"] = [rows[i].get("chroma:document", "") for i in ids]
            out["metadatas"] = [{k: v for k, v in rows[i].items() if k != "chroma:document"} for i in ids]
        return out


def get_collection():
    if args.bm25_only:
        return SqliteCollection("chroma_db/chroma.sqlite3"), None
    import chromadb
    from embedder import Embedder
    client = chromadb.PersistentClient(path="./chroma_db")
    return client.get_or_create_collection(name="campus_docs"), Embedder()


# ---------------------------------------------------------------- the three searchers
def search_bm25(collection, question, _emb):
    corpus = retrieval._load_corpus(collection)
    idx = corpus["bm25"].top(retrieval._query_tokens(question), TOP_K)
    return [corpus["metas"][i] for i in idx]


def search_vector(collection, question, emb):
    res = collection.query(query_embeddings=[emb], n_results=TOP_K)
    return res["metadatas"][0]


def search_hybrid(collection, question, emb):
    with contextlib.redirect_stdout(io.StringIO()):  # retrieval prints progress lines
        _, metas = retrieval.hybrid_search(collection, question, emb, top_k=TOP_K)
    return metas


# ---------------------------------------------------------------- scoring
def is_hit(meta, gold_for_q):
    pages = gold_for_q.get(meta.get("source"))
    if not pages:
        return False
    lo = int(meta.get("page") or 0)
    hi = int(meta.get("page_end") or lo)
    return any(lo <= p <= hi for p in pages)


def first_hit_rank(metas, gold_for_q):
    for r, m in enumerate(metas, 1):
        if is_hit(m, gold_for_q):
            return r
    return None


def summarize(ranks):
    n = len(ranks)
    pct = lambda k: 100.0 * sum(1 for r in ranks if r is not None and r <= k) / n
    mrr = sum((1.0 / r) if r else 0.0 for r in ranks) / n
    return {"n": n, "hit@1": pct(1), "hit@3": pct(3), "hit@6": pct(6), "mrr": mrr}


def main():
    qpath = next((p for p in (args.questions, "questions.json", "eval/questions.json")
                  if os.path.exists(p)), None)
    if not qpath:
        sys.exit("Could not find questions.json. Put it in an 'eval' folder next to this script "
                 "(eval/questions.json) or pass --questions PATH.")
    questions = json.load(open(qpath, encoding="utf-8"))
    pdf_pages = load_pdf_pages()
    gold = find_gold(questions, pdf_pages)

    missing = [q["id"] for q in questions if not gold[q["id"]]]
    if args.check_gold or missing:
        for q in questions:
            where = "; ".join(f"{s} p.{sorted(p)}" for s, p in gold[q["id"]].items()) or "NOT FOUND"
            print(f"{q['id']}  {where}")
        if missing:
            print("\nQuestions whose evidence was not found in any PDF:", missing)
            if not args.check_gold:
                sys.exit(1)
        if args.check_gold:
            return

    collection, emb_model = get_collection()
    methods = {"BM25 only (keyword)": search_bm25}
    if not args.bm25_only:
        methods = {"Vector only": search_vector, **methods, "Hybrid (vector + BM25 + RRF)": search_hybrid}
        if args.rerank:
            methods["Hybrid + rerank"] = search_hybrid  # RERANK is on, so hybrid_search reranks

    results, per_q = {}, {}
    for name, fn in methods.items():
        ranks = []
        for q in questions:
            emb = emb_model.encode(q["q"]).tolist() if emb_model else None
            metas = fn(collection, q["q"], emb)
            ranks.append(first_hit_rank(metas, gold[q["id"]]))
        results[name] = {"all": summarize(ranks)}
        for t in ("keyword", "paraphrase"):
            sub = [r for r, q in zip(ranks, questions) if q["type"] == t]
            results[name][t] = summarize(sub)
        per_q[name] = {q["id"]: r for q, r in zip(questions, ranks)}

    lines = [f"Retrieval evaluation - {len(questions)} questions, top {TOP_K} chunks, "
             f"{len(pdf_pages)} PDFs\n",
             "| Method | Hit@1 | Hit@3 | Hit@6 | MRR |", "|---|---|---|---|---|"]
    for name, r in results.items():
        a = r["all"]
        lines.append(f"| {name} | {a['hit@1']:.1f}% | {a['hit@3']:.1f}% | {a['hit@6']:.1f}% | {a['mrr']:.3f} |")
    lines += ["", "By question type (Hit@6):", "| Method | Keyword-style | Paraphrased |", "|---|---|---|"]
    for name, r in results.items():
        lines.append(f"| {name} | {r['keyword']['hit@6']:.1f}% (n={r['keyword']['n']}) | "
                     f"{r['paraphrase']['hit@6']:.1f}% (n={r['paraphrase']['n']}) |")
    lines += ["", "Questions with no correct chunk in the top 6:"]
    for name, pq in per_q.items():
        failed = [qid for qid, r in pq.items() if r is None]
        lines.append(f"- {name}: {', '.join(failed) if failed else 'none'}")
    report = "\n".join(lines)
    print(report)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    suffix = "_bm25" if args.bm25_only else ""
    open(args.out + suffix + ".md", "w", encoding="utf-8").write(report + "\n")
    json.dump({"summary": results, "first_correct_rank": per_q},
              open(args.out + suffix + ".json", "w", encoding="utf-8"), indent=2)
    print(f"\nSaved {args.out}{suffix}.md / .json")


if __name__ == "__main__":
    main()