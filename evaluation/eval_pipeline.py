"""
evaluation/eval_pipeline.py
Lightweight RAG evaluation using a small set of ground-truth QA pairs.

Metrics computed
----------------
- Faithfulness  : answer tokens that appear in the retrieved context (recall proxy)
- Answer length : word count of the generated answer
- Source count  : number of source chunks returned
- Latency       : time taken per query (seconds)

Usage
-----
    python -m evaluation.eval_pipeline

You can extend EVAL_DATASET with your own question/expected_answer pairs.
"""

import sys
import time
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.rag_pipeline import get_pipeline

# ── Ground-truth dataset ─────────────────────────────────────────────────────
# Add your own Q&A pairs here. `expected_keywords` are terms you expect to
# appear in a correct answer — used as a simple faithfulness check.
EVAL_DATASET = [
    {
        "question": "What are the termination conditions mentioned in the agreement?",
        "expected_keywords": ["termination", "notice", "breach", "clause"],
    },
    {
        "question": "What are the liability limitations in the contract?",
        "expected_keywords": ["liability", "damages", "limit", "indemnif"],
    },
    {
        "question": "Who are the parties to the agreement?",
        "expected_keywords": ["party", "parties", "between", "agreement"],
    },
    {
        "question": "What is the governing law for disputes?",
        "expected_keywords": ["govern", "law", "jurisdiction", "dispute"],
    },
    {
        "question": "What are the payment terms specified?",
        "expected_keywords": ["payment", "invoice", "due", "days"],
    },
]


# ── Metrics ───────────────────────────────────────────────────────────────────

def faithfulness_score(answer: str, context_docs: list) -> float:
    """
    Fraction of answer words that appear in the retrieved context.
    This is a crude but dependency-free proxy for faithfulness.
    """
    context_text = " ".join(d.page_content.lower() for d in context_docs)
    answer_words = answer.lower().split()
    if not answer_words:
        return 0.0
    overlap = sum(1 for w in answer_words if w in context_text)
    return round(overlap / len(answer_words), 4)


def keyword_hit_rate(answer: str, keywords: list[str]) -> float:
    """Fraction of expected keywords found in the answer (case-insensitive)."""
    answer_lower = answer.lower()
    hits = sum(1 for kw in keywords if kw.lower() in answer_lower)
    return round(hits / len(keywords), 4) if keywords else 0.0


# ── Runner ────────────────────────────────────────────────────────────────────

def run_evaluation(dataset: list[dict] | None = None) -> list[dict]:
    dataset = dataset or EVAL_DATASET
    pipeline = get_pipeline()

    results = []
    print("\n" + "=" * 70)
    print("LegalDocAssistant — Evaluation Report")
    print("=" * 70)

    for i, item in enumerate(dataset, start=1):
        question = item["question"]
        keywords = item.get("expected_keywords", [])

        print(f"\n[{i}/{len(dataset)}] Q: {question}")

        t0     = time.perf_counter()
        output = pipeline.query(question, retrieve_k=10, rerank_n=3)
        latency = round(time.perf_counter() - t0, 2)

        answer  = output["answer"]
        sources = output["sources"]

        faith   = faithfulness_score(answer, sources)
        hit     = keyword_hit_rate(answer, keywords)
        n_words = len(answer.split())
        n_srcs  = len(sources)

        print(f"   Answer preview : {answer[:120]}…")
        print(f"   Faithfulness   : {faith:.2%}")
        print(f"   Keyword hits   : {hit:.2%}  {keywords}")
        print(f"   Answer length  : {n_words} words")
        print(f"   Sources used   : {n_srcs}")
        print(f"   Latency        : {latency}s")

        results.append({
            "question":       question,
            "answer":         answer,
            "faithfulness":   faith,
            "keyword_hit":    hit,
            "answer_words":   n_words,
            "sources_count":  n_srcs,
            "latency_sec":    latency,
            "sources": [
                {
                    "source": d.metadata.get("source"),
                    "page":   d.metadata.get("page"),
                }
                for d in sources
            ],
        })

    # ── Summary ──────────────────────────────────────────────────────────────
    avg = lambda key: round(sum(r[key] for r in results) / len(results), 4)  # noqa: E731

    print("\n" + "=" * 70)
    print("  Summary")
    print("=" * 70)
    print(f"  Questions evaluated : {len(results)}")
    print(f"  Avg faithfulness    : {avg('faithfulness'):.2%}")
    print(f"  Avg keyword hit     : {avg('keyword_hit'):.2%}")
    print(f"  Avg answer length   : {avg('answer_words')} words")
    print(f"  Avg latency         : {avg('latency_sec')}s")
    print("=" * 70 + "\n")

    # Save JSON report
    report_path = ROOT / "evaluation" / "eval_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  📄 Full report saved → {report_path}\n")

    return results


if __name__ == "__main__":
    run_evaluation()
