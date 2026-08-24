# Northwind Traders — Enterprise Knowledge Assistant

A RAG-based knowledge assistant over a small Finance/HR/IT/Legal/Sales document set, built for
the Senior AI Engineer take-home (Azure AI + RAG Architecture, Implementation & Problem Solving).

It ships as **two parallel pipeline profiles over the same code path**:

- **`baseline`** — deliberately naive RAG (fixed-size chunking, vector-only, no reranking, no
  ACL, no ambiguity/insufficient-evidence detection) — reproduces the assignment's six failure
  scenarios on purpose.
- **`improved`** — every fix applied (section-aware chunking, hybrid search + reranking, query
  rewriting, department ACL, version resolution, ambiguity + confidence gating).

Running the evaluation harness against both profiles is what produces the "before vs after"
comparison the assignment asks for — see [`eval/results/comparison.md`](eval/results/comparison.md).

It also ships with **three backends**:

- **`local`** (default) — zero cloud dependencies: a deterministic hashed embedding, BM25 +
  cosine hybrid search fused with Reciprocal Rank Fusion, and an extractive (non-LLM) answer
  mode. This exists so the whole pipeline — ingestion, ACL, versioning, ambiguity detection, the
  eval harness — is runnable and testable with `pip install` and nothing else.
- **`github`** — free, no credit card: real chat + embeddings via [GitHub
  Models](https://github.com/marketplace/models) (OpenAI-API-compatible, authenticated with a
  GitHub personal access token), paired with the same local hybrid search `local` uses (GitHub
  Models doesn't include a managed search service). Useful when Azure access isn't available yet —
  it exercises the real generation/citation/guardrail code path, just without Azure AI Search's
  managed semantic reranker.
- **`azure`** — the real, documented production path: Azure OpenAI (chat + embeddings) + Azure
  AI Search (hybrid + semantic ranker). See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for
  the full production architecture and the written answers to the assignment's Step 2 / Step 5
  questions.

## Quickstart

```bash
python -m venv .venv
.venv/Scripts/activate   # .venv/bin/activate on macOS/Linux
pip install -r requirements.txt

# Build both indexes (local backend, no credentials needed)
python scripts/ingest.py --profile improved --backend local
python scripts/ingest.py --profile baseline --backend local

# Chat
streamlit run app.py          # UI, with a role/backend/profile switcher in the sidebar
python cli.py --role sales    # terminal chat
uvicorn api:app --reload      # POST /chat {"query": "...", "role": "finance"}
```

To use the free GitHub Models backend: copy `.env.example` to `.env`, add a GitHub personal
access token as `GITHUB_MODELS_TOKEN`, then `python scripts/ingest.py --profile improved --backend
github` and run the app with `BACKEND=github`.

To run against real Azure services: same `.env` file, fill in your Azure OpenAI and Azure AI
Search credentials, then re-run ingestion and the app with `BACKEND=azure`.

## Repository layout

```
src/
  config.py               Backend + pipeline-profile configuration (baseline vs improved knobs)
  access_control.py        Role -> department ACL mapping
  rag_pipeline.py           Orchestrator: rewrite -> retrieve -> resolve -> rerank -> gate -> generate
  ingestion/
    parsers.py               PDF/DOCX/XLSX -> heading-tagged text blocks
    chunking.py               Baseline (naive fixed-size) vs improved (section-aware) chunking
    catalog.py                 Loads data/catalog.json (department/version/effective-date metadata)
    pipeline.py                 Ties parsing+chunking+metadata into index-ready records
  embeddings/                Azure OpenAI (production) / local hashed embeddings (offline dev)
  vectorstore/               Azure AI Search (production) / local BM25+cosine hybrid (offline dev)
  retrieval/
    query_rewrite.py           Conversational query condensing (Scenario 6)
    retriever.py                 Hybrid search, multi-query expansion (Scenario 2), version resolution (Scenario 3)
    reranker.py                  Local lexical reranker (Azure path uses the managed semantic ranker)
    ambiguity.py                  Ambiguous-query detection (Scenario 5)
    confidence.py                 Insufficient-evidence gating (Scenario 4)
  generation/
    prompt.py, llm_client.py, generator.py    Grounded generation + citations, extractive fallback
    guardrails.py                              Citation verification / hallucination flag
data/
  raw/                     The 11 source documents (Finance/HR/IT/Legal/Sales)
  catalog.json              Curated metadata: department, doc_family, version, effective dates
eval/
  dataset.json              24 questions / 27 turns across 7 categories (see below)
  evaluate.py, compare.py    Run + diff baseline vs improved
  results/                   Generated: baseline.json, improved.json, comparison.md
docs/
  ARCHITECTURE.md            Production Azure architecture + Step 2 / Step 5 written answers
  architecture-diagram.svg
app.py / api.py / cli.py    Streamlit UI / FastAPI / terminal chat
scripts/ingest.py           Build the retrieval index for a given (backend, profile)
```

## The six failure scenarios, and where each is fixed

| # | Scenario | Root cause (baseline) | Fix (improved) | Code |
|---|---|---|---|---|
| 1 | Correct document, wrong chunk | Fixed 300-char windows with no overlap slice straight through tables — a price ends up in a different chunk than the category name it belongs to | Section-aware chunking (headings stay with their content, tables kept atomic) + hybrid search + reranking | `ingestion/chunking.py`, `retrieval/reranker.py` |
| 2 | Info across multiple sections/docs | Single retrieval pass, one side of a comparison crowds out the other in top-k | Detect comparison phrasing ("compare X and Y", "X vs Y"), retrieve each side separately, merge | `retrieval/retriever.py::_expand_queries` |
| 3 | Conflicting/superseded documents | No version awareness — an older rate card can outrank the current one | `doc_family` + `effective_date` in metadata; superseded docs are dropped from results unless the query explicitly names their year/version | `ingestion/catalog.py::latest_version_by_family`, `retrieval/retriever.py::_resolve_versions` |
| 4 | Hallucination / missing info | Always retrieves *something*, always answers | A confidence gate computed from retrieval signal (embedding similarity + lexical overlap locally; the Azure semantic ranker score in production) blocks generation below threshold | `retrieval/confidence.py` |
| 5 | Ambiguous query ("What is the limit?") | Picks one of several unrelated "limits" and answers confidently | Detects low-content-word queries whose top hits spread across ≥3 unrelated sections and asks a clarifying question instead of guessing | `retrieval/ambiguity.py` |
| 6 | Conversational context | Either ignores history (follow-ups retrieve nothing) or pollutes retrieval with raw chat history | Condenses history + new turn into one standalone query *before* retrieval (LLM rewrite in production, heuristic fallback locally) — never embeds raw history | `retrieval/query_rewrite.py` |

Plus **document-level access control** (Step 5, Q4) as a seventh axis, tested the same way:
department is a filterable field, enforced server-side at the retriever, never as a post-hoc
check — see the `access_control` category in the eval set and `src/access_control.py`.

## Evaluation

`eval/dataset.json` has 24 cases / 27 turns across straightforward, multi-document, conflicting-
version, ambiguous, no-answer, follow-up (multi-turn), and access-control categories — each with
an expected document, expected answer keywords, and/or an expected system behavior
(`expect_ambiguous`, `expect_insufficient_evidence`, `forbidden_departments`).

```bash
python eval/evaluate.py --profile baseline
python eval/evaluate.py --profile improved
python eval/compare.py
```

Measures, per the assignment's list: retrieval hit-rate/precision@k; generation answer-keyword
coverage, groundedness (cited-sentence ratio), citation correctness, hallucination rate,
ambiguous/no-answer detection accuracy; access-control violation rate; system latency and
(when an LLM is configured) token usage and estimated cost.

Latest local-backend run (`eval/results/comparison.md`, generated, not hand-edited):

| Metric | Baseline | Improved |
|---|---|---|
| Retrieval hit rate | 0.84 | **1.00** |
| Retrieval precision@k | 0.47 | **0.63** |
| Answer keyword coverage | 0.49 | **0.79** |
| Citation correctness | 0.84 | **1.00** |
| Ambiguous-query detection | 0.00 | **1.00** |
| ACL violation rate | 0.07 | **0.00** |
| Hallucination rate | 0.22 | **0.19** |

Two metrics look worse for `improved` in the raw numbers (groundedness, latency) — both are
explained as measurement artifacts of the local/no-LLM backend, not real regressions; see the
"Known caveats" section of `eval/results/comparison.md` for the detail, and rerun with
`BACKEND=azure` for a representative (LLM-graded, network-latency-inclusive) comparison.

## Bonus items implemented

Query rewriting · hybrid search · semantic reranking (Azure semantic ranker in production, local
lexical reranker for offline dev) · metadata filtering · confidence scoring · guardrails
(citation verification) · document-level access control · automated evaluation pipeline
(`eval/`). Not implemented: response caching and Application Insights wiring are designed for in
`docs/ARCHITECTURE.md` but not code — see "What I'd change before production" below.

## What I'd change before production

- Wire actual Application Insights custom events per pipeline stage (the timing/token
  instrumentation is already there in `RAGResult`; it just logs to nowhere in local mode).
- Add semantic response caching (embedding-similarity cache keyed on rewritten query + department
  scope) — the biggest cost lever per `docs/ARCHITECTURE.md` §7 that isn't built yet.
- Replace the hand-curated `data/catalog.json` with a real metadata source (SharePoint columns /
  a document management system) before this would work past the assignment's fixed 11 documents.
- Add a cross-encoder or LLM-based reranker for the local backend (currently lexical-overlap only)
  so offline eval numbers are closer to what Azure's semantic ranker would produce.
