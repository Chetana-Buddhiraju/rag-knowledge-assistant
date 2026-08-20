# Baseline vs Improved — Evaluation Comparison

Backend: `local` — 24 cases / 27 turns

| Metric | Baseline | Improved | Change |
|---|---|---|---|
| Retrieval hit rate (higher better) | 0.8421 | 1.0000 | better (+0.1579) |
| Retrieval precision@k (higher better) | 0.4737 | 0.6316 | better (+0.1579) |
| Answer keyword coverage (higher better) | 0.4921 | 0.7851 | better (+0.2930) |
| Groundedness — cited sentence ratio (higher better) | 0.4251 | 0.3112 | worse (-0.1139) |
| Citation correctness (higher better) | 0.8421 | 1.0000 | better (+0.1579) |
| Hallucination rate (LOWER better) | 0.2222 | 0.1852 | better (-0.0370) |
| Ambiguous-query detection accuracy (higher better) | 0.0000 | 1.0000 | better (+1.0000) |
| "No answer" detection accuracy (higher better) | 0.7778 | 0.8000 | better (+0.0222) |
| ACL violation rate (LOWER better, should be 0) | 0.0741 | 0.0000 | better (-0.0741) |
| Avg latency, ms (LOWER better) | 0.3582 | 0.7284 | worse (+0.3702) |
| P95 latency, ms (LOWER better) | 0.5500 | 1.2200 | worse (+0.6700) |
| Total tokens used across eval set | 0 | 0 | flat (+0.0000) |
| Estimated LLM cost, USD | n/a | n/a | n/a |

## What changed which metric

- **Chunking (section-aware, tables kept atomic) + hybrid search + reranking** -> retrieval hit rate / precision@k (Scenario 1).
- **Multi-query expansion on comparison phrasing** -> retrieval recall on `multi_document` category questions (Scenario 2).
- **Version resolution (doc_family + effective_date)** -> citation correctness on `conflicting_versions` questions stops citing the superseded rate card (Scenario 3).
- **Confidence gate before generation** -> hallucination rate on `no_answer` questions, at the cost of a few more "insufficient evidence" responses on genuinely hard questions (Scenario 4).
- **Ambiguity detection** -> ambiguous-query detection accuracy; baseline always guesses one answer (Scenario 5).
- **Department ACL pre-filter** -> ACL violation rate drops to 0; baseline has no filter at all and leaks cross-department chunks (Step 5, Q4).
- **Query rewriting** -> keyword coverage / hit rate on `follow_up` conversation turns, where the raw follow-up text alone retrieves poorly (Scenario 6).

## Known caveats in this comparison

- **Groundedness went down, not up.** This is a metric artifact, not a regression: in local/no-LLM (extractive) mode, one retrieved chunk = one citation, and the guardrail's groundedness check is *per sentence*. Improved's larger, section-aware chunks pack more sentences behind a single citation marker than baseline's small 300-char chunks do, so naive per-sentence counting scores it lower even though every sentence is still fully attributed to its (correct) source chunk. With a real LLM (BACKEND=azure), the model is instructed to cite per claim regardless of chunk size, so this artifact goes away — rerun the eval with Azure OpenAI configured to get a real groundedness signal.
- **Latency looks worse for improved, in absolute local numbers.** Both profiles run in well under 2ms locally with no network calls — the delta is entirely the extra CPU work of multi-query expansion and reranking, not a meaningful production signal. In production (BACKEND=azure), latency is dominated by the Azure AI Search and Azure OpenAI network round-trips (tens to hundreds of ms each), which swamp this difference; rerun with BACKEND=azure for a representative latency comparison.

Full per-question detail: `eval/results/baseline.json`, `eval/results/improved.json`.