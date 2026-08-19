# Reflection & Action Plan — ChuNguyenTuanAnh

## Lecture → code mapping
| Concept | Implementation | Observation |
|---|---|---|
| Conservative coreference | `resolve_coreferences()` | LLM calls only for chunks with pronoun/generic-reference triggers; ambiguity remains auditable. |
| Strict schema allowlist | `ALLOWED_NODE_TYPES`, `ALLOWED_RELATIONS` | Unsupported relation labels are rejected before ingestion. |
| Entity resolution | `build_resolution_map()` | FAISS ANN is only candidate generation; lexical/type guard decides merge safety. |
| Bulk ingestion | `Neo4jStore.bulk_insert_*()` | Uses `UNWIND` batches; no row-by-row network writes. |
| Super-node mitigation | `GraphRetriever.retrieve()` | degree >100 is capped to at most 50 recent edges plus a global edge cap. |
| Evaluation | `evaluate()` + `Judge` | Same generator family, golden reference and 3 judge dimensions expose quality/cost trade-offs. |

## Debugging lesson
A deterministic near-dedup test exposed that hashing `title + body` missed syndicated copies whose title changed while body stayed the same. The fix fingerprints article body and keeps an audit table. CI also exposed duplicate GitHub Action triggers, which were separated into lightweight push CI versus sentinel-only integration runs.

## Action plan
For a production knowledge assistant, keep provenance on every relation, namespace graph ingestion by dataset/version, gate entity merges with both semantic and lexical evidence, and add a self-correcting retrieval route before expanding graph radius. Community reports should serve global questions while local BFS handles entity-centric questions.

## Bonus evidence
- Near-duplicates removed: 5
- Community reports built: 163
