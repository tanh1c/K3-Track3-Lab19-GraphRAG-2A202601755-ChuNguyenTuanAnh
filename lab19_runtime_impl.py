"""Production-oriented runtime for Lab 19 GraphRAG vs Flat RAG.

The runtime intentionally keeps the instructor schema/allowlists, locks the
source corpus to the first 5,000 HackerNoon rows, rate-limits Groq calls, keeps
provenance on every graph edge, and exports rubric-facing diagnostics.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import re
import time
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from lab19_core import (
    merge_guard,
    near_dedup_dataframe,
    normalize_entity,
    normalize_text,
    supernode_edge_limit,
)
from lab19_pipeline import (
    ALLOWED_NODE_TYPES,
    ALLOWED_RELATIONS,
    build_chunks,
    needs_coreference,
    select_extraction_source,
    standardize_news,
    validate_extracted_triples,
)
from lab19_utils import JsonlCheckpoint, RateLimitPolicy, parse_retry_after_seconds

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DATASET_NAME = "HackerNoon/tech-company-news-data-dump"
DATASET_SCOPE = "first5000"
GOLDEN_PATH = Path("data/graphrag_golden_50_first5000.csv")
GOLDEN_DETAILED_PATH = Path("data/graphrag_golden_50_first5000_detailed.csv")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GLOBAL_EDGE_CAP = 250
MAX_GRAPH_CONTEXT_CHARS = 14_000
SUPER_NODE_DEGREE = 100
SUPER_NODE_EDGE_CAP = 50

MANUAL_ALIASES = {
    "msft": "Microsoft",
    "microsoft corp": "Microsoft",
    "microsoft corporation": "Microsoft",
    "goog": "Google",
    "googl": "Google",
    "google llc": "Google",
    "meta platforms": "Meta",
    "meta platforms inc": "Meta",
    "aapl": "Apple",
    "apple inc": "Apple",
    "amzn": "Amazon",
    "amazon com": "Amazon",
}


@dataclass(frozen=True)
class RunConfig:
    mode: str
    extraction_max_chunks: int
    golden_limit: int
    coref_batch_size: int = 4
    extraction_batch_size: int = 4
    groq_timeout_s: float = 45.0
    groq_min_interval_s: float = 12.0
    groq_max_retries: int = 7
    vector_top_k_flat: int = 6
    vector_top_k_graph: int = 4

    @classmethod
    def for_mode(cls, mode: str) -> "RunConfig":
        mode = str(mode).strip().lower()
        if mode == "smoke":
            return cls(mode="smoke", extraction_max_chunks=24, golden_limit=3)
        if mode == "full":
            return cls(mode="full", extraction_max_chunks=400, golden_limit=50)
        raise ValueError(f"Unsupported run mode: {mode}")


class GroqRuntime:
    def __init__(self, config: RunConfig):
        from groq import Groq

        api_key = os.getenv("GROQ_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("Missing GROQ_API_KEY")
        self.fast_model = os.getenv("GROQ_FAST_MODEL", "llama-3.1-8b-instant").strip()
        self.generation_model = (
            os.getenv("GROQ_GENERATION_MODEL", "").strip()
            or self.fast_model
        )
        self.client = Groq(
            api_key=api_key,
            timeout=config.groq_timeout_s,
            max_retries=0,
        )
        self.config = config
        self.policy = RateLimitPolicy(base_delay_s=2.0, max_delay_s=120.0, jitter_s=1.0)
        self._last_request_started = 0.0

    def _pace(self) -> None:
        elapsed = time.monotonic() - self._last_request_started
        wait = self.config.groq_min_interval_s - elapsed
        if wait > 0:
            time.sleep(wait)
        self._last_request_started = time.monotonic()

    @staticmethod
    def _headers_from_exception(exc: Exception) -> Mapping[str, Any] | None:
        response = getattr(exc, "response", None)
        return getattr(response, "headers", None) if response is not None else None

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        name = type(exc).__name__
        if name in {
            "RateLimitError",
            "APIConnectionError",
            "APITimeoutError",
            "InternalServerError",
        }:
            return True
        status = getattr(exc, "status_code", None)
        return isinstance(status, int) and (status == 429 or status >= 500)

    def chat(
        self,
        *,
        system: str,
        user: str,
        model: str | None = None,
        max_tokens: int = 1200,
        json_mode: bool = False,
    ) -> tuple[str, dict[str, Any]]:
        last: Exception | None = None
        for attempt in range(self.config.groq_max_retries):
            self._pace()
            try:
                kwargs: dict[str, Any] = {
                    "model": model or self.fast_model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "temperature": 0.0,
                    "max_tokens": max_tokens,
                }
                if json_mode:
                    kwargs["response_format"] = {"type": "json_object"}
                response = self.client.chat.completions.create(**kwargs)
                usage_obj = getattr(response, "usage", None)
                usage = {
                    "prompt_tokens": getattr(usage_obj, "prompt_tokens", None),
                    "completion_tokens": getattr(usage_obj, "completion_tokens", None),
                    "total_tokens": getattr(usage_obj, "total_tokens", None),
                }
                return (response.choices[0].message.content or "").strip(), usage
            except Exception as exc:  # provider SDK exception hierarchy is optional at import time
                last = exc
                if not self._is_retryable(exc) or attempt == self.config.groq_max_retries - 1:
                    raise
                retry_after = parse_retry_after_seconds(self._headers_from_exception(exc))
                delay = self.policy.delay_for(attempt, retry_after_s=retry_after)
                print(
                    f"[groq] retryable {type(exc).__name__}; "
                    f"attempt={attempt + 1}/{self.config.groq_max_retries}; sleep={delay:.1f}s"
                )
                time.sleep(delay)
        raise RuntimeError(last or "Groq retry loop exhausted")

    @staticmethod
    def parse_json(text: str) -> dict[str, Any]:
        clean = str(text).strip()
        clean = re.sub(r"^```(?:json)?\s*", "", clean, flags=re.I)
        clean = re.sub(r"\s*```$", "", clean)
        left, right = clean.find("{"), clean.rfind("}")
        if left < 0 or right <= left:
            raise ValueError("No JSON object found in LLM response")
        return json.loads(clean[left : right + 1])

    def chat_json(self, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        text, usage = self.chat(json_mode=True, **kwargs)
        return self.parse_json(text), usage


class Embedder:
    def __init__(self):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(EMBEDDING_MODEL)

    def encode(self, texts: list[str], batch_size: int = 128) -> np.ndarray:
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 256,
            normalize_embeddings=True,
        ).astype("float32")


class FlatIndex:
    def __init__(self, chunks_df: pd.DataFrame, embedder: Embedder):
        import faiss

        self.store = chunks_df.reset_index(drop=True).copy()
        vectors = embedder.encode(self.store["text"].fillna("").tolist())
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        self.embedder = embedder

    def retrieve(self, query: str, k: int) -> tuple[str, pd.DataFrame]:
        qv = self.embedder.encode([query], batch_size=1)
        scores, ids = self.index.search(qv, min(k, self.index.ntotal))
        rows: list[dict[str, Any]] = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:
                continue
            row = self.store.iloc[int(idx)]
            rows.append(
                {
                    "score": float(score),
                    "chunk_id": row.chunk_id,
                    "source_row_id": int(row.source_row_id),
                    "published_date": row.published_date,
                    "text": row.text,
                }
            )
        df = pd.DataFrame(rows)
        context = "\n\n".join(
            f"[chunk_id={r.chunk_id} | source_row={r.source_row_id} | "
            f"date={r.published_date} | score={r.score:.3f}]\n{r.text}"
            for r in df.itertuples(index=False)
        )
        return context, df


class DSU:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def download_first_5000(output_path: Path) -> pd.DataFrame:
    from datasets import load_dataset

    hf_token = os.getenv("HF_TOKEN", "").strip()
    if not hf_token:
        raise RuntimeError("Missing HF_TOKEN")
    dataset = load_dataset(
        DATASET_NAME,
        split="train",
        streaming=True,
        token=hf_token,
    )
    rows: list[dict[str, Any]] = []
    for source_row_id, item in enumerate(dataset):
        if source_row_id >= 5000:
            break
        row = dict(item)
        row["source_row_id_download"] = source_row_id
        rows.append(row)
    if len(rows) != 5000:
        raise RuntimeError(f"Expected exactly 5000 streamed rows, got {len(rows)}")
    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print("[data] Source policy FIRST_5000_ROWS_ONLY; rows=5000; random_sampling=DISABLED")
    return df


def prepare_data(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_path = output_dir / "source_first5000.csv"
    raw = download_first_5000(raw_path)
    news = standardize_news(raw)
    before_near = len(news)
    news, near_audit = near_dedup_dataframe(news, max_hamming=3)
    chunks = build_chunks(news)
    news.to_csv(output_dir / "news_first5000_clean.csv", index=False)
    chunks.to_csv(output_dir / "chunks_first5000.csv", index=False)
    near_audit.to_csv(output_dir / "near_dedup_audit.csv", index=False)
    print(
        f"[data] standardized/exact-dedup={before_near:,}; "
        f"after-near-dedup={len(news):,}; chunks={len(chunks):,}"
    )
    return news, chunks, near_audit


def resolve_coreferences(
    source_df: pd.DataFrame,
    llm: GroqRuntime,
    config: RunConfig,
    output_dir: Path,
) -> pd.DataFrame:
    checkpoint = JsonlCheckpoint(output_dir / "checkpoints" / "coref.jsonl", "chunk_id")
    completed = checkpoint.completed_keys()
    rows: dict[str, dict[str, Any]] = {r["chunk_id"]: r for r in checkpoint.rows()}

    # Chunks without pronoun/generic references do not need an LLM call.
    pending_llm: list[dict[str, Any]] = []
    for row in source_df.itertuples(index=False):
        if row.chunk_id in completed:
            continue
        if not needs_coreference(row.text):
            item = {
                "chunk_id": row.chunk_id,
                "resolved_text": row.text,
                "unresolved_mentions": [],
                "coref_route": "NO_TRIGGER",
            }
            checkpoint.upsert(item)
            rows[row.chunk_id] = item
        else:
            pending_llm.append({"chunk_id": row.chunk_id, "text": row.text})

    system = (
        "You are a conservative coreference-resolution component for a knowledge-graph pipeline. "
        "Resolve pronouns/generic references only when the antecedent is explicit in the SAME chunk. "
        "Never invent facts. Preserve dates, numbers, tickers and product names. Return strict JSON."
    )
    for start in range(0, len(pending_llm), config.coref_batch_size):
        batch = pending_llm[start : start + config.coref_batch_size]
        prompt = (
            "Return {\"items\":[{\"chunk_id\":\"...\",\"resolved_text\":\"...\","
            "\"unresolved_mentions\":[\"...\"]}]}. Ambiguous mentions must stay unchanged. INPUT:\n"
            + json.dumps(batch, ensure_ascii=False)
        )
        try:
            obj, _ = llm.chat_json(
                system=system,
                user=prompt,
                model=llm.fast_model,
                max_tokens=1400,
            )
            by_id = {str(x.get("chunk_id")): x for x in obj.get("items", [])}
        except Exception as exc:
            print(f"[coref] batch failed after retries: {type(exc).__name__}: {exc}")
            by_id = {}
        for raw in batch:
            parsed = by_id.get(raw["chunk_id"], {})
            item = {
                "chunk_id": raw["chunk_id"],
                "resolved_text": normalize_text(parsed.get("resolved_text") or raw["text"]),
                "unresolved_mentions": parsed.get("unresolved_mentions", ["COREF_BATCH_FAILED"] if not parsed else []),
                "coref_route": "LLM" if parsed else "FALLBACK_ORIGINAL",
            }
            checkpoint.upsert(item)
            rows[item["chunk_id"]] = item

    coref_df = pd.DataFrame([rows[cid] for cid in source_df["chunk_id"] if cid in rows])
    merged = source_df.merge(coref_df, on="chunk_id", how="left")
    merged["resolved_text"] = merged["resolved_text"].fillna(merged["text"])
    return merged


def extract_triples(
    source_df: pd.DataFrame,
    llm: GroqRuntime,
    config: RunConfig,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    checkpoint = JsonlCheckpoint(output_dir / "checkpoints" / "extraction.jsonl", "chunk_id")
    completed = checkpoint.completed_keys()
    errors: list[dict[str, Any]] = []

    system = (
        "Extract a HIGH-PRECISION knowledge graph from technology-news text. "
        f"Allowed node types: {sorted(ALLOWED_NODE_TYPES)}. "
        f"Allowed relations: {sorted(ALLOWED_RELATIONS)}. "
        "Use only explicitly supported facts. Prefer precision over recall. "
        "Every relation MUST include a short verbatim evidence span and confidence 0..1. Return strict JSON."
    )
    pending = source_df[~source_df["chunk_id"].isin(completed)].copy()
    for start in range(0, len(pending), config.extraction_batch_size):
        batch = pending.iloc[start : start + config.extraction_batch_size]
        payload = [
            {
                "chunk_id": r.chunk_id,
                "published_date": r.published_date,
                "text": r.resolved_text,
            }
            for r in batch.itertuples(index=False)
        ]
        prompt = (
            "Return {\"items\":[{\"chunk_id\":\"...\",\"relations\":[{"
            "\"source\":\"...\",\"source_type\":\"Company|Person|Technology\","
            "\"relation\":\"ALLOWED_RELATION\",\"target\":\"...\","
            "\"target_type\":\"Company|Person|Technology\",\"evidence\":\"...\","
            "\"confidence\":0.0}]}]}. INPUT:\n"
            + json.dumps(payload, ensure_ascii=False)
        )
        try:
            obj, _ = llm.chat_json(
                system=system,
                user=prompt,
                model=llm.fast_model,
                max_tokens=1800,
            )
            by_id = {str(x.get("chunk_id")): x for x in obj.get("items", [])}
        except Exception as exc:
            print(f"[extract] batch failed after retries: {type(exc).__name__}: {exc}")
            by_id = {}
            errors.append({"start": int(start), "error": f"{type(exc).__name__}: {exc}"})

        for r in batch.itertuples(index=False):
            item = by_id.get(r.chunk_id, {"chunk_id": r.chunk_id, "relations": []})
            checkpoint.upsert(item)

    meta = source_df.set_index("chunk_id").to_dict("index")
    triples: list[dict[str, Any]] = []
    for item in checkpoint.rows():
        cid = str(item.get("chunk_id", ""))
        if cid not in meta:
            continue
        info = meta[cid]
        published_date = normalize_text(info.get("published_date"))
        if not published_date:
            continue  # edge-provenance integrity is stricter than recall
        for relation in item.get("relations", []) or []:
            source = normalize_text(relation.get("source"))
            target = normalize_text(relation.get("target"))
            st = relation.get("source_type")
            tt = relation.get("target_type")
            rel = relation.get("relation")
            evidence = normalize_text(relation.get("evidence"))
            try:
                confidence = float(relation.get("confidence", 0.0))
            except (TypeError, ValueError):
                confidence = -1.0
            if not source or not target or not evidence:
                continue
            if st not in ALLOWED_NODE_TYPES or tt not in ALLOWED_NODE_TYPES:
                continue
            if rel not in ALLOWED_RELATIONS or not (0 <= confidence <= 1):
                continue
            triples.append(
                {
                    "source_raw": source,
                    "source_type": st,
                    "relation": rel,
                    "target_raw": target,
                    "target_type": tt,
                    "source_chunk_id": cid,
                    "source_row_id": int(info["source_row_id"]),
                    "published_date": published_date,
                    "evidence": evidence,
                    "confidence": confidence,
                }
            )

    columns = [
        "source_raw", "source_type", "relation", "target_raw", "target_type",
        "source_chunk_id", "source_row_id", "published_date", "evidence", "confidence",
    ]
    triples_df = pd.DataFrame(triples, columns=columns)
    triples_df = validate_extracted_triples(triples_df)
    error_df = pd.DataFrame(errors)
    triples_df.to_csv(output_dir / "raw_triples.csv", index=False)
    error_df.to_csv(output_dir / "extraction_errors.csv", index=False)
    print(f"[extract] valid triples={len(triples_df):,}; errors={len(error_df):,}")
    return triples_df, error_df


def build_resolution_map(
    raw_triples_df: pd.DataFrame,
    embedder: Embedder,
    threshold: float = 0.90,
    top_k: int = 5,
) -> tuple[dict[tuple[str, str], str], pd.DataFrame]:
    import faiss

    mentions: list[tuple[str, str]] = []
    for r in raw_triples_df.itertuples(index=False):
        mentions.extend([(r.source_type, r.source_raw), (r.target_type, r.target_raw)])
    counts = Counter((typ, normalize_entity(name)) for typ, name in mentions)
    display: dict[tuple[str, str], str] = {}
    for typ, name in mentions:
        display.setdefault((typ, normalize_entity(name)), name)

    mapping: dict[tuple[str, str], str] = {}
    audit: list[dict[str, Any]] = []
    for key in counts:
        typ, norm = key
        if typ == "Company" and norm in MANUAL_ALIASES:
            mapping[key] = MANUAL_ALIASES[norm]
            audit.append(
                {
                    "type": typ,
                    "left": display[key],
                    "right": MANUAL_ALIASES[norm],
                    "similarity": 1.0,
                    "decision": "MERGE_MANUAL",
                }
            )

    for typ in sorted(ALLOWED_NODE_TYPES):
        keys = [key for key in counts if key[0] == typ and key not in mapping]
        if not keys:
            continue
        names = [display[key] for key in keys]
        vectors = embedder.encode(names)
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        sims, nbrs = index.search(vectors, min(top_k, len(names)))
        dsu = DSU(len(names))
        for i in range(len(names)):
            for score, j in zip(sims[i], nbrs[i]):
                j = int(j)
                if j < 0 or i >= j or float(score) < threshold:
                    continue
                allowed = merge_guard(names[i], names[j], typ)
                audit.append(
                    {
                        "type": typ,
                        "left": names[i],
                        "right": names[j],
                        "similarity": float(score),
                        "decision": "MERGE_VECTOR" if allowed else "REJECT_GUARD",
                    }
                )
                if allowed:
                    dsu.union(i, j)
        groups: dict[int, list[int]] = defaultdict(list)
        for i in range(len(names)):
            groups[dsu.find(i)].append(i)
        for indices in groups.values():
            best = sorted(
                indices,
                key=lambda i: (-counts[keys[i]], len(names[i]), names[i].lower()),
            )[0]
            canonical = names[best]
            for i in indices:
                mapping[keys[i]] = canonical

    for key in counts:
        mapping.setdefault(key, display[key])
    return mapping, pd.DataFrame(audit)


def canonicalize_triples(
    raw_df: pd.DataFrame,
    mapping: dict[tuple[str, str], str],
) -> pd.DataFrame:
    df = raw_df.copy()

    def canonical(name: str, typ: str) -> str:
        norm = normalize_entity(name)
        return mapping.get((typ, norm), MANUAL_ALIASES.get(norm, name))

    df["source_name"] = [canonical(n, t) for n, t in zip(df.source_raw, df.source_type)]
    df["target_name"] = [canonical(n, t) for n, t in zip(df.target_raw, df.target_type)]
    df["source_name_norm"] = df.source_name.map(normalize_entity)
    df["target_name_norm"] = df.target_name.map(normalize_entity)
    df["source_id"] = [
        "lab19_" + hashlib.sha1(f"{t}:{n}".encode()).hexdigest()[:24]
        for t, n in zip(df.source_type, df.source_name_norm)
    ]
    df["target_id"] = [
        "lab19_" + hashlib.sha1(f"{t}:{n}".encode()).hexdigest()[:24]
        for t, n in zip(df.target_type, df.target_name_norm)
    ]
    return df[df.source_id != df.target_id].reset_index(drop=True)


def build_nodes(triples_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for r in triples_df.itertuples(index=False):
        rows.extend(
            [
                {
                    "id": r.source_id,
                    "name": r.source_name,
                    "name_norm": r.source_name_norm,
                    "type": r.source_type,
                    "alias": r.source_raw,
                },
                {
                    "id": r.target_id,
                    "name": r.target_name,
                    "name_norm": r.target_name_norm,
                    "type": r.target_type,
                    "alias": r.target_raw,
                },
            ]
        )
    if not rows:
        return pd.DataFrame(columns=["id", "name", "name_norm", "type", "aliases", "aliases_norm"])
    tmp = pd.DataFrame(rows)
    out: list[dict[str, Any]] = []
    for (node_id, name, name_norm, typ), group in tmp.groupby(
        ["id", "name", "name_norm", "type"], sort=False
    ):
        aliases = sorted(set(group["alias"].map(normalize_text)))
        out.append(
            {
                "id": node_id,
                "name": name,
                "name_norm": name_norm,
                "type": typ,
                "aliases": aliases,
                "aliases_norm": sorted(set(normalize_entity(x) for x in aliases)),
            }
        )
    return pd.DataFrame(out)


def _batches(records: list[dict[str, Any]], size: int = 1000) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(records), size):
        yield records[start : start + size]


class Neo4jStore:
    def __init__(self):
        from neo4j import GraphDatabase

        uri = os.getenv("NEO4J_URI", "").strip()
        user = os.getenv("NEO4J_USER", "neo4j").strip() or "neo4j"
        password = os.getenv("NEO4J_PASSWORD", "").strip()
        self.database = os.getenv("NEO4J_DATABASE", "neo4j").strip() or "neo4j"
        if not uri or not password:
            raise RuntimeError("Missing Neo4j configuration")
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.driver.verify_connectivity()

    def close(self) -> None:
        self.driver.close()

    def run(self, query: str, **params: Any) -> list[dict[str, Any]]:
        with self.driver.session(database=self.database) as session:
            result = session.run(query, **params)
            rows = [record.data() for record in result]
            result.consume()
        return rows

    def setup_schema(self) -> None:
        statements = [
            "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE",
            "CREATE INDEX entity_name_norm IF NOT EXISTS FOR (n:Entity) ON (n.name_norm)",
            "CREATE INDEX company_name_norm IF NOT EXISTS FOR (n:Company) ON (n.name_norm)",
            "CREATE INDEX person_name_norm IF NOT EXISTS FOR (n:Person) ON (n.name_norm)",
            "CREATE INDEX technology_name_norm IF NOT EXISTS FOR (n:Technology) ON (n.name_norm)",
        ]
        for statement in statements:
            self.run(statement)

    def reset_scope(self) -> None:
        self.run("MATCH (n:Entity {dataset_scope:$scope}) DETACH DELETE n", scope=DATASET_SCOPE)

    def bulk_insert_nodes(self, nodes_df: pd.DataFrame, batch_size: int = 1000) -> None:
        for typ in sorted(ALLOWED_NODE_TYPES):
            part = nodes_df[nodes_df.type == typ]
            if part.empty:
                continue
            query = f"""
            UNWIND $rows AS row
            MERGE (n:Entity {{id: row.id}})
            SET n:{typ},
                n.name=row.name,
                n.name_norm=row.name_norm,
                n.entity_type=row.type,
                n.aliases=row.aliases,
                n.aliases_norm=row.aliases_norm,
                n.dataset_scope=$scope
            """
            for batch in _batches(part.to_dict("records"), batch_size):
                self.run(query, rows=batch, scope=DATASET_SCOPE)

    def bulk_insert_edges(self, triples_df: pd.DataFrame, batch_size: int = 1000) -> None:
        for rel in sorted(ALLOWED_RELATIONS):
            part = triples_df[triples_df.relation == rel]
            if part.empty:
                continue
            query = f"""
            UNWIND $rows AS row
            MATCH (s:Entity {{id: row.source_id}})
            MATCH (t:Entity {{id: row.target_id}})
            MERGE (s)-[r:{rel} {{source_chunk_id: row.source_chunk_id}}]->(t)
            SET r.published_date=row.published_date,
                r.source_row_id=row.source_row_id,
                r.evidence=row.evidence,
                r.confidence=row.confidence,
                r.dataset_scope=$scope
            """
            cols = [
                "source_id", "target_id", "source_chunk_id", "source_row_id",
                "published_date", "evidence", "confidence",
            ]
            for batch in _batches(part[cols].to_dict("records"), batch_size):
                self.run(query, rows=batch, scope=DATASET_SCOPE)

    def checks(self) -> tuple[dict[str, int], pd.DataFrame]:
        invalid = self.run(
            """
            MATCH ()-[r]->()
            WHERE r.dataset_scope=$scope AND
                  (r.source_chunk_id IS NULL OR r.source_chunk_id='' OR
                   r.published_date IS NULL OR r.published_date='' OR
                   r.evidence IS NULL OR r.evidence='')
            RETURN count(r) AS n
            """,
            scope=DATASET_SCOPE,
        )[0]["n"]
        counts = {
            "nodes": int(self.run("MATCH (n:Entity {dataset_scope:$scope}) RETURN count(n) AS n", scope=DATASET_SCOPE)[0]["n"]),
            "edges": int(self.run("MATCH ()-[r]->() WHERE r.dataset_scope=$scope RETURN count(r) AS n", scope=DATASET_SCOPE)[0]["n"]),
            "invalid_provenance_edges": int(invalid),
        }
        if counts["invalid_provenance_edges"] != 0:
            raise AssertionError(f"Invalid provenance edges: {counts['invalid_provenance_edges']}")
        top = pd.DataFrame(
            self.run(
                """
                MATCH (n:Entity {dataset_scope:$scope})
                OPTIONAL MATCH (n)-[r]-()
                WITH n, count(r) AS degree
                RETURN n.id AS id, n.name AS name, n.entity_type AS type, degree
                ORDER BY degree DESC LIMIT 15
                """,
                scope=DATASET_SCOPE,
            )
        )
        return counts, top

    def node_degree(self, node_id: str) -> int:
        rows = self.run(
            """
            MATCH (n:Entity {id:$id})
            OPTIONAL MATCH (n)-[r]-()
            RETURN count(r) AS degree
            """,
            id=node_id,
        )
        return int(rows[0]["degree"]) if rows else 0

    def recent_edges(self, node_id: str, limit: int) -> list[dict[str, Any]]:
        return self.run(
            """
            MATCH (n:Entity {id:$id})
            MATCH (n)-[r]-(m:Entity)
            WHERE r.dataset_scope=$scope
            RETURN startNode(r).id AS source_id,
                   startNode(r).name AS source_name,
                   startNode(r).entity_type AS source_type,
                   type(r) AS relation,
                   endNode(r).id AS target_id,
                   endNode(r).name AS target_name,
                   endNode(r).entity_type AS target_type,
                   r.source_chunk_id AS source_chunk_id,
                   r.source_row_id AS source_row_id,
                   r.published_date AS published_date,
                   r.evidence AS evidence,
                   m.id AS neighbor_id
            ORDER BY coalesce(r.published_date,'') DESC
            LIMIT $limit
            """,
            id=node_id,
            scope=DATASET_SCOPE,
            limit=int(limit),
        )

    def all_edges(self, limit: int = 20_000) -> pd.DataFrame:
        return pd.DataFrame(
            self.run(
                """
                MATCH (a:Entity)-[r]->(b:Entity)
                WHERE r.dataset_scope=$scope
                RETURN a.id AS source, a.name AS source_name,
                       b.id AS target, b.name AS target_name,
                       type(r) AS relation
                LIMIT $limit
                """,
                scope=DATASET_SCOPE,
                limit=int(limit),
            )
        )

    def write_community_ids(self, rows: list[dict[str, Any]]) -> None:
        for batch in _batches(rows, 1000):
            self.run(
                """
                UNWIND $rows AS row
                MATCH (n:Entity {id:row.id})
                SET n.community_id=row.community_id
                """,
                rows=batch,
            )


class EntityMatcher:
    def __init__(self, nodes_df: pd.DataFrame, embedder: Embedder):
        import faiss

        self.nodes = nodes_df.reset_index(drop=True).copy()
        self.embedder = embedder
        self.vectors = embedder.encode(self.nodes["name"].tolist()) if len(self.nodes) else np.empty((0, 384), dtype="float32")
        self.index = None
        if len(self.nodes):
            self.index = faiss.IndexFlatIP(self.vectors.shape[1])
            self.index.add(self.vectors)

    def match(self, query: str, k: int = 4, threshold: float = 0.38) -> list[dict[str, Any]]:
        if self.index is None:
            return []
        qnorm = normalize_entity(query)
        exact: list[dict[str, Any]] = []
        for row in self.nodes.itertuples(index=False):
            candidates = [row.name_norm] + list(row.aliases_norm or [])
            if any(cand and re.search(rf"(?<!\w){re.escape(cand)}(?!\w)", qnorm) for cand in candidates):
                exact.append({"id": row.id, "name": row.name, "type": row.type, "match": "LEXICAL"})
        qv = self.embedder.encode([query], batch_size=1)
        scores, ids = self.index.search(qv, min(max(k * 3, 6), self.index.ntotal))
        semantic: list[dict[str, Any]] = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0 or float(score) < threshold:
                continue
            row = self.nodes.iloc[int(idx)]
            semantic.append(
                {"id": row.id, "name": row.name, "type": row.type, "match": "SEMANTIC", "score": float(score)}
            )
        merged = {item["id"]: item for item in semantic[:k]}
        for item in exact:
            merged[item["id"]] = item
        return list(merged.values())[: max(k, len(exact))]


class GraphRetriever:
    def __init__(self, store: Neo4jStore, matcher: EntityMatcher):
        self.store = store
        self.matcher = matcher

    @staticmethod
    def textualize(edges: list[dict[str, Any]]) -> str:
        edges = sorted(edges, key=lambda edge: edge.get("published_date") or "", reverse=True)
        lines: list[str] = []
        used = 0
        for edge in edges:
            line = (
                f"{edge['source_name']} [{edge['source_type']}] -{edge['relation']}-> "
                f"{edge['target_name']} [{edge['target_type']}] | "
                f"date={edge.get('published_date') or 'unknown'} | "
                f"source_row={edge.get('source_row_id')} | "
                f"chunk={edge.get('source_chunk_id') or 'unknown'}"
            )
            if edge.get("evidence"):
                line += f" | evidence={normalize_text(edge['evidence'])}"
            if used + len(line) + 1 > MAX_GRAPH_CONTEXT_CHARS:
                break
            lines.append(line)
            used += len(line) + 1
        return "\n".join(lines)

    def retrieve(self, query: str, max_hops: int = 2, edge_limit: int = 50) -> dict[str, Any]:
        seeds = self.matcher.match(query)
        if not seeds:
            return {
                "context": "",
                "edges": pd.DataFrame(),
                "diagnostics": {"reason": "NO_SEED", "matched_seeds": [], "supernode_events": []},
            }
        frontier = deque((seed["id"], 0) for seed in seeds)
        expanded: set[str] = set()
        seen_edges: set[tuple[Any, ...]] = set()
        collected: list[dict[str, Any]] = []
        supernodes: list[dict[str, Any]] = []
        while frontier and len(collected) < GLOBAL_EDGE_CAP:
            node_id, hop = frontier.popleft()
            if node_id in expanded or hop >= max_hops:
                continue
            expanded.add(node_id)
            degree = self.store.node_degree(node_id)
            limit = supernode_edge_limit(
                degree,
                requested=edge_limit,
                threshold=SUPER_NODE_DEGREE,
                cap=SUPER_NODE_EDGE_CAP,
            )
            if degree > SUPER_NODE_DEGREE:
                supernodes.append({"node_id": node_id, "degree": degree, "limit": limit})
            for edge in self.store.recent_edges(node_id, limit):
                key = (
                    edge.get("source_id"), edge.get("relation"), edge.get("target_id"), edge.get("source_chunk_id")
                )
                if key in seen_edges:
                    continue
                seen_edges.add(key)
                collected.append(edge)
                if len(collected) >= GLOBAL_EDGE_CAP:
                    break
                neighbor = edge.get("neighbor_id")
                if neighbor and neighbor not in expanded and hop + 1 < max_hops:
                    frontier.append((neighbor, hop + 1))
        return {
            "context": self.textualize(collected),
            "edges": pd.DataFrame(collected),
            "diagnostics": {
                "matched_seeds": seeds,
                "expanded_nodes": len(expanded),
                "collected_edges": len(collected),
                "supernode_events": supernodes,
            },
        }

    def self_correct(self, query: str) -> dict[str, Any]:
        hop2 = self.retrieve(query, max_hops=2, edge_limit=50)
        d2 = hop2["diagnostics"]
        sufficient2 = bool(d2.get("matched_seeds")) and int(d2.get("collected_edges", 0)) >= 3
        if sufficient2:
            hop2["route"] = "hop2"
            return hop2
        hop3 = self.retrieve(query, max_hops=3, edge_limit=50)
        d3 = hop3["diagnostics"]
        sufficient3 = bool(d3.get("matched_seeds")) and int(d3.get("collected_edges", 0)) >= 3
        hop3["route"] = "hop3" if sufficient3 else "hop3+vector"
        return hop3


def generate_answer(llm: GroqRuntime, question: str, context: str) -> tuple[str, dict[str, Any]]:
    system = (
        "Answer only from supplied context. Be concise but complete. Do not invent facts. "
        "Cite provenance inline as [chunk_id=...] or [source_row=...] whenever possible. "
        "If evidence is insufficient or conflicting, state that explicitly."
    )
    return llm.chat(
        system=system,
        user=f"QUESTION:\n{question}\n\nCONTEXT:\n{context}\n\nANSWER:",
        model=llm.generation_model,
        max_tokens=900,
    )


def answer_flat(question: str, flat: FlatIndex, llm: GroqRuntime, config: RunConfig) -> dict[str, Any]:
    start = time.perf_counter()
    context, docs = flat.retrieve(question, config.vector_top_k_flat)
    answer, usage = generate_answer(llm, question, context)
    return {
        "answer": answer,
        "context": context,
        "retrieved": docs,
        "latency_s": time.perf_counter() - start,
        "total_tokens": usage.get("total_tokens"),
    }


def answer_graph(
    question: str,
    graph: GraphRetriever,
    flat: FlatIndex,
    llm: GroqRuntime,
    config: RunConfig,
) -> dict[str, Any]:
    start = time.perf_counter()
    graph_result = graph.self_correct(question)
    vector_k = 8 if graph_result.get("route") == "hop3+vector" else config.vector_top_k_graph
    vector_context, vector_docs = flat.retrieve(question, vector_k)
    context = f"=== GRAPH ===\n{graph_result['context']}\n\n=== VECTOR ===\n{vector_context}"
    answer, usage = generate_answer(llm, question, context)
    return {
        "answer": answer,
        "context": context,
        "graph_debug": graph_result,
        "vector_docs": vector_docs,
        "latency_s": time.perf_counter() - start,
        "total_tokens": usage.get("total_tokens"),
    }


class Judge:
    def __init__(self, groq: GroqRuntime):
        self.provider = os.getenv("JUDGE_PROVIDER", "openai").strip().lower()
        self.model = os.getenv("JUDGE_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
        self.groq = groq
        self.openai = None
        if self.provider == "openai":
            from openai import OpenAI

            key = os.getenv("OPENAI_API_KEY", "").strip()
            if not key:
                raise RuntimeError("JUDGE_PROVIDER=openai but OPENAI_API_KEY is missing")
            self.openai = OpenAI(api_key=key, timeout=60.0, max_retries=3)
        elif self.provider != "groq":
            raise ValueError(f"Unsupported JUDGE_PROVIDER: {self.provider}")

    def score(self, question: str, reference: str, answer: str, context: str) -> dict[str, Any]:
        system = (
            "You are a strict RAG evaluator. Score only the candidate answer against the reference "
            "and supplied candidate context. Return strict JSON."
        )
        prompt = f"""QUESTION:
{question}

REFERENCE ANSWER:
{reference}

CANDIDATE ANSWER:
{answer}

CANDIDATE CONTEXT:
{context[:14000]}

Return JSON exactly with integer scores 1..5:
{{"comprehensiveness":1,"faithfulness":1,"multi_hop_reasoning":1,"rationale":"2-5 sentences"}}
"""
        if self.provider == "groq":
            obj, _ = self.groq.chat_json(
                system=system,
                user=prompt,
                model=self.model,
                max_tokens=500,
            )
        else:
            response = self.openai.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"},
                max_tokens=500,
            )
            obj = json.loads(response.choices[0].message.content or "{}")
        out: dict[str, Any] = {}
        for key in ["comprehensiveness", "faithfulness", "multi_hop_reasoning"]:
            out[key] = max(1, min(5, int(obj.get(key, 1))))
        out["rationale"] = normalize_text(obj.get("rationale"))
        return out


def evaluate(
    golden: pd.DataFrame,
    flat: FlatIndex,
    graph: GraphRetriever,
    llm: GroqRuntime,
    judge: Judge,
    config: RunConfig,
    output_dir: Path,
) -> pd.DataFrame:
    checkpoint_path = output_dir / "graphrag_eval_checkpoint.csv"
    existing = pd.read_csv(checkpoint_path) if checkpoint_path.exists() else pd.DataFrame()
    completed_ids = set(existing["id"].astype(str)) if not existing.empty and "id" in existing else set()
    rows = existing.to_dict("records") if not existing.empty else []
    for q in golden.head(config.golden_limit).itertuples(index=False):
        if str(q.id) in completed_ids:
            continue
        print(f"[eval] {q.id}: {q.question[:90]}")
        flat_result = answer_flat(q.question, flat, llm, config)
        graph_result = answer_graph(q.question, graph, flat, llm, config)
        flat_judge = judge.score(q.question, q.reference_answer, flat_result["answer"], flat_result["context"])
        graph_judge = judge.score(q.question, q.reference_answer, graph_result["answer"], graph_result["context"])
        diagnostics = graph_result["graph_debug"]["diagnostics"]
        rows.append(
            {
                "id": q.id,
                "group": q.group,
                "question": q.question,
                "reference_answer": q.reference_answer,
                "flat_answer": flat_result["answer"],
                "graph_answer": graph_result["answer"],
                "flat_comprehensiveness": flat_judge["comprehensiveness"],
                "graph_comprehensiveness": graph_judge["comprehensiveness"],
                "flat_faithfulness": flat_judge["faithfulness"],
                "graph_faithfulness": graph_judge["faithfulness"],
                "flat_multi_hop_reasoning": flat_judge["multi_hop_reasoning"],
                "graph_multi_hop_reasoning": graph_judge["multi_hop_reasoning"],
                "flat_latency_s": flat_result["latency_s"],
                "graph_latency_s": graph_result["latency_s"],
                "flat_total_tokens": flat_result.get("total_tokens"),
                "graph_total_tokens": graph_result.get("total_tokens"),
                "flat_judge_rationale": flat_judge["rationale"],
                "graph_judge_rationale": graph_judge["rationale"],
                "graph_route": graph_result["graph_debug"].get("route"),
                "graph_edges": int(diagnostics.get("collected_edges", 0)),
                "graph_supernode_events": len(diagnostics.get("supernode_events", [])),
                "graph_matched_seeds": json.dumps(diagnostics.get("matched_seeds", []), ensure_ascii=False),
            }
        )
        pd.DataFrame(rows).to_csv(checkpoint_path, index=False)
    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "graphrag_eval_results.csv", index=False)
    return result


def comparison_table(eval_df: pd.DataFrame) -> pd.DataFrame:
    metric_map = {
        "Comprehensiveness": ("flat_comprehensiveness", "graph_comprehensiveness"),
        "Faithfulness": ("flat_faithfulness", "graph_faithfulness"),
        "Multi-hop reasoning": ("flat_multi_hop_reasoning", "graph_multi_hop_reasoning"),
        "Latency (s)": ("flat_latency_s", "graph_latency_s"),
        "Token usage": ("flat_total_tokens", "graph_total_tokens"),
    }
    rows: list[dict[str, Any]] = []
    for group, grouped in eval_df.groupby("group"):
        for metric, (flat_col, graph_col) in metric_map.items():
            flat_value = pd.to_numeric(grouped[flat_col], errors="coerce").mean()
            graph_value = pd.to_numeric(grouped[graph_col], errors="coerce").mean()
            delta = graph_value - flat_value if pd.notna(flat_value) and pd.notna(graph_value) else np.nan
            rows.append(
                {
                    "question_group": group,
                    "metric": metric,
                    "flat_rag": round(float(flat_value), 3) if pd.notna(flat_value) else np.nan,
                    "graph_rag": round(float(graph_value), 3) if pd.notna(graph_value) else np.nan,
                    "delta_graph_minus_flat": round(float(delta), 3) if pd.notna(delta) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def build_community_reports(
    store: Neo4jStore,
    nodes_df: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    import networkx as nx

    edge_df = store.all_edges()
    if edge_df.empty:
        empty = pd.DataFrame(columns=["community_id", "size", "hubs", "dominant_relations", "report"])
        empty.to_csv(output_dir / "community_reports.csv", index=False)
        return empty
    graph = nx.Graph()
    graph.add_edges_from(edge_df[["source", "target"]].itertuples(index=False, name=None))
    communities = nx.community.louvain_communities(graph, seed=SEED)
    membership: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    node_lookup = nodes_df.set_index("id")["name"].to_dict()
    for community_id, members in enumerate(communities):
        members = set(members)
        membership.extend({"id": node_id, "community_id": int(community_id)} for node_id in members)
        degree_rank = sorted(members, key=lambda node: graph.degree(node), reverse=True)[:8]
        hubs = [node_lookup.get(node, node) for node in degree_rank]
        relation_counts = Counter(
            edge.relation
            for edge in edge_df.itertuples(index=False)
            if edge.source in members and edge.target in members
        )
        dominant = [name for name, _ in relation_counts.most_common(5)]
        report = (
            f"Community {community_id} contains {len(members)} entities. "
            f"Main hubs: {', '.join(hubs)}. Dominant relation types: {', '.join(dominant) or 'none'}."
        )
        reports.append(
            {
                "community_id": int(community_id),
                "size": len(members),
                "hubs": " | ".join(hubs),
                "dominant_relations": " | ".join(dominant),
                "report": report,
            }
        )
    store.write_community_ids(membership)
    reports_df = pd.DataFrame(reports).sort_values("size", ascending=False).reset_index(drop=True)
    reports_df.to_csv(output_dir / "community_reports.csv", index=False)
    return reports_df


def global_search_community_reports(
    query: str,
    reports_df: pd.DataFrame,
    embedder: Embedder,
    k: int = 5,
) -> pd.DataFrame:
    if reports_df.empty:
        return reports_df.copy()
    vectors = embedder.encode(reports_df["report"].tolist())
    query_vector = embedder.encode([query], batch_size=1)[0]
    scores = vectors @ query_vector
    top = np.argsort(-scores)[: min(k, len(scores))]
    out = reports_df.iloc[top].copy()
    out["similarity"] = scores[top]
    return out


def write_bonus_metrics(
    near_audit: pd.DataFrame,
    community_reports: pd.DataFrame,
    eval_df: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    route_counts = eval_df.get("graph_route", pd.Series(dtype=str)).value_counts().to_dict()
    rows = [
        {"bonus": "Near-Dedup SimHash/LSH", "metric": "near_duplicates_removed", "value": len(near_audit)},
        {"bonus": "Global Search Community Reports", "metric": "communities", "value": len(community_reports)},
        {"bonus": "Self-Correction Graph Retrieval", "metric": "hop2", "value": int(route_counts.get("hop2", 0))},
        {"bonus": "Self-Correction Graph Retrieval", "metric": "hop3", "value": int(route_counts.get("hop3", 0))},
        {"bonus": "Self-Correction Graph Retrieval", "metric": "hop3+vector", "value": int(route_counts.get("hop3+vector", 0))},
    ]
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "bonus_metrics.csv", index=False)
    return df


def _mean_metric(eval_df: pd.DataFrame, prefix: str, metric: str) -> float:
    return float(pd.to_numeric(eval_df[f"{prefix}_{metric}"], errors="coerce").mean())


def generate_reports(
    eval_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    top_degree_df: pd.DataFrame,
    audit_df: pd.DataFrame,
    near_audit: pd.DataFrame,
    community_reports: pd.DataFrame,
    graph_counts: dict[str, int],
    reports_dir: Path,
) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    rejected = audit_df[audit_df.get("decision", pd.Series(dtype=str)).eq("REJECT_GUARD")].copy() if not audit_df.empty else pd.DataFrame()
    rejected = rejected.sort_values("similarity", ascending=False) if not rejected.empty else rejected
    rejected_example = rejected.iloc[0].to_dict() if not rejected.empty else None
    top3 = top_degree_df.head(3).to_dict("records") if not top_degree_df.empty else []

    quality_cols = [
        ("comprehensiveness", "Comprehensiveness"),
        ("faithfulness", "Faithfulness"),
        ("multi_hop_reasoning", "Multi-hop reasoning"),
    ]
    metrics_lines = []
    for key, label in quality_cols:
        metrics_lines.append(
            f"- **{label}:** Flat={_mean_metric(eval_df, 'flat', key):.2f}, "
            f"Graph={_mean_metric(eval_df, 'graph', key):.2f}"
        )
    flat_latency = float(pd.to_numeric(eval_df["flat_latency_s"], errors="coerce").mean())
    graph_latency = float(pd.to_numeric(eval_df["graph_latency_s"], errors="coerce").mean())
    flat_tokens = float(pd.to_numeric(eval_df["flat_total_tokens"], errors="coerce").mean())
    graph_tokens = float(pd.to_numeric(eval_df["graph_total_tokens"], errors="coerce").mean())

    scored = eval_df.copy()
    scored["flat_quality"] = scored[["flat_comprehensiveness", "flat_faithfulness", "flat_multi_hop_reasoning"]].mean(axis=1)
    scored["graph_quality"] = scored[["graph_comprehensiveness", "graph_faithfulness", "graph_multi_hop_reasoning"]].mean(axis=1)
    scored["delta"] = scored["graph_quality"] - scored["flat_quality"]
    best = scored.sort_values("delta", ascending=False).iloc[0]
    worst = scored.sort_values("delta", ascending=True).iloc[0]

    supernode_lines = "\n".join(
        f"- {idx + 1}. {row.get('name')} ({row.get('type')}), degree={row.get('degree')}"
        for idx, row in enumerate(top3)
    ) or "- Không có dữ liệu."
    rejected_line = (
        f"{rejected_example.get('left')} ↔ {rejected_example.get('right')} "
        f"(sim={float(rejected_example.get('similarity', 0)):.3f})"
        if rejected_example
        else "Không có cặp REJECT_GUARD trong sample hiện tại."
    )

    technical = f"""# Technical Defense — Lab 19 GraphRAG vs Flat RAG

## 1. Coreference sai ở tình huống nào?
False coreference xảy ra khi đại từ/generic mention có nhiều antecedent hợp lệ trong cùng chunk. Pipeline dùng conservative trigger và giữ nguyên văn bản khi mơ hồ; unresolved mentions được checkpoint để audit. Điều này ưu tiên precision vì một false coreference có thể tạo false edge lan truyền trong graph.

## 2. Entity-resolution threshold là bao nhiêu, vì sao?
Vector candidate threshold là **0.90**. Sau ANN candidate search, lexical/type guard vẫn bắt buộc. Mức này ưu tiên precision; false merge nguy hiểm hơn false split trong knowledge graph dùng cho reasoning.

## 3. Candidate similarity cao nhưng không nên merge?
Ví dụ audit thực nghiệm: **{rejected_line}**. Guard đặc biệt chặn company-vs-product prefix và person chỉ trùng họ.

## 4. Top 3 super-node và degree?
{supernode_lines}

## 5. Vì sao ưu tiên edge mới nhất có thể đúng/sai?
Đúng cho câu hỏi trạng thái hiện tại và kiểm soát context explosion; sai với câu hỏi lịch sử vì cạnh cũ có thể là evidence quyết định. Vì vậy policy chỉ kích hoạt khi degree >100, cap 50, đồng thời giữ provenance/date để failure analysis.

## 6. Flat RAG thắng nhóm nào?
Kết quả theo group nằm trong `outputs/graphrag_vs_flatrag_summary.csv`. Ca có Graph-minus-Flat thấp nhất là **{worst['id']} ({worst['group']})**, delta quality={worst['delta']:.2f}.

## 7. GraphRAG thắng nhóm nào?
Ca có Graph-minus-Flat cao nhất là **{best['id']} ({best['group']})**, delta quality={best['delta']:.2f}. Graph traversal hữu ích khi evidence phải nối qua nhiều entity/document.

## 8. Latency/token trade-off?
Flat latency trung bình={flat_latency:.2f}s; GraphRAG={graph_latency:.2f}s. Flat token trung bình={flat_tokens:.1f}; GraphRAG={graph_tokens:.1f}. GraphRAG trả giá bằng traversal + graph context để tăng khả năng multi-hop.

## 9. AI Coding Agent đề xuất gì mà không dùng, vì sao?
Không dùng pairwise cosine O(N²) cho near-dedup/entity resolution. Near-dedup dùng SimHash+LSH; entity resolution dùng FAISS ANN + lexical guard. Không dùng golden metadata để chọn extraction chunks vì đó là benchmark leakage.

## 10. Scale lên toàn bộ dataset: bottleneck đầu tiên là gì?
LLM extraction/rate limit là bottleneck đầu tiên, sau đó mới tới embedding/indexing và graph fan-out. Hướng scale: durable queue + checkpoint, async workers theo quota, ANN/blocking cho resolution, batch UNWIND, partition/community summaries và cache retrieval.

## Empirical metrics
{chr(10).join(metrics_lines)}
- Graph nodes={graph_counts.get('nodes', 0)}, edges={graph_counts.get('edges', 0)}, invalid provenance={graph_counts.get('invalid_provenance_edges', -1)}.
"""
    (reports_dir / "technical_defense.md").write_text(technical, encoding="utf-8")

    failure = f"""# Failure Analysis — Lab 19

## Case 1 — GraphRAG advantage
- **Question:** {best['id']} — {best['question']}
- **Flat answer:** {best['flat_answer']}
- **Graph answer:** {best['graph_answer']}
- **Quality delta:** {best['delta']:.2f}
- **Root cause interpretation:** Flat retrieval ranks chunks independently; GraphRAG can connect canonical entities and provenance-bearing edges before combining vector evidence.

## Case 2 — GraphRAG failure / weakest relative case
- **Question:** {worst['id']} — {worst['question']}
- **Flat answer:** {worst['flat_answer']}
- **Graph answer:** {worst['graph_answer']}
- **Quality delta:** {worst['delta']:.2f}
- **Likely root causes to audit:** missing extraction edge, imperfect seed/entity resolution, or super-node recency cap. `graph_route`, matched seeds and edge count are exported per question for tracing rather than guessing.
"""
    (reports_dir / "failure_analysis.md").write_text(failure, encoding="utf-8")

    reflection = f"""# Reflection & Action Plan — ChuNguyenTuanAnh

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
- Near-duplicates removed: {len(near_audit)}
- Community reports built: {len(community_reports)}
"""
    reflection_path = reports_dir / "reflection_ChuNguyenTuanAnh.md"
    reflection_path.write_text(reflection, encoding="utf-8")

    combined = (
        "# Lab 19 Report — GraphRAG vs Flat RAG\n\n"
        + technical
        + "\n\n---\n\n"
        + failure
        + "\n\n---\n\n"
        + reflection
    )
    (reports_dir / "lab_report.md").write_text(combined, encoding="utf-8")


def run_lab(mode: str) -> dict[str, Any]:
    config = RunConfig.for_mode(mode)
    output_dir = Path("outputs")
    reports_dir = Path("reports")
    output_dir.mkdir(exist_ok=True)
    reports_dir.mkdir(exist_ok=True)
    print(f"[run] mode={config.mode}; source_scope=FIRST_5000_ROWS_ONLY")

    llm = GroqRuntime(config)
    news_df, chunks_df, near_audit = prepare_data(output_dir)
    extraction_source = select_extraction_source(chunks_df, limit=config.extraction_max_chunks)
    extraction_source.to_csv(output_dir / "extraction_source.csv", index=False)
    print(
        f"[extract] selected {len(extraction_source)} chunks across source rows "
        f"{int(extraction_source.source_row_id.min()) if len(extraction_source) else 'n/a'}.."
        f"{int(extraction_source.source_row_id.max()) if len(extraction_source) else 'n/a'}"
    )
    resolved_source = resolve_coreferences(extraction_source, llm, config, output_dir)
    raw_triples, extraction_errors = extract_triples(resolved_source, llm, config, output_dir)
    if raw_triples.empty:
        raise RuntimeError("Extraction produced zero valid triples")

    embedder = Embedder()
    entity_map, audit_df = build_resolution_map(raw_triples, embedder)
    triples_df = canonicalize_triples(raw_triples, entity_map)
    nodes_df = build_nodes(triples_df)
    audit_df.to_csv(output_dir / "entity_resolution_audit.csv", index=False)
    triples_df.to_csv(output_dir / "canonical_triples.csv", index=False)
    nodes_df.to_csv(output_dir / "nodes.csv", index=False)

    store = Neo4jStore()
    try:
        store.setup_schema()
        store.reset_scope()
        store.bulk_insert_nodes(nodes_df)
        store.bulk_insert_edges(triples_df)
        graph_counts, top_degree = store.checks()
        top_degree.to_csv(output_dir / "supernode_diagnostics.csv", index=False)
        (output_dir / "graph_checks.json").write_text(
            json.dumps(graph_counts, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        flat = FlatIndex(chunks_df, embedder)
        matcher = EntityMatcher(nodes_df, embedder)
        graph = GraphRetriever(store, matcher)
        golden = pd.read_csv(GOLDEN_PATH)
        judge = Judge(llm)
        eval_df = evaluate(golden, flat, graph, llm, judge, config, output_dir)
        summary_df = comparison_table(eval_df)
        summary_df.to_csv(output_dir / "graphrag_vs_flatrag_summary.csv", index=False)

        communities = build_community_reports(store, nodes_df, output_dir)
        # Demonstrate global-search routing on one broad cross-doc query without using gold metadata for indexing.
        cross_doc = golden[golden.group.eq("cross-doc")].head(1)
        if not cross_doc.empty and not communities.empty:
            query = cross_doc.iloc[0].question
            global_hits = global_search_community_reports(query, communities, embedder)
            global_hits.insert(0, "query", query)
            global_hits.to_csv(output_dir / "community_global_search_demo.csv", index=False)
        else:
            pd.DataFrame().to_csv(output_dir / "community_global_search_demo.csv", index=False)

        bonus_df = write_bonus_metrics(near_audit, communities, eval_df, output_dir)
        generate_reports(
            eval_df,
            summary_df,
            top_degree,
            audit_df,
            near_audit,
            communities,
            graph_counts,
            reports_dir,
        )
        manifest = {
            "mode": config.mode,
            "source_policy": "FIRST_5000_ROWS_ONLY",
            "source_rows_downloaded": 5000,
            "articles_after_dedup": len(news_df),
            "chunks_indexed": len(chunks_df),
            "chunks_llm_extracted": len(extraction_source),
            "triples": len(triples_df),
            "nodes": len(nodes_df),
            "golden_questions_evaluated": len(eval_df),
            "entity_audit_rows": len(audit_df),
            "near_dedup_rows": len(near_audit),
            "communities": len(communities),
            "invalid_provenance_edges": graph_counts["invalid_provenance_edges"],
        }
        (output_dir / "run_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print("[run] completed", json.dumps(manifest, ensure_ascii=False))
        return manifest
    finally:
        store.close()
