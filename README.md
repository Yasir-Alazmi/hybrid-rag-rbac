# Hybrid RAG with Role-Based Access

[![CI](https://github.com/Yasir-Alazmi/hybrid-rag-rbac/actions/workflows/ci.yml/badge.svg)](https://github.com/Yasir-Alazmi/hybrid-rag-rbac/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Prometheus](https://img.shields.io/badge/Prometheus-Metrics-E6522C?style=flat-square&logo=prometheus&logoColor=white)](https://prometheus.io/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat-square&logo=docker&logoColor=white)](https://www.docker.com/)

A small reference implementation of role-based hybrid Retrieval-Augmented Generation (RAG): JWT authentication with role-based access control applied inside retrieval, BM25 + dense retrieval fused with Reciprocal Rank Fusion, prompt-injection and PII guards, and a role-scoped semantic cache. It is a learning and architecture project: it has not been deployed or load-tested, and its default components are lightweight stand-ins (see [What the components really are](#what-the-components-really-are)).

### Result: BEIR SciFact (5,183 abstracts, 300 queries, CPU)

| Retrieval | nDCG@10 | Recall@100 | p50 ms |
|---|---|---|---|
| BM25 only | 0.661 | 0.876 | 13.84 |
| Dense (all-MiniLM-L6-v2) | 0.648 | 0.925 | 22.32 |
| **Hybrid BM25 + MiniLM (RRF)** | **0.679** | **0.955** | 44.81 |

Hybrid fusion finds more of the relevant documents (Recall@100 0.876 → 0.955) for about 31 ms of extra p50 latency over BM25.
Baselines match published BEIR numbers. Full table, including the configurations that did *not* help: [below](#retrieval-quality-on-beir-scifact-scriptsevaluate_beirpy). To reproduce it, see [Quickstart step 3](#3-reproduce-the-beir-scifact-benchmark).

### What I learned

- **Fusion mainly helps recall.** Hybrid retrieval adds +0.018 nDCG@10 over BM25 but lifts Recall@100 from 0.876 to 0.955, which is what matters for a RAG first stage.
- **A non-semantic default halves quality.** With the default hashing embedding, hybrid retrieval scores 0.346 nDCG@10, about half of BM25 alone (0.661).
- **The heuristic re-ranker hurts a strong first stage.** It adds +0.145 on the hashing hybrid but costs -0.032 on the neural hybrid.

---

## What the components really are

| Component | Name in the code | What it actually is |
| :--- | :--- | :--- |
| Default embedding | `DeterministicHashingEmbedding` | Token sign-hash projection, not semantic; for CI and tests |
| Neural embedding | `SentenceTransformerEmbedding` | `all-MiniLM-L6-v2` when installed |
| Re-ranker | `HeuristicReranker` (`src/reranker/heuristic.py`) | Hand-written lexical heuristic (coverage, density, position), not a trained cross-encoder; the old name `CrossEncoderReranker` is kept as an import alias |
| Default generator | `DeterministicGroundedGenerator` | Extractive: stitches retrieved sentences and adds citations; no language model |
| Optional generator | `LLMAnswerGenerator` | Calls an OpenAI-compatible or Ollama endpoint; not exercised in the benchmark |
| Vector store | `DenseVectorStore` | In-memory brute-force cosine scan with JSON persistence |
| Evaluation corpus (default) | Golden Reference Corpus | 2 documents and 4 queries |

---

## Architecture

```
[ Client Request ] ──► [ Sliding Window Rate Limiter (200 req/min) ]
                               │
                               ▼
        [ Bearer token / JWT verification ]
        (HMAC-SHA256 JWT / API Tokens: PUBLIC < INTERNAL < CONFIDENTIAL < RESTRICTED)
                               │
                               ▼
        [ FastAPI endpoints (/api/v1/query, /ingest, /documents/{id}, /eval) ]
                                │
        ┌───────────────────────┴───────────────────────┐
        ▼                                               ▼
 [ Injection Detector ]                         [ PII Sanitizer ]
 (System Boundary Heuristics)                   (National ID, Cards, Keys)
        │                                               │
        └───────────────────────┬───────────────────────┘
                                ▼
             [ Role-Scoped Bounded Semantic Cache ] ──(Hit)──► Return cached answer
                                │ (Miss)
                                ▼
            [ Role-Based Access Control (RBAC Filter) ]
            (Clearance Level >= Document Classification)
                                │
                                ▼
              [ Hybrid Retrieval Engine (RRF Fusion) ]
              ├── BM25Okapi Lexical Index (k1=1.5, b=0.75)
              └── Dense Vector Store (Pluggable Embedding Engine)
                                │
                                ▼
                  [ Heuristic Re-Ranking ]
                  (lexical heuristic, not a trained model)
                                │
                                ▼
             [ Pluggable Answer Generation Abstraction ]
             ├── Extractive answer with citations (offline, no LLM)
             └── Live LLM Generator (OpenAI / Local Ollama via HTTPX)
                                │
                                ▼
             [ Prometheus Telemetry (/api/v1/metrics) ]
             ├── rag_stage_duration_seconds{stage="retrieval"}
             └── rag_stage_duration_seconds{stage="generation"}
```

---

## Components

### 1. Identity, RBAC & API Hardening (`src/core/security.py`, `src/api/main.py`)
- **HMAC-SHA256 JWT Decoding**: Validates cryptographic signatures, token expiration (`exp`), and tenant clearance claims (`sub`, `role`, `clearance`).
- **Protected Administrative & Evaluation Endpoints**: `POST /api/v1/ingest`, `DELETE /api/v1/documents/{id}`, and `POST /api/v1/eval` require Bearer authorization with $\ge$ `CONFIDENTIAL` clearance. Unauthenticated calls return `HTTP 401 Unauthorized`; insufficient clearance calls return `HTTP 403 Forbidden`.
- **Strict Header Enforcement**: Unrecognized or malformed tokens explicitly return `401 Unauthorized` rather than silently degrading permissions.
- **Fail-Closed Defaults**: an unknown, empty or missing role gets `PUBLIC`; a chunk with an unknown or missing classification is never returned to anyone; ingest rejects unknown classification labels (`422`). A JWT `clearance` claim that is not an integer from 0 to 3 is rejected with `401`.
- **One Clearance per Request**: the clearance resolved from the bearer token is the only value used for both retrieval filtering and cache scoping, so the two cannot disagree. For a JWT, the signed `clearance` claim wins over the role name; without a claim, the clearance comes from the role.
- **Demo Tokens Outside Production Only**: the demo tokens in `src/core/security.py` are loaded only when `APP_ENV` is not `production`/`prod`. In production, only signed JWTs and tokens from `API_TOKENS_JSON` are accepted.
- **Restricted CORS Policy**: Configurable, environment-governed allowed origins with strict method (`GET`, `POST`, `DELETE`, `OPTIONS`) and header filtering, with no wildcard origins on credentialed endpoints.

### 2. Dual-Mode Embedding Engine Interface (`src/retrieval/embeddings.py`)
- **Interface Contract (`BaseEmbeddingEngine`)**: Decouples the vector store and cache from underlying embedding algorithms.
- **`DeterministicHashingEmbedding` (128-d, the default)**: a zero-dependency sign-hash projection of tokens. It is **not semantic** (it only encodes which words occur) and is meant for offline CI and reproducible tests. On a real benchmark it performs far below BM25 and lowers the hybrid result (see Results); use the neural engine for real retrieval.
- **`SentenceTransformerEmbedding` (384-d)**: neural embeddings using Sentence Transformers (`all-MiniLM-L6-v2`); falls back to the hashing engine if the package is missing. This is the configuration that gives the benchmark gains below.

### 3. Pluggable Answer Generation Abstraction (`src/generation/generator.py`)
- **Protocol Contract (`BaseAnswerGenerator`)**: Supports interchangeable synthesis backends via factory configuration (`GENERATION_PROVIDER`).
- **`DeterministicGroundedGenerator`**: Extractive factual synthesis that combines facts across retrieved chunks, appends verified section citations `[Document Title - Section]`, and deterministically refuses insufficient context (*"Insufficient enterprise documentation was retrieved to answer this query with verifiable certainty"*).
- **`LLMAnswerGenerator`**: Pluggable live generation adapter calling OpenAI-compatible APIs or local Ollama endpoints via `httpx`, with a system prompt that tells the model to answer only from the retrieved context.

### 4. Role-Scoped Semantic Cache
- Caches query vectors and responses with bounded LRU eviction (1,000 entries).
- **Privilege Boundary Enforcement**: entries are keyed by `(clearance_level, query)` and served only to callers with exactly the same clearance. A lower-clearance user never gets a higher-clearance answer, and a higher-clearance user never gets a lower-clearance answer, which may be missing documents they are allowed to see.
- **Invalidation**: the whole cache is dropped on every ingest and document delete, so cached answers never cite removed documents or miss new ones.

### 5. Evaluation & Stage Latency Tracking (`src/evaluation/`, `scripts/evaluate_rag.py`)
- Computes retrieval and generation metrics:
  - Context Precision@K, Context Recall@K
  - Mean Reciprocal Rank (MRR)
  - Normalized Discounted Cumulative Gain (NDCG@5)
  - Answer Faithfulness & Keyword Relevance
- Separates **retrieval latency** from **generation latency** in Prometheus telemetry and API responses.

---

## Benchmarks and evaluation

### Retrieval quality on BEIR SciFact (`scripts/evaluate_beir.py`)

BEIR **SciFact**: 5,183 scientific abstracts, 300 test queries with relevance judgements. Each row uses this repository's own components on the same corpus. Latency is one query at a time on a CPU (Windows 11, Python 3.12, no GPU); full output in [`results/beir_scifact.md`](results/beir_scifact.md).

| Configuration | nDCG@10 | MRR@10 | Recall@100 | p50 latency (ms) |
| :--- | :--- | :--- | :--- | :--- |
| BM25 only | 0.661 | 0.628 | 0.876 | 13.84 |
| Dense, default hashing embedding | 0.172 | 0.152 | 0.439 | 9.02 |
| Hybrid (BM25 + hashing dense), default settings | 0.346 | 0.320 | 0.844 | 25.3 |
| Hybrid (hashing) + heuristic re-ranker | 0.491 | 0.456 | 0.844 | 44.8 |
| Dense, MiniLM neural embedding | 0.648 | 0.607 | 0.925 | 22.32 |
| **Hybrid (BM25 + MiniLM), RRF** | 0.679 | 0.638 | 0.955 | 44.81 |
| Hybrid (MiniLM) + heuristic re-ranker | 0.647 | 0.611 | 0.955 | 51.68 |

Sanity check: BM25 (0.661) and MiniLM dense (0.648) are close to the published BEIR numbers for SciFact (about 0.665 and 0.645), so the harness is measuring correctly.

What the numbers say:
- **With the neural embedding, hybrid retrieval is the best configuration** (0.679), a modest gain over BM25 alone (+0.018 nDCG@10) and a clearer gain in Recall@100 (0.955 vs 0.876).
- **The default hashing configuration hurts.** Hybrid with hashing scores 0.346, about half of BM25 alone, because the default fusion weight (0.60) gives most of the vote to a non-semantic signal. The default is for offline tests only.
- **The heuristic re-ranker helps weak first stages and hurts strong ones**: +0.145 on the hashing hybrid, -0.032 on the neural hybrid. It is a hand-written scoring function (query-term coverage, density, position), not a trained cross-encoder.
- **Latency is tens of milliseconds at 5,183 documents** (p50 13.8 ms for BM25, 44.8 ms for neural hybrid), not the 2 ms measured below on a two-document corpus. The vector store is a brute-force in-memory scan, so it will not scale to millions of documents without a real index.

<details>
<summary>Latency on the two-document sample corpus (<code>scripts/benchmark.py</code>)</summary>

**Corpus: two sample documents.** These latencies measure the pipeline overhead on a tiny corpus and say nothing about performance at realistic size (see the benchmark above). Measured locally across 100 consecutive requests on Windows (AMD64, Python 3.13):

| Performance Metric | Cold Hybrid Retrieval + Synthesis | Semantic Cache Hit |
| :--- | :--- | :--- |
| **p50 Latency (Median)** | **2.37 ms** | **2.06 ms** |
| **p95 Latency** | **2.86 ms** | **2.44 ms** |
| **p99 Latency** | **4.02 ms** | **2.67 ms** |
| **Mean Execution Time** | **2.44 ms** | **2.08 ms** |

> **Benchmark Scope Note**: Latency covers full Ingestion -> PII Sanitization -> Injection Screening -> Role-Scoped Cache -> Hybrid Retrieval -> Heuristic Re-Ranker -> Deterministic Grounded Synthesis. Calls to an external LLM API are not included.

</details>

<details>
<summary>Regression evaluation on the sample corpus (<code>scripts/evaluate_rag.py</code>)</summary>

**This is a regression test, not a quality estimate.** It runs 4 queries against 2 sample documents (with ground-truth documents and PII/injection checks); with n = 4 each metric moves in large steps. Use the SciFact benchmark above for retrieval quality.

| Evaluation Metric | Aggregate Score |
| :--- | :--- |
| **Mean Context Precision@K** | **0.833** |
| **Mean Context Recall@K** | **1.000** |
| **Mean Reciprocal Rank (MRR)** | **0.833** |
| **Mean NDCG@5** | **0.875** |
| **Answer Faithfulness** | **0.875** |
| **Answer Citation Correctness** | **1.000** |
| **Mean Retrieval Latency** | **0.22 ms** |
| **Mean Generation Latency (Local)** | **0.10 ms** |

</details>

---

## Scaling notes (design only, not implemented)

Everything runs in one process with in-memory stores. Moving to several nodes would mean replacing each component roughly as follows:

```
[ In-Memory Reference ]                [ Possible replacement ]
DenseVectorStore (JSON + SHA256)  ──►  PostgreSQL + pgvector / Qdrant Cloud
BM25Index (Inverted Index)        ──►  Elasticsearch / OpenSearch
SemanticCache (In-Memory LRU)     ──►  Redis Cluster with RedisVL (RediSearch)
Pre-Shared Bearer Tokens          ──►  Keycloak / Azure Entra ID (OIDC / OAuth2)
Prometheus Python Collector       ──►  OpenTelemetry Collector + Prometheus + Grafana
```

Detailed migration schemas, SQL DDL, and configuration examples are documented in [`docs/architecture.md`](docs/architecture.md).

---

## Directory Structure

```
hybrid-rag-rbac/
├── .github/workflows/ci.yml       # GitHub Actions CI matrix (Python 3.10 & 3.11)
├── configs/config.yaml            # Retrieval, cache and security settings
├── data/
│   ├── sample_documents/         # Two sample policy documents
│   └── storage/                  # Persistent vector snapshots with SHA256 verification
├── docs/
│   ├── architecture.md           # Design trade-offs and scaling notes
│   └── runbook.md                # Day-2 operations, key rotation, and Prometheus alerts
├── results/
│   ├── beir_scifact.json         # BEIR SciFact retrieval benchmark (raw)
│   ├── beir_scifact.md           # BEIR SciFact retrieval benchmark (table)
│   ├── benchmark_report.json     # Latency percentiles on the 2-document sample corpus
│   └── evaluation_report.json    # Precision, Recall, MRR, and NDCG evaluation metrics
├── scripts/
│   ├── benchmark.py              # Latency benchmark harness (p50/p95/p99)
│   ├── evaluate_beir.py          # BEIR SciFact retrieval benchmark
│   └── evaluate_rag.py           # Regression evaluation on the 2-document sample corpus
├── src/
│   ├── api/                      # FastAPI routes, schemas, and rate-limiting middleware
│   ├── cache/                    # Role-scoped bounded LRU semantic query cache
│   ├── core/                     # JWT auth, RBAC, Prometheus stage metrics
│   ├── evaluation/               # Precision, Recall, MRR, NDCG@K, faithfulness scoring
│   ├── generation/               # Pluggable generation abstraction (Grounded + LLM)
│   ├── guardrails/               # PII sanitization and adversarial injection screening
│   ├── ingestion/                # Document parsing and recursive token chunker
│   ├── reranker/                 # Lexical heuristic re-ranker (heuristic.py; not a neural cross-encoder)
│   └── retrieval/                # Dual-mode embeddings, BM25, and hybrid RRF retriever
├── tests/                        # pytest suite: 85 tests in 13 files
├── Dockerfile                    # Multi-stage container (non-root user)
├── docker-compose.yml            # Local service orchestration
├── pyproject.toml                # Build system & pytest configuration
└── requirements.txt              # Dependencies
```

---

## Quickstart

### 1. Installation
```bash
git clone https://github.com/Yasir-Alazmi/hybrid-rag-rbac.git
cd hybrid-rag-rbac
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the tests
```bash
python -m pytest tests/ -v
```

### 3. Reproduce the BEIR SciFact benchmark
The neural rows need `sentence-transformers` (the `embeddings` extra in `pyproject.toml`). Without it the neural engine falls back to the hashing embedding, so the MiniLM rows cannot be reproduced.
```bash
pip install -e ".[embeddings]"
python scripts/evaluate_beir.py               # downloads SciFact (about 3 MB) to data/beir/ and the MiniLM model
python scripts/evaluate_beir.py --no-neural   # BM25 and hashing rows only; no extra install needed
```
Both write `results/beir_scifact.json` and `results/beir_scifact.md`. Quality numbers should match; latency depends on your machine.

### 4. Run the sample-corpus evaluation and latency benchmark
```bash
python scripts/evaluate_rag.py
python scripts/benchmark.py
```

### 5. Start the API server
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```
- Interactive Swagger UI: `http://localhost:8000/docs`
- Prometheus metrics: `http://localhost:8000/api/v1/metrics`
- Health probe: `http://localhost:8000/api/v1/health`

### 6. Run with Docker
`docker-compose.yml` runs the API with `APP_ENV=production`, so it needs a real `JWT_SECRET_KEY` (the app refuses to start with the default one, and compose stops with an error if the variable is unset). The demo tokens are disabled in this mode; pass your own tokens through `API_TOKENS_JSON` or use signed JWTs.
```bash
export JWT_SECRET_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(48))')"
export API_TOKENS_JSON='{"my-admin-token": ["admin", 3]}'   # optional
docker compose up --build
```
The container runs a single uvicorn worker on purpose. The indexes and the semantic cache live in process memory, so with several workers each process would hold its own copy: an ingest or delete handled by one worker would not reach the others, and they would serve different, stale results. Scaling out needs shared stores first (see [Scaling notes](#scaling-notes-design-only-not-implemented)).

### 7. Query with Bearer Authentication (cURL)
The tokens below (`token-ciso-root`, `token-admin-restricted`) are hard-coded demo tokens defined in `src/core/security.py`. They are loaded only when `APP_ENV` is not production (for example with the local `uvicorn` command above), so they do not work under `docker compose`. Use signed JWTs or `API_TOKENS_JSON` anywhere else.

```bash
# Query with a demo admin token
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer token-ciso-root" \
  -d '{"query": "What are the restricted incident recovery protocols?", "top_k": 3}'

# Delete a Document (Requires Admin/Manager Bearer Token)
curl -X DELETE http://localhost:8000/api/v1/documents/enterprise_cloud_security_sla \
  -H "Authorization: Bearer token-admin-restricted"

# Run Evaluation Benchmark (Requires Admin/Manager Bearer Token)
curl -X POST http://localhost:8000/api/v1/eval \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer token-admin-restricted" \
  -d '{"retrieved_ids": ["c1"], "ground_truth_ids": ["c1"], "answer": "Annual leave is 30 days.", "context_chunks": ["Annual vacation entitlement is 30 days."]}'
```

---

## Author & Contact

**Yasir Alazmi**
- Email: [yasir.alazmi@outlook.sa](mailto:yasir.alazmi@outlook.sa)
- LinkedIn: [yasir-alazmi-471832436](https://www.linkedin.com/in/yasir-alazmi-471832436)
- GitHub: [Yasir-Alazmi](https://github.com/Yasir-Alazmi)

_Design, evaluation and review by the author; parts of the implementation were written with AI coding assistants and then reviewed and tested._
