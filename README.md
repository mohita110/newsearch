# Newsgroups Semantic Search System

A full semantic search pipeline over the 20 Newsgroups corpus, featuring:
- **TF-IDF + LSA embeddings** (300-dim) stored in a custom lightweight vector store
- **Fuzzy C-Means clustering** (k=12) with per-document soft membership distributions
- **Cluster-partitioned semantic cache** with configurable cosine similarity threshold
- **FastAPI service** (with stdlib fallback) exposing `/query`, `/cache/stats`, `/cache`

---

## Project Structure

```
newsearch/
├── part1_embed.py         # Corpus ingestion, TF-IDF+LSA embeddings, vector store
├── part2_cluster.py       # Fuzzy C-Means, cluster sweep, visualisations
├── part3_cache.py         # Semantic cache (from scratch), threshold analysis
├── api.py                 # FastAPI service (+ stdlib http.server fallback)
├── run_pipeline.py        # Master script: runs Parts 1→2→3 in sequence
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── embeddings/            # Generated: models, embeddings, cluster artefacts
├── analysis/              # Generated: plots (PNG)
└── cache/                 # Generated: persisted cache state
```

---

## Quickstart

### 1. Set up the environment

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Place the dataset

Make sure `twenty_newsgroups.zip` is extracted so that:
```
newsearch/newsgroups_raw/20_newsgroups/<category>/<file>
```

Or point `DATA_ROOT` in `part1_embed.py` to your extraction path.

### 3. Run the full pipeline

```bash
python run_pipeline.py
```

This runs Parts 1–3 (~10–20 min depending on CPU):
- Builds and persists embeddings
- Trains Fuzzy C-Means
- Generates 5 analysis plots in `analysis/`
- Runs threshold sensitivity analysis

### 4. Start the API

```bash
# If FastAPI is installed:
uvicorn api:app --host 0.0.0.0 --port 8000

# Fallback (always works):
python api.py
```

---

## API Reference

### POST /query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the best treatment for migraines?"}'
```

**Response:**
```json
{
  "query": "What is the best treatment for migraines?",
  "cache_hit": false,
  "matched_query": null,
  "similarity_score": 0.731,
  "result": [
    {
      "doc_id": "sci.med/59416",
      "label": "sci.med",
      "score": 0.912,
      "snippet": "..."
    }
  ],
  "dominant_cluster": 7,
  "latency_ms": 12.4
}
```

On the next similar query:
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How do people deal with migraine headaches?"}'
```
```json
{
  "cache_hit": true,
  "matched_query": "What is the best treatment for migraines?",
  "similarity_score": 0.912,
  ...
}
```

### GET /cache/stats

```bash
curl http://localhost:8000/cache/stats
```
```json
{
  "total_entries": 42,
  "hit_count": 17,
  "miss_count": 25,
  "hit_rate": 0.405,
  "threshold": 0.88,
  "partition_sizes": {"3": 12, "7": 8, "11": 5, ...}
}
```

### DELETE /cache

```bash
curl -X DELETE http://localhost:8000/cache
```
```json
{"message": "Cache flushed.", "entries_deleted": 0}
```

---

## Docker

```bash
# Build (includes dataset extraction + pipeline)
docker build -t newsearch .

# Run
docker run -p 8000:8000 newsearch

# Or with docker-compose (mounts pre-built artefacts):
docker-compose up
```

---

## Design Decisions

### Embeddings (Part 1)
- **TF-IDF + LSA (300 dims)** instead of sentence-transformers:
  fully offline, ~23 MB RAM for the full corpus matrix, and LSA is the
  canonical baseline for the 20NG dataset.
- **Cleaning**: strip email headers (keep Subject), remove quoted lines,
  drop PGP/binary blocks, enforce min 20 tokens.
- **Vector store**: numpy matrix + JSON metadata. Brute-force cosine search
  over 20k docs takes ~2ms — no FAISS/ChromaDB needed at this scale.

### Fuzzy Clustering (Part 2)
- **FCM from scratch** (no skfuzzy): the membership update equations are
  straightforward; implementing from scratch ensures full control and removes
  the external dependency.
- **k=12 chosen empirically**: the Partition Coefficient elbow occurs at k≈12.
  The 20 labelled categories collapse to ~12 genuine semantic groups
  (comp.* merge, rec.sport.* merge, talk.politics.* merge, etc.).
- **Fuzziness m=2.0**: standard Bezdek (1981) value. Lower m→hard clustering,
  higher m→uniform memberships.
- **Pre-reduction to 50 dims before FCM**: Euclidean distance in 300 dims
  suffers from concentration of measure; 50 dims retains 85%+ of variance
  while making distances geometrically meaningful.

### Semantic Cache (Part 3)
- **Cluster-partitioned lookup**: queries are routed to the top-2 cluster
  partitions based on FCM membership, reducing scan size by ~6× at steady state.
- **Threshold τ=0.88**: chosen after sweeping τ ∈ [0.70, 0.99] on synthetic
  paraphrase/unrelated pairs. At τ=0.88:
  - Paraphrases ("gun control" / "firearms legislation") correctly hit
  - Same-domain-but-different ("Windows" / "Linux") correctly miss
  - F1 of hit detection is maximised at this threshold.
- **No Redis/Memcached**: pure Python dict/list structure, serialised to JSON.

---

## Output Plots

| File | Contents |
|------|----------|
| `analysis/cluster_sweep.png` | PC and PE vs k – justifies k=12 choice |
| `analysis/membership_heatmap.png` | 300-doc sample membership matrix |
| `analysis/cluster_composition.png` | True label distribution per cluster |
| `analysis/cluster_2d.png` | PCA 2D projection coloured by cluster |
| `analysis/threshold_analysis.png` | Hit rate, false-hit rate, F1 vs τ |
