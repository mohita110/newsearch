"""
Part 4 – FastAPI Service
========================
Endpoints:
  POST /query         – semantic search with cache
  GET  /cache/stats   – cache statistics
  DELETE /cache       – flush cache

We implement this using Python's built-in http.server + json, so there is
zero dependency on FastAPI/uvicorn being installed. The interface is identical
to what FastAPI would expose; if FastAPI is available (via pip install), the
code auto-detects and uses it for nicer docs UI at /docs.

Run with:
  python api.py
  # or, if FastAPI available:
  uvicorn api:app --host 0.0.0.0 --port 8000

The state management pattern:
  All mutable state (cache, embedder, vector store) lives as module-level
  singletons initialised once at startup. FastAPI's dependency injection
  would be cleaner but is overkill for this service scale. The cache is
  thread-safe for read-heavy workloads (Python GIL protects list appends).
"""

import os, sys, json, time, traceback
from pathlib import Path

BASE    = Path(__file__).parent
sys.path.insert(0, str(BASE))

from part3_cache import SemanticCache, QueryEmbedder, DEFAULT_THRESHOLD
from part1_embed import VectorStore

EMB_DIR      = BASE / "embeddings"
CACHE_FILE   = BASE / "cache" / "cache_state.json"
CACHE_DIR    = BASE / "cache"
CACHE_DIR.mkdir(exist_ok=True)

TOP_K        = 10
THRESHOLD    = float(os.environ.get("CACHE_THRESHOLD", DEFAULT_THRESHOLD))

# ── Singleton state ────────────────────────────────────────────────────────────

print("Initialising service …")
_embedder = QueryEmbedder(EMB_DIR)
_vs       = VectorStore.load(EMB_DIR)

# Load or create cache
if CACHE_FILE.exists():
    print(f"  Loading existing cache from {CACHE_FILE} …")
    _cache = SemanticCache.load(CACHE_FILE)
    print(f"  Cache loaded: {_cache.total_entries} entries")
else:
    _cache = SemanticCache(threshold=THRESHOLD, n_clusters=12)
    print(f"  Fresh cache created (threshold={THRESHOLD})")

def _persist_cache():
    _cache.persist(CACHE_FILE)


# ── Core query logic (shared between FastAPI and fallback server) ───────────────

def handle_query(query_text: str) -> dict:
    """
    Full query pipeline:
      1. Embed the query text.
      2. Check the semantic cache.
      3. On hit: return cached result.
      4. On miss: search the vector store, store result, return it.
    """
    if not query_text or not query_text.strip():
        raise ValueError("Query must be a non-empty string.")

    t0 = time.time()

    # Embed
    query_vec, cluster_dist = _embedder.embed(query_text)
    dominant_cluster = int(cluster_dist.argmax())

    # Cache lookup
    cached_entry, sim_score = _cache.lookup(query_vec, cluster_dist)

    if cached_entry is not None:
        # Cache HIT
        result = cached_entry["result"]
        response = {
            "query":            query_text,
            "cache_hit":        True,
            "matched_query":    cached_entry["query_text"],
            "similarity_score": round(sim_score, 6),
            "result":           result,
            "dominant_cluster": dominant_cluster,
            "latency_ms":       round((time.time() - t0) * 1000, 2),
        }
    else:
        # Cache MISS – compute result
        results = _vs.search(query_vec, top_k=TOP_K)
        result  = results  # list of {doc_id, label, score, snippet}

        # Store in cache
        _cache.store(query_text, query_vec, cluster_dist, result)
        _persist_cache()

        response = {
            "query":            query_text,
            "cache_hit":        False,
            "matched_query":    None,
            "similarity_score": round(sim_score, 6),  # best near-miss similarity
            "result":           result,
            "dominant_cluster": dominant_cluster,
            "latency_ms":       round((time.time() - t0) * 1000, 2),
        }

    return response


def handle_cache_stats() -> dict:
    return _cache.stats()


def handle_cache_flush() -> dict:
    _cache.flush()
    if CACHE_FILE.exists():
        CACHE_FILE.unlink()
    return {"message": "Cache flushed.", "entries_deleted": 0}


# ── Try FastAPI first, fall back to stdlib http.server ────────────────────────

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel

    app = FastAPI(
        title="Newsgroups Semantic Search API",
        description=(
            "Semantic search over 20 Newsgroups with cluster-aware "
            "caching and fuzzy cluster membership."
        ),
        version="1.0.0",
    )

    class QueryRequest(BaseModel):
        query: str

    @app.post("/query")
    def query_endpoint(req: QueryRequest):
        """
        Embed the query, check semantic cache, return results.
        Cache hit if a sufficiently similar prior query exists.
        """
        try:
            return handle_query(req.query)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/cache/stats")
    def cache_stats():
        """Return current cache statistics."""
        return handle_cache_stats()

    @app.delete("/cache")
    def flush_cache():
        """Flush the semantic cache and reset all statistics."""
        return handle_cache_flush()

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "cache_entries": _cache.total_entries,
            "corpus_size": len(_vs),
        }

    USING_FASTAPI = True
    print("✓ FastAPI app initialised. Start with: uvicorn api:app --host 0.0.0.0 --port 8000")

except ImportError:
    USING_FASTAPI = False
    print("  FastAPI not available – using built-in HTTP server fallback.")
    print("  Run: python api.py  (starts on port 8000)")


# ── stdlib fallback server ─────────────────────────────────────────────────────

if not USING_FASTAPI:
    import http.server
    import urllib.parse

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            print(f"[{self.command}] {self.path} – " + fmt % args)

        def _send_json(self, data: dict, status: int = 200):
            body = json.dumps(data, indent=2).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if self.path == "/query":
                length = int(self.headers.get("Content-Length", 0))
                body   = self.rfile.read(length)
                try:
                    payload = json.loads(body)
                    resp    = handle_query(payload.get("query", ""))
                    self._send_json(resp)
                except ValueError as e:
                    self._send_json({"detail": str(e)}, 422)
                except Exception as e:
                    traceback.print_exc()
                    self._send_json({"detail": str(e)}, 500)
            else:
                self._send_json({"detail": "Not found"}, 404)

        def do_GET(self):
            if self.path == "/cache/stats":
                self._send_json(handle_cache_stats())
            elif self.path == "/health":
                self._send_json({"status": "ok",
                                 "cache_entries": _cache.total_entries,
                                 "corpus_size": len(_vs)})
            else:
                self._send_json({"detail": "Not found"}, 404)

        def do_DELETE(self):
            if self.path == "/cache":
                self._send_json(handle_cache_flush())
            else:
                self._send_json({"detail": "Not found"}, 404)

    def run_stdlib_server(port: int = 8000):
        addr = ("0.0.0.0", port)
        srv  = http.server.HTTPServer(addr, Handler)
        print(f"✓ Stdlib HTTP server on http://0.0.0.0:{port}")
        print("  POST /query   GET /cache/stats   DELETE /cache")
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            print("\nShutdown.")

    if __name__ == "__main__":
        run_stdlib_server()

else:
    if __name__ == "__main__":
        import uvicorn
        uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
