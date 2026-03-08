"""
run_pipeline.py – Master script
================================
Runs Parts 1 → 2 → 3 in sequence, leaves system ready for Part 4 (API).

Usage:
  python run_pipeline.py
  # Then start the API:
  python api.py
  # or (if fastapi+uvicorn installed):
  uvicorn api:app --host 0.0.0.0 --port 8000
"""

import sys, json, time, pickle
import numpy as np
from pathlib import Path

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))

EMB_DIR = BASE / "embeddings"
EMB_DIR.mkdir(exist_ok=True)
(BASE / "analysis").mkdir(exist_ok=True)
(BASE / "cache").mkdir(exist_ok=True)


def banner(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


# ── PART 1 ─────────────────────────────────────────────────────────────────────
banner("PART 1 – Embedding & Vector Store")
t0 = time.time()

from part1_embed import load_corpus, build_embeddings, VectorStore, DATA_ROOT

doc_ids, labels, texts = load_corpus(DATA_ROOT)
print(f"  Loaded {len(texts)} documents")

tfidf, svd, embeddings = build_embeddings(texts)

vs = VectorStore()
vs.build(doc_ids, labels, embeddings, texts)
vs.persist(EMB_DIR)

with open(EMB_DIR / "tfidf.pkl", "wb") as f:
    pickle.dump(tfidf, f)
with open(EMB_DIR / "svd.pkl", "wb") as f:
    pickle.dump(svd, f)
(EMB_DIR / "labels.json").write_text(json.dumps(labels))
(EMB_DIR / "doc_ids.json").write_text(json.dumps(doc_ids))
(EMB_DIR / "texts.json").write_text(json.dumps([t[:500] for t in texts]))

print(f"\n  Part 1 done in {time.time()-t0:.1f}s  |  {embeddings.shape}")


# ── PART 2 ─────────────────────────────────────────────────────────────────────
banner("PART 2 – Fuzzy Clustering (NMF warm-start FCM)")
t1 = time.time()

from sklearn.decomposition import TruncatedSVD
from part2_cluster import (
    FuzzyCMeans, nmf_soft_cluster, sweep_clusters, plot_sweep,
    describe_clusters, plot_membership_heatmap, plot_cluster_composition,
    plot_2d_projection, N_CLUSTERS, FCM_M, FCM_MAX_ITER, FCM_TOL, CLUSTER_DIM
)

reducer = TruncatedSVD(n_components=CLUSTER_DIM, random_state=42)
X_lsa   = reducer.fit_transform(embeddings)
with open(EMB_DIR / "cluster_reducer.pkl", "wb") as f:
    pickle.dump(reducer, f)

X_tfidf = tfidf.transform(texts)

print("Running cluster sweep …")
sweep = sweep_clusters(X_tfidf, X_lsa, [8, 10, 12, 15, 18, 20])
plot_sweep(sweep, N_CLUSTERS)

print(f"Final NMF k={N_CLUSTERS} + FCM …")
nmf_model, U_nmf = nmf_soft_cluster(X_tfidf, N_CLUSTERS)
fcm = FuzzyCMeans(n_clusters=N_CLUSTERS, m=FCM_M, max_iter=FCM_MAX_ITER, tol=FCM_TOL)
fcm.fit(X_lsa, U_init=U_nmf)

with open(EMB_DIR / "fcm.pkl", "wb") as f:
    pickle.dump(fcm, f)
with open(EMB_DIR / "nmf.pkl", "wb") as f:
    pickle.dump(nmf_model, f)
np.save(EMB_DIR / "membership_matrix.npy", fcm.U)

texts_snippets = json.loads((EMB_DIR / "texts.json").read_text())
cluster_info   = describe_clusters(fcm, labels, texts_snippets, tfidf, nmf_model)
(EMB_DIR / "cluster_info.json").write_text(json.dumps(cluster_info, indent=2, default=str))

plot_membership_heatmap(fcm, labels)
plot_cluster_composition(cluster_info)
plot_2d_projection(X_lsa, fcm, labels)

print(f"\n  Part 2 done in {time.time()-t1:.1f}s  |  "
      f"PC={fcm.partition_coefficient:.4f}  PE={fcm.partition_entropy:.4f}")

print("\n  Cluster Summary:")
for c, info in cluster_info.items():
    cats  = ", ".join(f"{cat}({cnt})" for cat, cnt in info["top_cats"])
    terms = ", ".join(info["top_terms"][:5])
    print(f"  C{c:2d}  n={info['size']:4d}  [{cats}]")
    print(f"         terms: {terms}")


# ── PART 3 ─────────────────────────────────────────────────────────────────────
banner("PART 3 – Semantic Cache Threshold Analysis")
t2 = time.time()

from part3_cache import threshold_analysis, DEFAULT_THRESHOLD
threshold_analysis()
print(f"\n  Chosen threshold τ={DEFAULT_THRESHOLD}")
print(f"  Part 3 done in {time.time()-t2:.1f}s")


# ── SUMMARY ────────────────────────────────────────────────────────────────────
banner("PIPELINE COMPLETE")
print(f"""
  Documents : {len(texts):,}  |  Dims: {embeddings.shape[1]}  |  Clusters: {N_CLUSTERS}
  PC={fcm.partition_coefficient:.4f} (1/C={1/N_CLUSTERS:.4f})  |  Cache τ={DEFAULT_THRESHOLD}
  Total time: {time.time()-t0:.1f}s

  Start the API:
    python api.py                          (stdlib, zero deps)
    uvicorn api:app --host 0.0.0.0 --port 8000  (FastAPI)

  Test:
    curl -X POST http://localhost:8000/query \\
         -H 'Content-Type: application/json' \\
         -d '{{"query": "clipper chip encryption controversy"}}'
""")
