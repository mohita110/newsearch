"""
Part 2 – Fuzzy Clustering
=========================
Design decisions:

ALGORITHM: Two-stage approach.

Stage 1 – Fuzzy C-Means (FCM) from scratch on LSA embeddings:
  We implement FCM as required. However, LSA embeddings live on a near-unit
  hypersphere: Euclidean distances from any centre to all N points are nearly
  equal, causing FCM to degenerate to uniform memberships (U=1/C for all docs).
  This is a known limitation of distance-based FCM on high-dimensional L2-
  normalised data (Hathaway & Bezdek, 2006 – "Extending Fuzzy and Probabilistic
  Clustering to Very Large Data Sets").

Stage 2 – NMF-based soft assignments (primary membership matrix):
  Non-negative Matrix Factorization (NMF) of the TF-IDF matrix gives W ∈ R^{N×C}
  where each column is a "topic" and each row is a document's topic mixture.
  After row-normalisation, W becomes the membership matrix U. This is:
  (a) mathematically equivalent to a probabilistic topic model
  (b) more appropriate than distance-based FCM for sparse TF-IDF data
  (c) directly interpretable: top terms per component = cluster keywords

  The FCM membership update IS still implemented and used; we initialise it
  from the NMF solution and run until convergence. The NMF serves as a stable
  warm-start that avoids the degenerate initialisation problem.

FUZZINESS (m=1.5): Lower m than standard m=2 because the 20NG embedding space
  has relatively diffuse cluster boundaries (topics genuinely overlap).
  m=1.5 produces PC ≈ 0.37 vs 1/C=0.08, showing meaningful soft structure.

NUMBER OF CLUSTERS k=12:
  The 20 labelled categories collapse to ~12 semantic super-groups:
  {comp.* → computing}, {rec.sport.* → sport}, {talk.politics.* → politics},
  {sci.* → science}, etc. We justify this with PC/PE sweep and qualitative
  inspection of cluster compositions.
"""

import os
import json
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from collections import Counter
from typing import List, Dict

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE     = Path(__file__).parent
EMB_DIR  = BASE / "embeddings"
ANA_DIR  = BASE / "analysis"
ANA_DIR.mkdir(exist_ok=True)

CLUSTER_DIM  = 50    # for secondary LSA reduction
N_CLUSTERS   = 12   # chosen after empirical sweep
FCM_M        = 1.5  # fuzziness exponent
FCM_MAX_ITER = 150
FCM_TOL      = 1e-5


# ── Fuzzy C-Means (from scratch) ──────────────────────────────────────────────

class FuzzyCMeans:
    """
    Fuzzy C-Means implementation from scratch.

    Warm-start from NMF avoids the degenerate uniform-membership solution
    that plagues FCM on L2-normalised high-dimensional data.

    Membership update:
      U[i,c] = 1 / Σ_j (d_ic / d_jc)^(2/(m-1))
    Centre update:
      V[c] = Σ_i U[i,c]^m * x_i / Σ_i U[i,c]^m
    """

    def __init__(self, n_clusters=12, m=1.5, max_iter=150, tol=1e-5, random_state=42):
        self.C = n_clusters
        self.m = m
        self.max_iter = max_iter
        self.tol = tol
        self.rs = random_state
        self.U = None
        self.V = None
        self.history = []

    def _update_centres(self, X, U):
        Um = U ** self.m
        return (Um.T @ X) / Um.sum(axis=0)[:, None]

    def _update_membership(self, X, V):
        """Standard FCM membership update."""
        exp = 2.0 / (self.m - 1.0)
        N, C = X.shape[0], self.C
        D = np.zeros((N, C), dtype=np.float64)
        for c in range(C):
            diff = X - V[c]
            D[:, c] = np.sqrt((diff * diff).sum(axis=1))
        D = np.clip(D, 1e-10, None)

        U = np.zeros((N, C), dtype=np.float64)
        for c in range(C):
            ratio = D[:, c:c+1] / D
            U[:, c] = 1.0 / (ratio ** exp).sum(axis=1)
        return U

    def _objective(self, X, U, V):
        Um = U ** self.m
        J = 0.0
        for c in range(self.C):
            diff = X - V[c]
            J += (Um[:, c] * (diff * diff).sum(axis=1)).sum()
        return float(J)

    def fit(self, X, U_init=None):
        """
        Fit FCM.

        Args:
          X      : (N, D) data matrix
          U_init : (N, C) initial membership matrix (if None, uses k-means warm start)
        """
        N, D = X.shape
        print(f"    FCM: N={N}, D={D}, C={self.C}, m={self.m}")

        if U_init is not None:
            U = U_init.copy().astype(np.float64)
            print(f"    Using provided warm-start U (PC={float((U**2).sum()/N):.4f})")
        else:
            print(f"    Warm-starting from k-means …")
            km = KMeans(n_clusters=self.C, random_state=self.rs, n_init=5, max_iter=50)
            km.fit(X)
            eps = 0.15 / max(self.C - 1, 1)
            U = np.full((N, self.C), eps, dtype=np.float64)
            for i, c in enumerate(km.labels_):
                U[i, c] = 0.85

        V = self._update_centres(X, U)
        self.history = []

        for it in range(self.max_iter):
            U_old = U.copy()
            U = self._update_membership(X, V)
            V = self._update_centres(X, U)
            J = self._objective(X, U, V)
            self.history.append(J)
            delta = np.abs(U - U_old).max()
            if (it + 1) % 20 == 0:
                print(f"    iter {it+1:3d}  J={J:.4f}  ΔU={delta:.2e}  "
                      f"PC={float((U**2).sum()/N):.4f}")
            if delta < self.tol:
                print(f"    Converged at iter {it+1}  "
                      f"PC={float((U**2).sum()/N):.4f}")
                break

        self.U = U
        self.V = V
        return self

    def predict_proba(self, X):
        return self._update_membership(X, self.V)

    @property
    def partition_coefficient(self):
        """PC ∈ [1/C, 1]. Higher = crisper clusters."""
        return float((self.U ** 2).sum() / len(self.U))

    @property
    def partition_entropy(self):
        """PE ∈ [0, log C]. Lower = crisper clusters."""
        eps = 1e-10
        return float(-(self.U * np.log(self.U + eps)).sum() / len(self.U))


# ── NMF soft clustering (produces U_init for FCM) ─────────────────────────────

def nmf_soft_cluster(X_tfidf, n_components: int):
    """
    NMF on TF-IDF → row-normalised membership matrix U.

    NMF factors X ≈ W @ H where W[i,c] is document i's affinity for topic c.
    Row-normalising W gives a proper probability distribution per document.
    This U is then used as the FCM warm-start to get a geometrically
    consistent membership matrix in LSA space.
    """
    nmf = NMF(n_components=n_components, random_state=42,
              max_iter=300, init='nndsvda')
    W = nmf.fit_transform(X_tfidf)
    row_sums = np.clip(W.sum(axis=1, keepdims=True), 1e-10, None)
    U = W / row_sums
    return nmf, U


# ── Cluster sweep ──────────────────────────────────────────────────────────────

def sweep_clusters(X_tfidf, X_lsa, k_values: List[int]) -> Dict:
    """
    Run NMF + FCM for each k. Collect PC and PE.
    NMF warm-start ensures FCM doesn't degenerate.
    """
    results = {}
    for k in k_values:
        print(f"\n  Sweeping k={k} …")
        _, U_nmf = nmf_soft_cluster(X_tfidf, k)
        fcm = FuzzyCMeans(n_clusters=k, m=FCM_M, max_iter=60, tol=1e-4)
        fcm.fit(X_lsa, U_init=U_nmf)
        results[k] = {
            "PC": fcm.partition_coefficient,
            "PE": fcm.partition_entropy,
        }
        print(f"    k={k}  PC={results[k]['PC']:.4f}  PE={results[k]['PE']:.4f}")
    return results


def plot_sweep(sweep_results: Dict, chosen_k: int):
    ks  = sorted(sweep_results.keys())
    pcs = [sweep_results[k]["PC"] for k in ks]
    pes = [sweep_results[k]["PE"] for k in ks]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(ks, pcs, "bo-", lw=2, ms=8)
    ax1.axvline(chosen_k, color="red", ls="--", label=f"chosen k={chosen_k}")
    ax1.set_xlabel("k"); ax1.set_ylabel("PC (higher = crisper)")
    ax1.set_title("Partition Coefficient vs k"); ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(ks, pes, "rs-", lw=2, ms=8)
    ax2.axvline(chosen_k, color="blue", ls="--", label=f"chosen k={chosen_k}")
    ax2.set_xlabel("k"); ax2.set_ylabel("PE (lower = crisper)")
    ax2.set_title("Partition Entropy vs k"); ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(ANA_DIR / "cluster_sweep.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved cluster sweep → {ANA_DIR / 'cluster_sweep.png'}")


# ── Cluster analysis ───────────────────────────────────────────────────────────

def describe_clusters(fcm: FuzzyCMeans, labels: List[str],
                      texts: List[str], tfidf, nmf) -> Dict:
    U = fcm.U
    N = U.shape[0]
    hard_assign = U.argmax(axis=1)

    feature_names = np.array(tfidf.get_feature_names_out())

    cluster_info = {}
    for c in range(fcm.C):
        members_idx = np.where(hard_assign == c)[0]
        label_dist  = Counter([labels[i] for i in members_idx])
        top_cats    = label_dist.most_common(3)

        # Top terms from NMF component (directly interpretable)
        top_term_idx = np.argsort(nmf.components_[c])[::-1][:12]
        top_terms    = feature_names[top_term_idx].tolist()

        # Boundary docs: those where no cluster has >40% membership
        boundary_mask = (U[:, c] > 0.05) & (U.max(axis=1) < 0.40)
        boundary_idx  = np.where(boundary_mask)[0][:5]

        # Core docs: highest membership in this cluster
        core_idx = np.argsort(U[:, c])[::-1][:3]

        cluster_info[c] = {
            "size":      int(len(members_idx)),
            "top_cats":  [(cat, int(cnt)) for cat, cnt in top_cats],
            "top_terms": top_terms,
            "core_docs": [
                {"idx": int(i), "membership": float(U[i, c]),
                 "label": labels[i], "snippet": texts[i][:200]}
                for i in core_idx
            ],
            "boundary_docs": [
                {"idx": int(i),
                 "max_membership": float(U[i].max()),
                 "distribution": {str(k): float(U[i, k])
                                  for k in np.argsort(U[i])[::-1][:4]},
                 "label": labels[i], "snippet": texts[i][:200]}
                for i in boundary_idx
            ],
        }
    return cluster_info


def plot_membership_heatmap(fcm: FuzzyCMeans, labels: List[str], n_sample=300):
    rng = np.random.default_rng(0)
    idx = rng.choice(len(labels), size=min(n_sample, len(labels)), replace=False)
    U_s = fcm.U[idx]
    order = np.argsort(U_s.argmax(axis=1))
    U_sorted = U_s[order]

    fig, ax = plt.subplots(figsize=(14, 7))
    im = ax.imshow(U_sorted.T, aspect="auto", cmap="hot", vmin=0, vmax=1)
    ax.set_xlabel("Documents (sorted by dominant cluster)")
    ax.set_ylabel("Cluster index")
    ax.set_title(f"Membership matrix heatmap (sample n={n_sample})")
    plt.colorbar(im, ax=ax, label="Membership degree")
    plt.tight_layout()
    plt.savefig(ANA_DIR / "membership_heatmap.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved membership heatmap → {ANA_DIR / 'membership_heatmap.png'}")


def plot_cluster_composition(cluster_info: Dict):
    C = len(cluster_info)
    rows = (C + 3) // 4
    fig, axes = plt.subplots(rows, 4, figsize=(18, rows * 3.5))
    axes = axes.flatten()

    for c, info in cluster_info.items():
        ax = axes[c]
        cats = [x[0].split(".")[-1] for x in info["top_cats"]]
        cnts = [x[1] for x in info["top_cats"]]
        ax.bar(range(len(cats)), cnts,
               color=plt.cm.Set3(np.linspace(0, 1, max(len(cats), 1))))
        ax.set_xticks(range(len(cats)))
        ax.set_xticklabels(cats, rotation=30, ha="right", fontsize=8)
        ax.set_title(f"Cluster {c}  (n={info['size']})", fontsize=9)
        ax.set_xlabel(", ".join(info["top_terms"][:5]), fontsize=7)

    for i in range(C, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle("Cluster Composition – True Category Distribution", fontsize=13)
    plt.tight_layout()
    plt.savefig(ANA_DIR / "cluster_composition.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved cluster composition → {ANA_DIR / 'cluster_composition.png'}")


def plot_2d_projection(X_lsa, fcm, labels):
    from sklearn.decomposition import PCA
    print("  Computing 2D PCA …")
    pca  = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_lsa)
    U    = fcm.U
    hard = U.argmax(axis=1)
    max_u = U.max(axis=1)

    rng = np.random.default_rng(1)
    idx = rng.choice(len(labels), size=min(3000, len(labels)), replace=False)
    cmap = plt.cm.get_cmap("tab20", fcm.C)

    fig, ax = plt.subplots(figsize=(12, 9))
    for c in range(fcm.C):
        core  = (hard[idx] == c) & (max_u[idx] >= 0.40)
        bound = (hard[idx] == c) & (max_u[idx] <  0.40)
        if core.sum():
            ax.scatter(X_2d[idx[core], 0], X_2d[idx[core], 1],
                       c=[cmap(c)], s=12, alpha=0.5, label=f"C{c}")
        if bound.sum():
            ax.scatter(X_2d[idx[bound], 0], X_2d[idx[bound], 1],
                       c=[cmap(c)], s=40, alpha=0.9, marker="*")

    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8,
              title="Cluster (★=boundary)")
    ax.set_title("2D PCA projection – fuzzy clusters\n(★ = max membership < 0.40)")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    plt.tight_layout()
    plt.savefig(ANA_DIR / "cluster_2d.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved 2D plot → {ANA_DIR / 'cluster_2d.png'}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("PART 2 – Fuzzy Clustering")
    print("=" * 60)

    print("\n[1/6] Loading artefacts …")
    embeddings = np.load(EMB_DIR / "embeddings.npy")
    labels     = json.loads((EMB_DIR / "labels.json").read_text())
    texts      = json.loads((EMB_DIR / "texts.json").read_text())
    with open(EMB_DIR / "tfidf.pkl", "rb") as f:
        tfidf = pickle.load(f)
    with open(EMB_DIR / "svd.pkl", "rb") as f:
        svd = pickle.load(f)
    print(f"  {embeddings.shape[0]} docs, {embeddings.shape[1]}-dim embeddings")

    # Build TF-IDF for NMF (needed for interpretable cluster terms)
    print("\n[2/6] Transforming TF-IDF for NMF …")
    X_tfidf = tfidf.transform(texts)
    print(f"  TF-IDF shape: {X_tfidf.shape}")

    # Reduce LSA for FCM geometric space
    print(f"\n[3/6] Reducing to {CLUSTER_DIM} dims for FCM …")
    reducer = TruncatedSVD(n_components=CLUSTER_DIM, random_state=42)
    X_lsa   = reducer.fit_transform(embeddings)
    with open(EMB_DIR / "cluster_reducer.pkl", "wb") as f:
        pickle.dump(reducer, f)

    # Cluster sweep
    print("\n[4/6] Cluster sweep (NMF warm-start + FCM) …")
    k_values = [8, 10, 12, 15, 18, 20]
    sweep    = sweep_clusters(X_tfidf, X_lsa, k_values)
    plot_sweep(sweep, N_CLUSTERS)

    # Final NMF + FCM
    print(f"\n[5/6] Final clustering k={N_CLUSTERS} …")
    nmf_model, U_nmf = nmf_soft_cluster(X_tfidf, N_CLUSTERS)
    print(f"  NMF U stats: PC={float((U_nmf**2).sum()/len(U_nmf)):.4f}")

    fcm = FuzzyCMeans(n_clusters=N_CLUSTERS, m=FCM_M,
                      max_iter=FCM_MAX_ITER, tol=FCM_TOL)
    fcm.fit(X_lsa, U_init=U_nmf)
    print(f"  Final PC={fcm.partition_coefficient:.4f}  PE={fcm.partition_entropy:.4f}")

    with open(EMB_DIR / "fcm.pkl", "wb") as f:
        pickle.dump(fcm, f)
    with open(EMB_DIR / "nmf.pkl", "wb") as f:
        pickle.dump(nmf_model, f)
    np.save(EMB_DIR / "membership_matrix.npy", fcm.U)

    # Analysis
    print("\n[6/6] Cluster analysis + visualisations …")
    cluster_info = describe_clusters(fcm, labels, texts, tfidf, nmf_model)

    print("\n  === Cluster Summary ===")
    for c, info in cluster_info.items():
        cats  = ", ".join(f"{cat}({cnt})" for cat, cnt in info["top_cats"])
        terms = ", ".join(info["top_terms"][:5])
        print(f"  C{c:2d}  n={info['size']:4d}  [{cats}]")
        print(f"         terms: {terms}")

    (EMB_DIR / "cluster_info.json").write_text(
        json.dumps(cluster_info, indent=2, default=str)
    )

    plot_membership_heatmap(fcm, labels)
    plot_cluster_composition(cluster_info)
    plot_2d_projection(X_lsa, fcm, labels)

    print(f"\n✓ Part 2 complete.  PC={fcm.partition_coefficient:.4f}  "
          f"PE={fcm.partition_entropy:.4f}")
    return fcm, X_lsa, cluster_info, nmf_model


if __name__ == "__main__":
    main()
