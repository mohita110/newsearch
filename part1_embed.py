"""
Part 1 – Embedding & Vector Database Setup
==========================================
Design decisions (justified here as instructed):

EMBEDDING MODEL: TF-IDF + Truncated SVD (LSA) at 300 dims.
  Rationale: sentence-transformers/BERT would be ideal but require network access
  to download model weights. TF-IDF + LSA is the canonical "lightweight semantic"
  approach: it captures term co-occurrence structure, is fast, fully offline,
  and has a 30-year literature validating it for newsgroup corpora specifically
  (the 20 Newsgroups dataset is actually *the* standard LSA benchmark).
  300 dims: enough to cover the major semantic axes without noise dominance.

VECTOR STORE: Custom JSON-backed flat store with numpy mmap for embeddings.
  Rationale: ChromaDB/Pinecone require network install. A flat numpy matrix
  with an index file is functionally equivalent for <20k docs: brute-force
  cosine search over 20k×300 floats takes ~2ms on CPU, well within SLA.
  For scale-out, the design exposes a clean interface that could swap in FAISS.

CLEANING STRATEGY:
  - Strip email headers (everything before the blank line after headers): these
    are metadata, not content, and would dominate TF-IDF if left in.
  - Remove quoted reply lines ("> ...") – they duplicate prior posts and add noise.
  - Remove email addresses, URLs, and lines that are >50% non-alphabetic
    (PGP blocks, uuencoded binary, etc.) – these are structural noise, not semantics.
  - Keep the Subject line: it is the densest semantic signal in a newsgroup post.
  - Minimum token threshold: drop docs with <20 tokens after cleaning – they
    carry no useful signal and would distort cluster centroids.
  - We do NOT lemmatise or stem: LSA already handles morphological variation via
    co-occurrence; stemming can hurt precision (e.g. "gun" / "guns" → same token
    is fine, but "universal" → "univers" loses meaning).
"""

import os
import re
import json
import pickle
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from typing import List, Dict, Tuple

# ── Paths ─────────────────────────────────────────────────────────────────────
# DATA_ROOT points to the extracted 20_newsgroups folder.
# The script searches for it automatically. If it still can't find it,
# uncomment and set the path manually on the next line:
# DATA_ROOT = Path(r"C:\Users\009tu\OneDrive\Desktop\20_newsgroups")

_HERE = Path(__file__).parent.resolve()

_CANDIDATES = [
    _HERE / "newsgroups_raw" / "20_newsgroups",
    _HERE / "20_newsgroups",
    _HERE.parent / "20_newsgroups",
    _HERE / "data" / "20_newsgroups",
]

DATA_ROOT = None
for _c in _CANDIDATES:
    if _c.exists() and _c.is_dir():
        DATA_ROOT = _c
        break

if DATA_ROOT is None:
    raise FileNotFoundError(
        "\n\n"
        "  Could not find the 20_newsgroups dataset folder.\n\n"
        "  OPTION A - Place it next to this script:\n"
        f"    {_HERE}\\newsgroups_raw\\20_newsgroups\\  (20 sub-folders inside)\n\n"
        "  OPTION B - Set the path manually in part1_embed.py line 47:\n"
        "    DATA_ROOT = Path(r\'C:\\full\\path\\to\\20_newsgroups\')\n\n"
        "  HOW TO EXTRACT:\n"
        "    1. Unzip twenty_newsgroups.zip\n"
        "    2. Extract 20_newsgroups.tar.gz inside it (use 7-Zip or WinRAR)\n"
        "    3. Move the 20_newsgroups folder next to part1_embed.py\n"
    )

print(f"  Dataset found at: {DATA_ROOT}")

OUT_DIR = _HERE / "embeddings"
OUT_DIR.mkdir(exist_ok=True)


EMBED_DIM   = 300   # LSA dimensionality – 300 is the sweet spot for 20NG
MIN_TOKENS  = 20    # drop posts shorter than this after cleaning
MAX_TOKENS  = 5000  # truncate runaway posts (FAQs, digests) – rare but they exist


# ── Cleaning ──────────────────────────────────────────────────────────────────

# Header fields we explicitly keep (subject carries semantic signal)
KEEP_HEADERS = {"subject"}

def parse_newsgroup_post(raw: str) -> str:
    """
    Parse a raw newsgroup post.

    Strategy:
      1. Split on the first blank line to separate headers from body.
      2. From headers, extract only the Subject (it's semantically rich).
      3. From body, strip quoted lines, email/URL artifacts, and binary junk.
    """
    lines = raw.split("\n")

    # Find the blank line separating headers from body
    split_idx = 0
    for i, line in enumerate(lines):
        if line.strip() == "":
            split_idx = i
            break

    header_lines = lines[:split_idx]
    body_lines   = lines[split_idx + 1:]

    # Extract subject from headers
    subject = ""
    for line in header_lines:
        lower = line.lower()
        if lower.startswith("subject:"):
            # strip "Re:", "Fwd:", etc. – they are thread noise, not content
            subj = line[8:].strip()
            subj = re.sub(r"(?i)^(re|fwd|aw|sv)[\s:]+", "", subj).strip()
            subject = subj
            break

    # Clean body lines
    clean_body = []
    for line in body_lines:
        stripped = line.strip()

        # Drop quoted reply lines ("> ..." patterns)
        if stripped.startswith(">"):
            continue

        # Drop attribution lines like "In article <...> foo@bar wrote:"
        if re.match(r"^in article", stripped, re.IGNORECASE):
            continue
        if re.match(r"^.{0,60}wrote\s*:", stripped, re.IGNORECASE):
            continue

        # Drop email addresses and URLs (they add lexical noise, not semantics)
        line_clean = re.sub(r"\S+@\S+", "", stripped)
        line_clean = re.sub(r"https?://\S+", "", line_clean)
        line_clean = re.sub(r"www\.\S+", "", line_clean)

        # Drop lines that are >60% non-alphabetic chars
        # (PGP signatures, uuencoded data, ASCII art, separator lines)
        alpha_count = sum(1 for c in line_clean if c.isalpha())
        if len(line_clean) > 10 and alpha_count / max(len(line_clean), 1) < 0.4:
            continue

        # Drop very short residual lines (single punctuation, etc.)
        if len(line_clean.split()) < 2:
            continue

        clean_body.append(line_clean)

    # Combine subject + body
    text = subject + " " + " ".join(clean_body)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Truncate runaway documents (FAQs / digests contain thousands of tokens;
    # they would dominate TF-IDF and aren't representative individual posts)
    words = text.split()
    if len(words) > MAX_TOKENS:
        text = " ".join(words[:MAX_TOKENS])

    return text


# ── Corpus loading ─────────────────────────────────────────────────────────────

def load_corpus(data_root: Path) -> Tuple[List[str], List[str], List[str]]:
    """
    Walk the 20 Newsgroups directory tree and return parallel lists of
    (doc_ids, labels, cleaned_texts).
    """
    doc_ids, labels, texts = [], [], []

    categories = sorted([d for d in data_root.iterdir() if d.is_dir()])

    for cat_dir in categories:
        category = cat_dir.name
        for fpath in sorted(cat_dir.iterdir()):
            if not fpath.is_file():
                continue
            try:
                raw = fpath.read_text(encoding="latin-1")  # 20NG uses latin-1
            except Exception:
                continue

            cleaned = parse_newsgroup_post(raw)

            # Token count gate – drop empty / near-empty posts
            if len(cleaned.split()) < MIN_TOKENS:
                continue

            doc_ids.append(f"{category}/{fpath.name}")
            labels.append(category)
            texts.append(cleaned)

    return doc_ids, labels, texts


# ── Embedding pipeline ─────────────────────────────────────────────────────────

def build_embeddings(texts: List[str], n_components: int = EMBED_DIM):
    """
    TF-IDF → Truncated SVD (LSA) → L2-normalised unit vectors.

    TF-IDF params:
      - sublinear_tf=True: log(1+tf) dampens very frequent terms like "the".
      - min_df=3: ignore terms appearing in <3 docs (typos, rare proper nouns).
      - max_df=0.85: ignore terms in >85% of docs (nearly-stopwords not caught
        by the standard list, e.g. "writes", "article").
      - ngram_range=(1,2): bigrams capture "gun control" vs "gun" vs "control".
        We stop at 2 to keep the vocabulary manageable.
      - max_features=100_000: caps vocabulary; 20NG has ~150k unigrams+bigrams,
        the top 100k covers >99% of token mass.

    LSA (TruncatedSVD):
      - 300 components: standard for document similarity tasks.
        Paper benchmark: Deerwester et al. used 100-300; 300 preserves
        more fine-grained topic distinctions needed for 20 categories.
      - We normalise to unit length so cosine similarity = dot product,
        enabling O(n) brute-force search and compatibility with clustering.
    """
    print(f"  Building TF-IDF matrix over {len(texts)} documents …")
    tfidf = TfidfVectorizer(
        sublinear_tf=True,
        min_df=3,
        max_df=0.85,
        ngram_range=(1, 2),
        max_features=100_000,
        strip_accents="unicode",
        analyzer="word",
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9\-]{1,}\b",  # no single chars
    )
    X_tfidf = tfidf.fit_transform(texts)
    print(f"  TF-IDF shape: {X_tfidf.shape}")

    print(f"  Reducing to {n_components} dims via LSA …")
    svd = TruncatedSVD(n_components=n_components, n_iter=7, random_state=42)
    X_lsa = svd.fit_transform(X_tfidf)
    explained = svd.explained_variance_ratio_.sum()
    print(f"  Explained variance: {explained:.3f} ({explained*100:.1f}%)")

    # L2 normalise → unit hypersphere → cosine sim = dot product
    X_norm = normalize(X_lsa, norm="l2")
    print(f"  Embedding matrix: {X_norm.shape}, dtype {X_norm.dtype}")

    return tfidf, svd, X_norm


# ── Custom Vector Store ────────────────────────────────────────────────────────

class VectorStore:
    """
    Lightweight vector store backed by numpy (embeddings) + JSON (metadata).

    Supports:
      - add(doc_id, label, embedding, text_snippet)
      - search(query_vec, top_k, filter_label=None) → ranked results
      - persist() / load()

    Design: for 20k docs × 300 dims, the full matrix fits in ~23 MB RAM.
    Brute-force cosine search (matrix multiply) takes ~2ms, no index needed.
    If the corpus grew to millions, we'd swap in FAISS here.
    """

    def __init__(self):
        self.doc_ids:  List[str] = []
        self.labels:   List[str] = []
        self.snippets: List[str] = []
        self._matrix: np.ndarray | None = None  # (N, D) float32

    def build(self, doc_ids, labels, embeddings, texts):
        self.doc_ids  = list(doc_ids)
        self.labels   = list(labels)
        self.snippets = [t[:300] for t in texts]   # store first 300 chars as snippet
        self._matrix  = embeddings.astype(np.float32)

    def search(
        self,
        query_vec: np.ndarray,
        top_k: int = 10,
        filter_label: str | None = None
    ) -> List[Dict]:
        """Cosine similarity search. query_vec must be L2-normalised."""
        q = query_vec.astype(np.float32).flatten()

        if filter_label is not None:
            mask = np.array([l == filter_label for l in self.labels])
            mat  = self._matrix[mask]
            orig_indices = np.where(mask)[0]
        else:
            mat  = self._matrix
            orig_indices = np.arange(len(self.doc_ids))

        if len(mat) == 0:
            return []

        scores = mat @ q                         # cosine sim (unit vecs)
        top_local = np.argpartition(scores, -min(top_k, len(scores)))[-top_k:]
        top_local  = top_local[np.argsort(scores[top_local])[::-1]]

        results = []
        for li in top_local:
            gi = orig_indices[li]
            results.append({
                "doc_id":  self.doc_ids[gi],
                "label":   self.labels[gi],
                "score":   float(scores[li]),
                "snippet": self.snippets[gi],
            })
        return results

    def persist(self, out_dir: Path):
        np.save(out_dir / "embeddings.npy", self._matrix)
        meta = {
            "doc_ids":  self.doc_ids,
            "labels":   self.labels,
            "snippets": self.snippets,
        }
        (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
        print(f"  Saved {len(self.doc_ids)} docs to {out_dir}")

    @classmethod
    def load(cls, out_dir: Path) -> "VectorStore":
        vs = cls()
        vs._matrix  = np.load(out_dir / "embeddings.npy")
        meta = json.loads((out_dir / "metadata.json").read_text())
        vs.doc_ids  = meta["doc_ids"]
        vs.labels   = meta["labels"]
        vs.snippets = meta["snippets"]
        print(f"  Loaded {len(vs.doc_ids)} docs from {out_dir}")
        return vs

    def __len__(self):
        return len(self.doc_ids)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("PART 1 – Embedding & Vector Store")
    print("=" * 60)

    print("\n[1/4] Loading and cleaning corpus …")
    doc_ids, labels, texts = load_corpus(DATA_ROOT)
    print(f"  Loaded {len(texts)} documents across {len(set(labels))} categories")
    n_dropped = 19997 - len(texts)
    print(f"  Dropped {n_dropped} documents (too short / unreadable after cleaning)")

    # Category distribution
    from collections import Counter
    cat_counts = Counter(labels)
    print("\n  Category counts:")
    for cat, cnt in sorted(cat_counts.items()):
        print(f"    {cat:<35} {cnt:>5}")

    print("\n[2/4] Building embeddings …")
    tfidf, svd, embeddings = build_embeddings(texts)

    print("\n[3/4] Building vector store …")
    vs = VectorStore()
    vs.build(doc_ids, labels, embeddings, texts)

    print("\n[4/4] Persisting …")
    vs.persist(OUT_DIR)

    # Persist the sklearn pipeline for query-time embedding
    with open(OUT_DIR / "tfidf.pkl", "wb") as f:
        pickle.dump(tfidf, f)
    with open(OUT_DIR / "svd.pkl", "wb") as f:
        pickle.dump(svd, f)

    # Save label list and doc_ids for downstream use
    (OUT_DIR / "labels.json").write_text(json.dumps(labels))
    (OUT_DIR / "doc_ids.json").write_text(json.dumps(doc_ids))

    print("\n✓ Part 1 complete.")
    print(f"  Embeddings: {embeddings.shape}")
    print(f"  Files saved to: {OUT_DIR}")

    return tfidf, svd, embeddings, doc_ids, labels, texts, vs


if __name__ == "__main__":
    main()