import os, re, glob
import numpy as np
import pandas as pd
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

TEXTS_DIR = "texts"
LOAD_FROM_DIR = True
K = 3
TOP_N = 15
SAVE_CSVS = True

nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('stopwords', quiet=True)

FALLBACK = [
    "Example astronomy text about galaxies and exoplanets.",
    "Example astronomy text about black holes and event horizons.",
    "Example astronomy text about rovers and Martian regolith.",
    "Example astronomy text about supernovae as standard candles.",
    "Example astronomy text about pulsars and radio interferometry.",
    "Example hoop text about NBA pick-and-roll and spacing.",
    "Example hoop text about rebounds and rim protection.",
    "Example hoop text about playoffs and rotations.",
    "Example hoop text about eFG% and analytics.",
    "Example hoop text about zone defense and fast breaks.",
    "Example cooking text about bread, gluten, and fermentation.",
    "Example cooking text about sauteing onions in olive oil.",
    "Example cooking text about roasting vegetables and caramelization.",
    "Example cooking text about pasta al dente and tomato sauce.",
    "Example cooking text about yogurt/kimchi and probiotics."
]

STOPWORDS = set(stopwords.words("english"))
LEMMA = WordNetLemmatizer()


def guess_topic_from_name(name: str) -> str:
    m = re.match(r"([A-Za-z0-9\- ]+)_", name)
    return m.group(1).lower() if m else "unknown"


def preprocess(text: str) -> str:
    tokens = re.findall(r"[a-z]+", text.lower())
    clean = [LEMMA.lemmatize(t) for t in tokens if t not in STOPWORDS and len(t) > 1]
    return " ".join(clean)


def top_terms(H: np.ndarray, vocab: np.ndarray, topn=10) -> pd.DataFrame:
    rows = []
    for k_idx, comp in enumerate(H):
        order = np.argsort(comp)[::-1][:topn]
        for rank, j in enumerate(order, 1):
            rows.append({
                "topic": k_idx,
                "rank": rank,
                "term": vocab[j],
                "weight": float(comp[j])
            })
    return pd.DataFrame(rows)


if LOAD_FROM_DIR:
    paths = sorted(glob.glob(os.path.join(TEXTS_DIR, "*.txt")))
    if not paths:
        print(f"No .txt files in {TEXTS_DIR}. Using fallback examples.")
        documents = FALLBACK
        filenames = [f"fallback_{i + 1}.txt" for i in range(len(FALLBACK))]
        topics = ["fallback"] * len(FALLBACK)
    else:
        documents, filenames = [], []
        for p in paths:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                documents.append(f.read())
            filenames.append(os.path.basename(p))
        topics = [guess_topic_from_name(n) for n in filenames]
else:
    documents = FALLBACK
    filenames = [f"manual_{i + 1}.txt" for i in range(len(FALLBACK))]
    topics = ["astronomy"] * 5 + ["basketball"] * 5 + ["cooking"] * 5

print(f"Loaded {len(documents)} documents.")

clean_docs = [preprocess(d) for d in documents]
if SAVE_CSVS:
    pd.DataFrame({"filename": filenames, "topic": topics, "original": documents, "cleaned": clean_docs}) \
        .to_csv("cleaned_documents.csv", index=False)

print("\nPreprocessing done (stopwords removed, lemmatized).")

tfidf_vect = TfidfVectorizer(min_df=1, max_df=0.9)
X_tfidf = tfidf_vect.fit_transform(clean_docs)
vocab_tfidf = np.array(tfidf_vect.get_feature_names_out())
print("\nTF-IDF shape:", X_tfidf.shape)

nmf_tfidf = NMF(n_components=K, random_state=42, init="nndsvda", max_iter=500)
W_tfidf = nmf_tfidf.fit_transform(X_tfidf)
H_tfidf = nmf_tfidf.components_

print(f"Reconstruction error (TF-IDF): {nmf_tfidf.reconstruction_err_:.4f}")

top_nmf_tfidf = top_terms(H_tfidf, vocab_tfidf, TOP_N)
print("\n=== NMF on TF-IDF — top terms per topic ===")
print(top_nmf_tfidf.groupby("topic").head(10).to_string(index=False))

doc_load_tfidf = pd.DataFrame(W_tfidf, columns=[f"NMF_TFIDF_topic_{i}" for i in range(K)])
doc_load_tfidf.insert(0, "doc_id", range(1, len(documents) + 1))
doc_load_tfidf.insert(1, "filename", filenames)
doc_load_tfidf.insert(2, "topic_label", topics)
doc_load_tfidf["NMF_TFIDF_argmax"] = doc_load_tfidf[
    [c for c in doc_load_tfidf.columns if c.startswith("NMF_TFIDF_topic_")]] \
    .values.argmax(axis=1)

print("\n=== NMF on TF-IDF — document loadings (all docs) ===")
cols_view = ['filename', 'topic_label'] + [f"NMF_TFIDF_topic_{i}" for i in range(K)] + ["NMF_TFIDF_argmax"]
print(doc_load_tfidf[cols_view].round(3).to_string(index=False))

count_vect = CountVectorizer(min_df=1, max_df=0.9)
X_counts = count_vect.fit_transform(clean_docs)
vocab_bow = np.array(count_vect.get_feature_names_out())
print("\nBoW shape:", X_counts.shape)

X_counts_log = X_counts.copy()
X_counts_log.data = np.log1p(X_counts_log.data)

nmf_bow = NMF(n_components=K, random_state=42, init="nndsvda", max_iter=500)
W_bow = nmf_bow.fit_transform(X_counts_log)
H_bow = nmf_bow.components_

print(f"Reconstruction error (BoW+log1p): {nmf_bow.reconstruction_err_:.4f}")

top_nmf_bow = top_terms(H_bow, vocab_bow, TOP_N)
print("\n=== NMF on BoW (log1p) — top terms per topic ===")
print(top_nmf_bow.groupby("topic").head(10).to_string(index=False))

doc_load_bow = pd.DataFrame(W_bow, columns=[f"NMF_BOW_topic_{i}" for i in range(K)])
doc_load_bow.insert(0, "doc_id", range(1, len(documents) + 1))
doc_load_bow.insert(1, "filename", filenames)
doc_load_bow.insert(2, "topic_label", topics)

print("\n=== NMF on BoW (log1p) — document loadings (all docs) ===")
cols_view_bow = ['filename', 'topic_label'] + [f"NMF_BOW_topic_{i}" for i in range(K)]
print(doc_load_bow[cols_view_bow].round(3).to_string(index=False))

Zt_tfidf = normalize(W_tfidf, norm="l2")
Zt_bow = normalize(W_bow, norm="l2")
S_tfidf = cosine_similarity(Zt_tfidf)
S_bow = cosine_similarity(Zt_bow)

print("\nCosine similarity in NMF topic space (TF-IDF) — first 5x5 (rounded):")
print(pd.DataFrame(S_tfidf, index=filenames, columns=filenames).iloc[:5, :5].round(2).to_string())

print("\nCosine similarity in NMF topic space (BoW+log1p) — first 5x5 (rounded):")
print(pd.DataFrame(S_bow, index=filenames, columns=filenames).iloc[:5, :5].round(2).to_string())

if SAVE_CSVS:
    top_nmf_tfidf.to_csv("top_terms_NMF_TFIDF.csv", index=False)
    doc_load_tfidf.to_csv("doc_loadings_NMF_TFIDF.csv", index=False)

    top_nmf_bow.to_csv("top_terms_NMF_BOW.csv", index=False)
    doc_load_bow.to_csv("doc_loadings_NMF_BOW.csv", index=False)

    print("\nSaved:")
    print("  - cleaned_documents.csv")
    print("  - top_terms_NMF_TFIDF.csv")
    print("  - doc_loadings_NMF_TFIDF.csv")
    print("  - top_terms_NMF_BOW.csv")
    print("  - doc_loadings_NMF_BOW.csv")

print("\nDone.")
