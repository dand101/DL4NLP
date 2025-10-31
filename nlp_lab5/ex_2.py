import os, re, glob
import numpy as np
import pandas as pd
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('stopwords', quiet=True)

TEXTS_DIR = "texts"
LOAD_FROM_DIR = True
K = 3
TOP_N = 15
SAVE_CSVS = True

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


def guess_topic_from_name(name: str) -> str:
    m = re.match(r"([A-Za-z0-9\- ]+)_", name)
    return m.group(1).lower() if m else "unknown"


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

STOPWORDS = set(stopwords.words("english"))
lemma = WordNetLemmatizer()


def preprocess(text: str) -> str:
    tokens = re.findall(r"[a-z]+", text.lower())
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    tokens = [lemma.lemmatize(t) for t in tokens]
    return " ".join(tokens)


clean_docs = [preprocess(d) for d in documents]

count_vect = CountVectorizer(min_df=1, max_df=0.9)
tfidf_vect = TfidfVectorizer(min_df=1, max_df=0.9)

X_counts = count_vect.fit_transform(clean_docs)
X_tfidf = tfidf_vect.fit_transform(clean_docs)

vocab_counts = np.array(count_vect.get_feature_names_out())
vocab_tfidf = np.array(tfidf_vect.get_feature_names_out())

print("BoW shape :", X_counts.shape)
print("TF-IDF shape:", X_tfidf.shape)

svd_bow = TruncatedSVD(n_components=K, random_state=42)
Z_bow = svd_bow.fit_transform(X_counts)
C_bow = svd_bow.components_

svd_tfidf = TruncatedSVD(n_components=K, random_state=42)
Z_tfidf = svd_tfidf.fit_transform(X_tfidf)
C_tfidf = svd_tfidf.components_


def top_terms(components, vocab, topn=10):
    rows = []
    for k_idx, comp in enumerate(components):
        order = np.argsort(comp)[::-1][:topn]
        for rank, j in enumerate(order, 1):
            rows.append({"component": k_idx, "rank": rank, "term": vocab[j], "weight": float(comp[j])})
    return pd.DataFrame(rows)


top_lsa_bow = top_terms(C_bow, vocab_counts, TOP_N)
top_lsa_tfidf = top_terms(C_tfidf, vocab_tfidf, TOP_N)

print("\n=== LSA on BoW — top terms per component ===")
print(top_lsa_bow.groupby("component").head(10).to_string(index=False))

print("\n=== LSA on TF-IDF — top terms per component ===")
print(top_lsa_tfidf.groupby("component").head(10).to_string(index=False))

print(f"\nExplained variance (BoW):    {svd_bow.explained_variance_ratio_.sum():.3f}")
print(f"Explained variance (TF-IDF): {svd_tfidf.explained_variance_ratio_.sum():.3f}")

print("\n=== LSA on BoW — document loadings (all docs) ===")
df_bow_load = pd.DataFrame(Z_bow, columns=[f"comp_{i}" for i in range(K)])
df_bow_load.insert(0, "filename", filenames)
if 'topics' in locals():
    df_bow_load.insert(1, "topic", topics)
print(df_bow_load.round(3).to_string(index=False))

print("\n=== LSA on TF-IDF — document loadings (all docs) ===")
df_tfidf_load = pd.DataFrame(Z_tfidf, columns=[f"comp_{i}" for i in range(K)])
df_tfidf_load.insert(0, "filename", filenames)
if 'topics' in locals():
    df_tfidf_load.insert(1, "topic", topics)
print(df_tfidf_load.round(3).to_string(index=False))

if SAVE_CSVS:
    top_lsa_bow.to_csv("top_terms_LSA_BOW.csv", index=False)
    top_lsa_tfidf.to_csv("top_terms_LSA_TFIDF.csv", index=False)

    df_bow = pd.DataFrame(Z_bow, columns=[f"LSA_BOW_comp_{i}" for i in range(K)])
    df_bow.insert(0, "doc_id", range(1, len(documents) + 1))
    df_bow.insert(1, "filename", filenames)
    df_bow.insert(2, "topic", topics)
    df_bow["LSA_BOW_argmax"] = df_bow[[c for c in df_bow.columns if c.startswith("LSA_BOW_comp_")]].values.argmax(
        axis=1)
    df_bow.to_csv("doc_loadings_LSA_BOW.csv", index=False)

    df_tfidf = pd.DataFrame(Z_tfidf, columns=[f"LSA_TFIDF_comp_{i}" for i in range(K)])
    df_tfidf.insert(0, "doc_id", range(1, len(documents) + 1))
    df_tfidf.insert(1, "filename", filenames)
    df_tfidf.insert(2, "topic", topics)
    df_tfidf["LSA_TFIDF_argmax"] = df_tfidf[
        [c for c in df_tfidf.columns if c.startswith("LSA_TFIDF_comp_")]].values.argmax(axis=1)
    df_tfidf.to_csv("doc_loadings_LSA_TFIDF.csv", index=False)

    print("\nSaved:")
    print("  - top_terms_LSA_BOW.csv")
    print("  - top_terms_LSA_TFIDF.csv")
    print("  - doc_loadings_LSA_BOW.csv")
    print("  - doc_loadings_LSA_TFIDF.csv")
