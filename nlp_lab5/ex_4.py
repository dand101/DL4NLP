import os, re, glob
import numpy as np
import pandas as pd
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('stopwords', quiet=True)

TEXTS_DIR = "texts"
LOAD_FROM_DIR = True
K = 3
TOP_N = 15
MAX_ITER = 200
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
        print(f"⚠️ No .txt files in {TEXTS_DIR}. Using fallback examples.")
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

vectorizer = CountVectorizer(min_df=1, max_df=0.9)
X_counts = vectorizer.fit_transform(clean_docs)
vocab = np.array(vectorizer.get_feature_names_out())

print("BoW shape:", X_counts.shape)

lda = LatentDirichletAllocation(
    n_components=K,
    learning_method="batch",
    max_iter=MAX_ITER,
    random_state=42,
    doc_topic_prior=None,
    topic_word_prior=None
)
W = lda.fit_transform(X_counts)
H = lda.components_
H_prob = H / H.sum(axis=1, keepdims=True)


def top_terms_probs(H_row, vocab, topn=10):
    idx = np.argsort(H_row)[::-1][:topn]
    return [(vocab[j], float(H_row[j])) for j in idx]


rows = []
for k in range(K):
    idx = np.argsort(H_prob[k])[::-1][:TOP_N]
    for r, j in enumerate(idx, 1):
        rows.append({"topic": k, "rank": r, "term": vocab[j], "prob": float(H_prob[k, j])})
top_lda = pd.DataFrame(rows)

print("\n=== LDA — top terms per topic (probabilities) ===")
print(top_lda.groupby("topic").head(10).to_string(index=False))

doc_topics = pd.DataFrame(W, columns=[f"LDA_topic_{i}" for i in range(K)])
doc_topics.insert(0, "doc_id", range(1, len(documents) + 1))
doc_topics.insert(1, "filename", filenames)
doc_topics.insert(2, "topic_label", topics)

print("\n=== LDA — document topic mixtures (all docs) ===")
cols = ["filename", "topic_label"] + [f"LDA_topic_{i}" for i in range(K)]
print(doc_topics[cols].round(3).to_string(index=False))

TOP_DOCS = 5
print("\nTop documents per LDA topic:")
for k in range(K):
    order = np.argsort(W[:, k])[::-1][:TOP_DOCS]
    print("  topic", k, ":", " | ".join(f"{filenames[i]} ({W[i, k]:.3f})" for i in order))

try:
    print(f"\nApprox. log-likelihood: {lda.score(X_counts):.2f}")
    print(f"Perplexity: {lda.perplexity(X_counts):.2f}")
except Exception as e:
    pass

if SAVE_CSVS:
    top_lda.to_csv("top_terms_LDA.csv", index=False)
    doc_topics.to_csv("doc_loadings_LDA.csv", index=False)
    print("\nSaved:")
    print("  - top_terms_LDA.csv")
    print("  - doc_loadings_LDA.csv")
