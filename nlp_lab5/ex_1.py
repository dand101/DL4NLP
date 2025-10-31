import os, re, glob
import numpy as np
import pandas as pd
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
nltk.download("stopwords", quiet=True)

LOAD_FROM_DIR = True
TEXTS_DIR = "texts"
SAVE_CSVS = True
TOP_N = 15

FALLBACK = [
    "Example astronomy text about galaxies and exoplanets.",
    "Example astronomy text about black holes and event horizons.",
    "Example astronomy text about rovers and Martian regolith.",
    "Example hoop text about NBA pick-and-roll and spacing.",
    "Example hoop text about rebounds and rim protection.",
    "Example cooking text about bread, gluten, and fermentation.",
]

if LOAD_FROM_DIR:
    paths = sorted(glob.glob(os.path.join(TEXTS_DIR, "*.txt")))
    if not paths:
        print(f"No .txt files found in {TEXTS_DIR}; using fallback examples.")
        documents = FALLBACK
        filenames = [f"fallback_{i + 1}.txt" for i in range(len(FALLBACK))]
    else:
        documents, filenames = [], []
        for p in paths:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                documents.append(f.read())
            filenames.append(os.path.basename(p))
else:
    documents = FALLBACK
    filenames = [f"manual_{i + 1}.txt" for i in range(len(FALLBACK))]

print(f"Loaded {len(documents)} documents.")

STOPWORDS = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def preprocess(text: str) -> str:
    tokens = re.findall(r"[a-z]+", text.lower())
    clean = [lemmatizer.lemmatize(t) for t in tokens if t not in STOPWORDS and len(t) > 1]
    return " ".join(clean)


clean_docs = [preprocess(doc) for doc in documents]

if SAVE_CSVS:
    pd.DataFrame({"filename": filenames, "original": documents, "cleaned": clean_docs}) \
        .to_csv("cleaned_documents.csv", index=False)

print("\nPreprocessing done (stopwords removed, lemmatized).")

count_vect = CountVectorizer(min_df=1, max_df=0.9)
tfidf_vect = TfidfVectorizer(min_df=1, max_df=0.9)

X_counts = count_vect.fit_transform(clean_docs)
X_tfidf = tfidf_vect.fit_transform(clean_docs)

vocab_counts = np.array(count_vect.get_feature_names_out())
vocab_tfidf = np.array(tfidf_vect.get_feature_names_out())

print(f"\nBag-of-Words shape: {X_counts.shape}")
print(f"TF-IDF shape:       {X_tfidf.shape}")

term_counts = np.asarray(X_counts.sum(axis=0)).ravel()
order_counts = np.argsort(term_counts)[::-1][:TOP_N]
print("\nTop terms (BoW total frequency):")
for rank, idx in enumerate(order_counts, start=1):
    print(f"{rank:>2}. {vocab_counts[idx]:<15} {int(term_counts[idx])}")

term_tfidf = np.asarray(X_tfidf.mean(axis=0)).ravel()
order_tfidf = np.argsort(term_tfidf)[::-1][:TOP_N]
print("\nTop terms (TF-IDF mean weight):")
for rank, idx in enumerate(order_tfidf, start=1):
    print(f"{rank:>2}. {vocab_tfidf[idx]:<15} {term_tfidf[idx]:.4f}")

if SAVE_CSVS:
    pd.Series(vocab_counts).to_csv("vocabulary_BoW.txt", index=False, header=False)
    pd.Series(vocab_tfidf).to_csv("vocabulary_TFIDF.txt", index=False, header=False)
    print("\nSaved:")
    print("  - cleaned_documents.csv")
    print("  - vocabulary_BoW.txt")
    print("  - vocabulary_TFIDF.txt")

print("\nDone.")
