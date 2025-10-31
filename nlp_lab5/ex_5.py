import os, re, glob
import numpy as np
import pandas as pd
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.decomposition import LatentDirichletAllocation

TEXTS_DIR = "texts"
LOAD_FROM_DIR = True
K = 3
TOPN_TERMS = 10
MAX_ITER = 400
SAVE_CSVS = True

nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
nltk.download("stopwords", quiet=True)

STOPWORDS = set(stopwords.words("english"))
LEMMA = WordNetLemmatizer()

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
        print(f"⚠️  No .txt files in {TEXTS_DIR}. Using fallback examples.")
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


def preprocess(text: str) -> str:
    tokens = re.findall(r"[a-z]+", text.lower())
    clean = [LEMMA.lemmatize(t) for t in tokens if t not in STOPWORDS and len(t) > 1]
    return " ".join(clean)


clean_docs = [preprocess(d) for d in documents]
clean_tokens = [d.split() for d in clean_docs]

tfidf_vect = TfidfVectorizer(min_df=1, max_df=0.9)
X_tfidf = tfidf_vect.fit_transform(clean_docs)
vocab_tfidf = np.array(tfidf_vect.get_feature_names_out())

count_vect = CountVectorizer(min_df=1, max_df=0.9)
X_counts = count_vect.fit_transform(clean_docs)
vocab_bow = np.array(count_vect.get_feature_names_out())


print("TF-IDF shape:", X_tfidf.shape)
print("BoW shape   :", X_counts.shape)
print(f"Using K={K}")

svd_tfidf = TruncatedSVD(n_components=K, random_state=42, n_iter=10)
Z_lsa = svd_tfidf.fit_transform(X_tfidf)
C_lsa = svd_tfidf.components_

nmf = NMF(n_components=K, init="nndsvda", max_iter=600, random_state=42)
W_nmf = nmf.fit_transform(X_tfidf)
H_nmf = nmf.components_

lda = LatentDirichletAllocation(
    n_components=K, learning_method="batch", max_iter=MAX_ITER,
    random_state=42, doc_topic_prior=None, topic_word_prior=None
)
W_lda = lda.fit_transform(X_counts)
H_lda = lda.components_
H_lda_prob = H_lda / H_lda.sum(axis=1, keepdims=True)


def topics_from_components(H, vocab, topn=10, signed=False):
    topics = []
    for k in range(H.shape[0]):
        comp = H[k]
        if signed:
            order = np.argsort(comp)[::-1][:topn]
        else:
            order = np.argsort(comp)[::-1][:topn]
        topics.append([vocab[j] for j in order])
    return topics


topics_lsa = topics_from_components(C_lsa, vocab_tfidf, topn=TOPN_TERMS, signed=True)
topics_nmf = topics_from_components(H_nmf, vocab_tfidf, topn=TOPN_TERMS, signed=False)

topics_lda = []
for k in range(H_lda_prob.shape[0]):
    idx = np.argsort(H_lda_prob[k])[::-1][:TOPN_TERMS]
    topics_lda.append([vocab_bow[j] for j in idx])

from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel

dictionary = Dictionary(clean_tokens)
corpus = [dictionary.doc2bow(toks) for toks in clean_tokens]


def coherence_all(topic_words, tokens, dictionary, corpus):
    cm_cv = CoherenceModel(topics=topic_words, texts=tokens, dictionary=dictionary, coherence='c_v', processes=1)
    cv = cm_cv.get_coherence()

    cm_npmi = CoherenceModel(topics=topic_words, texts=tokens, dictionary=dictionary, coherence='c_npmi', processes=1)
    cnpmi = cm_npmi.get_coherence()

    cm_umass = CoherenceModel(topics=topic_words, corpus=corpus, dictionary=dictionary, coherence='u_mass', processes=1)
    umass = cm_umass.get_coherence()

    return cv, cnpmi, umass


cv_lsa, cnpmi_lsa, umass_lsa = coherence_all(topics_lsa, clean_tokens, dictionary, corpus)
cv_nmf, cnpmi_nmf, umass_nmf = coherence_all(topics_nmf, clean_tokens, dictionary, corpus)
cv_lda, cnpmi_lda, umass_lda = coherence_all(topics_lda, clean_tokens, dictionary, corpus)


def topic_diversity(topic_words, topn=10):
    words = []
    for tw in topic_words:
        words.extend(tw[:topn])
    return len(set(words)) / (len(topic_words) * topn)


td_lsa = topic_diversity(topics_lsa, TOPN_TERMS)
td_nmf = topic_diversity(topics_nmf, TOPN_TERMS)
td_lda = topic_diversity(topics_lda, TOPN_TERMS)

try:
    perplex = lda.perplexity(X_counts)
except Exception:
    perplex = np.nan

rows = [
    {"model": "LSA (SVD TF-IDF)", "K": K, "c_v": cv_lsa, "c_npmi": cnpmi_lsa, "u_mass": umass_lsa,
     "topic_div@10": td_lsa, "perplexity": np.nan},
    {"model": "NMF (TF-IDF)", "K": K, "c_v": cv_nmf, "c_npmi": cnpmi_nmf, "u_mass": umass_nmf,
     "topic_div@10": td_nmf, "perplexity": np.nan},
    {"model": "LDA (BoW)", "K": K, "c_v": cv_lda, "c_npmi": cnpmi_lda, "u_mass": umass_lda, "topic_div@10": td_lda,
     "perplexity": perplex},
]
metrics = pd.DataFrame(rows)
print("\n=== Topic Model Evaluation Metrics ===")
print(metrics.round(4).to_string(index=False))


def print_topics(name, topic_words, topn=10):
    print(f"\n{name} — Top {topn} terms per topic")
    for k, tw in enumerate(topic_words):
        print(f"  Topic {k}: " + ", ".join(tw[:topn]))


print_topics("LSA", topics_lsa, TOPN_TERMS)
print_topics("NMF", topics_nmf, TOPN_TERMS)
print_topics("LDA", topics_lda, TOPN_TERMS)

if SAVE_CSVS:
    metrics.to_csv("topic_model_metrics.csv", index=False)


    def save_topic_terms(fname, topic_words):
        rows = []
        for k, tw in enumerate(topic_words):
            for r, term in enumerate(tw, 1):
                rows.append({"topic": k, "rank": r, "term": term})
        pd.DataFrame(rows).to_csv(fname, index=False)


    save_topic_terms("topics_LSA.csv", topics_lsa)
    save_topic_terms("topics_NMF.csv", topics_nmf)
    save_topic_terms("topics_LDA.csv", topics_lda)
    print("\nSaved:")
    print("  - topic_model_metrics.csv")
    print("  - topics_LSA.csv")
    print("  - topics_NMF.csv")
    print("  - topics_LDA.csv")
