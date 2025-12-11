# ============================================
# POLAR Subtask 1 - Exploratory Data Analysis
# ============================================

import os
import glob
import re
from collections import Counter

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def simple_tokenize(text: str):
    text = str(text).lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^\w]+", " ", text)
    return text.split()


# ---------------- CONFIG --------------------
DATA_ROOT = "../data/dev_phase/subtask1"
SPLIT = "train"
TEXT_COL = "text"
LABEL_COL = "polarization"
ID_COL = "id"
LANG_COL = "lang"

SAVE_FIGS = True
FIG_DIR = "./figures"
# -------------------------------------------

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)


def maybe_savefig(name: str):
    """
    Save current figure into FIG_DIR/name and close it.
    name can contain subfolders, e.g. 'correlations/train_eng.png'.
    """
    if SAVE_FIGS:
        full_path = os.path.join(FIG_DIR, name)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        plt.savefig(full_path, bbox_inches="tight")
        print(f"Saved figure to: {full_path}")
    plt.close()


# ============= 1. LOAD DATA ================
split_dir = os.path.join(DATA_ROOT, SPLIT)
csv_files = glob.glob(os.path.join(split_dir, "*.csv"))

print(f"Found {len(csv_files)} CSV files in {split_dir}")
assert csv_files, "No CSV files found – check DATA_ROOT and SPLIT."

dfs = []
for path in csv_files:
    lang = os.path.splitext(os.path.basename(path))[0]
    df_lang = pd.read_csv(path)
    df_lang[LANG_COL] = lang
    dfs.append(df_lang)

df = pd.concat(dfs, ignore_index=True)

print("\n=== GLOBAL SHAPE ===")
print(df.shape)
print("\n=== FIRST 5 ROWS ===")
print(df.head())

print("\n=== COLUMNS ===")
print(df.columns)

# ============= 2. BASIC INFO ================
print("\n=== DATA TYPES ===")
print(df.dtypes)

print("\n=== BASIC INFO (non-null counts) ===")
print(df.info())

print("\n=== SAMPLE ROWS (random) ===")
print(df.sample(5, random_state=42))

print("\n=== MISSING VALUES PER COLUMN ===")
print(df.isna().sum())

print("\n=== NUMBER OF DUPLICATED ROWS ===")
print(df.duplicated().sum())

print("\n=== NUMBER OF DUPLICATED TEXTS (same text, any lang) ===")
print(df.duplicated(subset=[TEXT_COL]).sum())

# ============= 3. LABEL & LANGUAGE DISTRIBUTIONS ===========
print("\n=== GLOBAL LABEL DISTRIBUTION ===")
print(df[LABEL_COL].value_counts())
print("\n=== GLOBAL LABEL DISTRIBUTION (NORMALIZED) ===")
print(df[LABEL_COL].value_counts(normalize=True))

print("\n=== NUMBER OF EXAMPLES PER LANGUAGE ===")
print(df[LANG_COL].value_counts())

plt.figure()
sns.countplot(data=df, x=LABEL_COL)
plt.title(f"Global Label Distribution ({SPLIT})")
plt.xlabel("Polarization label")
plt.ylabel("Count")
maybe_savefig(f"{SPLIT}_global_label_distribution.png")

plt.figure(figsize=(12, 4))
sns.countplot(
    data=df,
    x=LANG_COL,
    order=df[LANG_COL].value_counts().index
)
plt.title(f"Number of Examples per Language ({SPLIT})")
plt.xlabel("Language")
plt.ylabel("Count")
plt.xticks(rotation=45)
maybe_savefig(f"{SPLIT}_samples_per_language.png")

label_lang_counts = pd.crosstab(df[LANG_COL], df[LABEL_COL])
print("\n=== CROSSTAB: LANGUAGE x LABEL ===")
print(label_lang_counts)

imbalance = df.groupby(LANG_COL)[LABEL_COL].mean()
imbalance.sort_values().plot(kind="bar", figsize=(12, 4))
plt.title("Proportion of Polarized Posts per Language")
maybe_savefig("imbalance_ratio_per_language.png")

from scipy.stats import entropy


def label_entropy(x):
    counts = x.value_counts(normalize=True)
    return entropy(counts)


entropy_df = df.groupby(LANG_COL)[LABEL_COL].apply(label_entropy)
print("\n=== LABEL ENTROPY PER LANGUAGE ===")
print(entropy_df)

plt.figure(figsize=(12, 5))
label_lang_counts.plot(kind="bar", stacked=False)
plt.title("Label Distribution per Language")
plt.xlabel("Language")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
maybe_savefig(f"{SPLIT}_label_distribution_per_language.png")

# ============= 4. TEXT LENGTH ANALYSIS =====================
df["char_len"] = df[TEXT_COL].astype(str).str.len()
df["word_len"] = df[TEXT_COL].astype(str).str.split().str.len()

print("\n=== TEXT LENGTH SUMMARY (CHARACTERS) ===")
print(df["char_len"].describe())

print("\n=== TEXT LENGTH SUMMARY (WORDS) ===")
print(df["word_len"].describe())

plt.figure()
sns.histplot(df["word_len"], bins=50)
plt.title("Distribution of Text Length (words) – All languages")
plt.xlabel("Number of words")
plt.ylabel("Frequency")
maybe_savefig(f"{SPLIT}_word_length_hist_all.png")

plt.figure()
sns.boxplot(data=df, x=LABEL_COL, y="word_len")
plt.title("Text Length by Label (all languages)")
plt.xlabel("Polarization label")
plt.ylabel("Number of words")
maybe_savefig(f"{SPLIT}_word_length_by_label.png")

plt.figure(figsize=(12, 4))
sns.boxplot(data=df, x=LANG_COL, y="word_len")
plt.title("Text Length per Language")
plt.xlabel("Language")
plt.ylabel("Number of words")
plt.xticks(rotation=45)
maybe_savefig(f"{SPLIT}_word_length_by_language.png")

lang_stats = df.groupby(LANG_COL).agg(
    n_samples=(ID_COL, "count"),
    mean_char_len=("char_len", "mean"),
    median_char_len=("char_len", "median"),
    mean_word_len=("word_len", "mean"),
    median_word_len=("word_len", "median"),
    pct_polarized=(LABEL_COL, lambda x: (x == 1).mean()),
)
print("\n=== PER-LANGUAGE STATISTICS ===")
print(lang_stats.sort_values("n_samples", ascending=False).to_string())

# ============= 5. TEXT QUALITY / NOISE =====================
url_pattern = re.compile(r"http\S+")
mention_pattern = re.compile(r"@\w+")
hashtag_pattern = re.compile(r"#\w+")
emoji_pattern = re.compile(
    r"[\U0001F600-\U0001F64F"
    r"\U0001F300-\U0001F5FF"
    r"\U0001F680-\U0001F6FF"
    r"\U0001F1E0-\U0001F1FF]",
    flags=re.UNICODE,
)


def has_pattern(pattern, text):
    return bool(pattern.search(str(text)))


df["has_url"] = df[TEXT_COL].apply(lambda t: has_pattern(url_pattern, t))
df["has_mention"] = df[TEXT_COL].apply(lambda t: has_pattern(mention_pattern, t))
df["has_hashtag"] = df[TEXT_COL].apply(lambda t: has_pattern(hashtag_pattern, t))
df["has_emoji"] = df[TEXT_COL].apply(lambda t: has_pattern(emoji_pattern, t))

print("\n=== NOISE INDICATORS (GLOBAL) ===")
for col in ["has_url", "has_mention", "has_hashtag", "has_emoji"]:
    print(f"{col}: {df[col].mean():.3f} proportion of texts")

very_short = df[df["word_len"] <= 2]
very_long = df[df["word_len"] >= 100]

print(f"\nVERY SHORT TEXTS (<=2 words): {len(very_short)}")
print(f"VERY LONG TEXTS (>=100 words): {len(very_long)}")

print("\nExamples of VERY short texts:")
print(very_short[[TEXT_COL, LABEL_COL, LANG_COL]].head())

print("\nExamples of VERY long texts:")
print(very_long[[TEXT_COL, LABEL_COL, LANG_COL]].head())


def vocab_size(text_series: pd.Series) -> int:
    vocab = set()
    for x in text_series.dropna().astype(str):
        vocab.update(simple_tokenize(x))
    return len(vocab)


vocab_stats = df.groupby(LANG_COL)[TEXT_COL].apply(vocab_size)
print("\n=== VOCABULARY SIZE PER LANGUAGE ===")
print(vocab_stats.sort_values(ascending=False))


def ttr(text_series: pd.Series) -> float:
    tokens = []
    for x in text_series.dropna().astype(str):
        tokens.extend(simple_tokenize(x))
    n_tokens = len(tokens)
    if n_tokens == 0:
        return np.nan
    return len(set(tokens)) / n_tokens


ttr_vals = df.groupby(LANG_COL)[TEXT_COL].apply(ttr)
print("\n=== TYPE–TOKEN RATIO PER LANGUAGE ===")
print(ttr_vals.sort_values(ascending=False))


# ============= 6. TOKEN FREQUENCY ANALYSIS =================

def get_top_tokens(sub_df: pd.DataFrame, n: int = 20):
    counter = Counter()
    for t in sub_df[TEXT_COL].dropna():
        counter.update(simple_tokenize(t))
    most_common = counter.most_common(n)
    return pd.DataFrame(most_common, columns=["token", "freq"])


for label_value in sorted(df[LABEL_COL].unique()):
    print(f"\n=== TOP TOKENS for label={label_value} (GLOBAL) ===")
    print(get_top_tokens(df[df[LABEL_COL] == label_value], n=20))

os.makedirs(os.path.join(FIG_DIR, "token_freq"), exist_ok=True)
for lang in sorted(df[LANG_COL].unique()):
    for label_value in sorted(df[LABEL_COL].unique()):
        subset = df[(df[LANG_COL] == lang) & (df[LABEL_COL] == label_value)]
        top_df = get_top_tokens(subset, n=20)
        out_path = os.path.join(
            FIG_DIR, "token_freq", f"{SPLIT}_top_tokens_{lang}_label{label_value}.csv"
        )
        top_df.to_csv(out_path, index=False)
        print(f"Saved top tokens table: {out_path}")

# ============= 7. GLOBAL + PER-LANGUAGE CORRELATIONS =======
numeric_cols = [LABEL_COL, "char_len", "word_len"]
corr = df[numeric_cols].corr()
print("\n=== CORRELATION MATRIX (GLOBAL) ===")
print(corr)

plt.figure()
sns.heatmap(corr, annot=True, fmt=".2f", vmin=-1, vmax=1)
plt.title("Correlation between length features and label (all languages)")
maybe_savefig(f"{SPLIT}_correlation_heatmap_global.png")

for lang in sorted(df[LANG_COL].unique()):
    sub = df[df[LANG_COL] == lang]
    sub_corr = sub[numeric_cols].corr()

    print(f"\n=== CORRELATION MATRIX – {lang.upper()} ===")
    print(sub_corr)

    plt.figure(figsize=(4, 3))
    sns.heatmap(sub_corr, annot=True, fmt=".2f", vmin=-1, vmax=1)
    plt.title(f"Correlation – {lang.upper()}")
    maybe_savefig(f"correlations/{SPLIT}_corr_{lang}.png")

for lang in sorted(df[LANG_COL].unique()):
    sub = df[df[LANG_COL] == lang]
    plt.figure(figsize=(6, 4))
    sns.histplot(sub["word_len"], bins=40)
    plt.title(f"Text Length Distribution (words) – {lang.upper()}")
    plt.xlabel("Number of words")
    plt.ylabel("Frequency")
    maybe_savefig(f"length_histograms/{SPLIT}_wordlen_{lang}.png")

for lang in sorted(df[LANG_COL].unique()):
    sub = df[df[LANG_COL] == lang]
    plt.figure(figsize=(5, 4))
    sns.boxplot(data=sub, x=LABEL_COL, y="word_len")
    plt.title(f"Text Length by Label – {lang.upper()}")
    plt.xlabel("Polarization label")
    plt.ylabel("Number of words")
    maybe_savefig(f"boxplots_label/{SPLIT}_box_label_{lang}.png")

from wordcloud import WordCloud

for lang in df[LANG_COL].unique():
    for label in [0, 1]:
        subset = df[(df[LANG_COL] == lang) & (df[LABEL_COL] == label)]
        text = " ".join(subset[TEXT_COL].astype(str))
        wc = WordCloud(width=800, height=600).generate(text)
        wc.to_file(f"{FIG_DIR}/wordcloud_{lang}_label{label}.png")

print("\nEDA FINISHED.")
