# augment_paraphrases_eng.py
# Model: gpt-4.1-mini
# API key from config.yaml / config.json
# Includes a progress bar (tqdm)

import time
import json
import random
from pathlib import Path
from typing import List

import pandas as pd
from tqdm import tqdm
from openai import OpenAI

# -------------------------
# PATHS
# -------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_CSV = PROJECT_ROOT / "data/dev_phase/subtask1/train/eng.csv"

CONFIG_YAML = PROJECT_ROOT / "config_api.yaml"
CONFIG_JSON = PROJECT_ROOT / "config_api.json"

# -------------------------
# DATA COLUMNS
# -------------------------
TEXT_COL = "text"
LABEL_COL = "polarization"

# -------------------------
# MODEL CONFIG
# -------------------------
MODEL = "gpt-4.1-mini"
TEMPERATURE = 0.3
MAX_OUTPUT_TOKENS = 220

EXTRA_PER_CLASS = 400
SEED = 42

# -------------------------
# FILTERING
# -------------------------
MIN_LEN_RATIO = 0.75
MAX_LEN_RATIO = 1.33
BANNED_PHRASES = [
    "some people say", "it seems", "allegedly", "reportedly",
    "in my opinion", "it is believed", "it appears", "may have"
]

# -------------------------
# OUTPUT
# -------------------------
OUT_CSV = DATA_CSV.with_name("eng_aug.csv")
LOG_JSONL = DATA_CSV.with_name("eng_aug_log.jsonl")


# -------------------------
# LOAD API KEY
# -------------------------
def load_api_key() -> str:
    if CONFIG_YAML.exists():
        import yaml
        with open(CONFIG_YAML, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)["openai_api_key"]
    if CONFIG_JSON.exists():
        with open(CONFIG_JSON, "r", encoding="utf-8") as f:
            return json.load(f)["openai_api_key"]
    raise RuntimeError("Missing config.yaml or config.json with openai_api_key")


client = OpenAI(api_key=load_api_key())

# -------------------------
# PROMPT
# -------------------------
INSTRUCTIONS = """
You are paraphrasing online text for a POLARIZATION (binary) classification dataset.

Rules:
- Preserve stance, target group(s), sentiment, and polarization intensity.
- Do NOT add or remove accusations, threats, slurs, or calls to action.
- Do NOT introduce hedging or neutrality (e.g. "some people say", "it seems").
- Keep English language and original style (informal if present).
- Keep hashtags, @mentions, emojis if present.
- Output ONLY the paraphrased text.
""".strip()


# -------------------------
# HELPERS
# -------------------------
def paraphrase_one(text: str, max_tries: int = 8) -> str:
    base_sleep = 1.0
    for attempt in range(max_tries):
        try:
            r = client.responses.create(
                model=MODEL,
                instructions=INSTRUCTIONS,
                input=text,
                temperature=TEMPERATURE,
                max_output_tokens=MAX_OUTPUT_TOKENS,
            )
            return (r.output_text or "").strip()
        except Exception:
            time.sleep(min(30, base_sleep * (2 ** attempt) + random.random()))
    raise RuntimeError("API failed after retries")


def passes_filters(orig: str, para: str) -> bool:
    if not para:
        return False
    ratio = len(para) / max(1, len(orig))
    if ratio < MIN_LEN_RATIO or ratio > MAX_LEN_RATIO:
        return False
    low = para.lower()
    if any(p in low for p in BANNED_PHRASES):
        return False
    return True


def sample_idxs(idxs: List[int], n: int) -> List[int]:
    if n <= len(idxs):
        return random.sample(idxs, n)
    return [random.choice(idxs) for _ in range(n)]


def augment(df, idxs, label, tag):
    rows = []
    with open(LOG_JSONL, "a", encoding="utf-8") as f:
        for i in tqdm(idxs, desc=f"Augment {tag}", leave=False):
            orig = str(df.loc[i, TEXT_COL])
            para = paraphrase_one(orig)
            ok = passes_filters(orig, para)

            f.write(json.dumps({
                "row": int(i),
                "label": int(label),
                "tag": tag,
                "ok": ok,
                "orig": orig,
                "para": para,
            }, ensure_ascii=False) + "\n")

            if ok:
                rows.append({
                    TEXT_COL: para,
                    LABEL_COL: label,
                    "augmented": True,
                    "aug_tag": tag,
                    "orig_row": int(i),
                })
    return pd.DataFrame(rows)


# -------------------------
# MAIN
# -------------------------
def main():
    random.seed(SEED)

    df = pd.read_csv(DATA_CSV)

    base = df.copy()
    base["augmented"] = False
    base["aug_tag"] = "original"
    base["orig_row"] = base.index

    idx0 = df.index[df[LABEL_COL] == 0].tolist()
    idx1 = df.index[df[LABEL_COL] == 1].tolist()

    print(f"Original counts: 0={len(idx0)}, 1={len(idx1)}")

    if len(idx0) > len(idx1):
        aug_bal = augment(df, sample_idxs(idx1, len(idx0) - len(idx1)), 1, "balance_pos")
    elif len(idx1) > len(idx0):
        aug_bal = augment(df, sample_idxs(idx0, len(idx1) - len(idx0)), 0, "balance_neg")
    else:
        aug_bal = pd.DataFrame()

    aug0 = augment(df, sample_idxs(idx0, EXTRA_PER_CLASS), 0, "extra_neg")
    aug1 = augment(df, sample_idxs(idx1, EXTRA_PER_CLASS), 1, "extra_pos")

    final = pd.concat([base, aug_bal, aug0, aug1], ignore_index=True)
    final.to_csv(OUT_CSV, index=False)

    print("\nFinal counts:")
    print(final[LABEL_COL].value_counts())
    print("Saved:", OUT_CSV)


if __name__ == "__main__":
    main()
