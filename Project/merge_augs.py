import re
from pathlib import Path
import pandas as pd

# ----------------------------
# PATHS (EDIT IF NEEDED)
# ----------------------------
HERE = Path(__file__).resolve().parent

ENG_AUG_CSV = HERE / "data/dev_phase/subtask1/train/eng_aug.csv"
BACKTRANS_CSV = HERE / "data/dev_phase/subtask1/train/train_augmented_backtranslation.csv"
OUT_CSV = HERE / "data/dev_phase/subtask1/train/eng_aug_plus_backtranslated.csv"


ID_COL = "id"
TEXT_COL = "text"
LABEL_COL = "polarization"

# ----------------------------
# Helpers
# ----------------------------
def normalize_text(s: str) -> str:
    s = str(s)
    s = s.replace("\u200b", "")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def ensure_columns(df: pd.DataFrame, source_tag: str) -> pd.DataFrame:
    if TEXT_COL not in df.columns:
        raise ValueError(f"[{source_tag}] Missing '{TEXT_COL}' column")

    if LABEL_COL not in df.columns:
        raise ValueError(f"[{source_tag}] Missing '{LABEL_COL}' column")

    if "augmented" not in df.columns:
        df["augmented"] = True

    if "aug_tag" not in df.columns:
        df["aug_tag"] = source_tag
    if "orig_row" not in df.columns:
        df["orig_row"] = pd.NA

    if ID_COL not in df.columns:
        df[ID_COL] = [f"{source_tag}_{i}" for i in range(len(df))]

    df[TEXT_COL] = df[TEXT_COL].astype(str).map(normalize_text)
    df[LABEL_COL] = df[LABEL_COL].astype(int)
    df["augmented"] = df["augmented"].astype(bool)

    keep = [ID_COL, TEXT_COL, LABEL_COL, "augmented", "aug_tag", "orig_row"]
    extra = [c for c in df.columns if c not in keep]
    return df[keep + extra]

def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    key = df[TEXT_COL].str.lower() + "||" + df[LABEL_COL].astype(str)
    df = df.copy()
    df["_key"] = key

    df["_priority"] = (~df["augmented"]).astype(int)
    df = df.sort_values(by=["_key", "_priority"], ascending=[True, False])

    df = df.drop_duplicates(subset="_key", keep="first")
    df = df.drop(columns=["_key", "_priority"])
    return df.reset_index(drop=True)

# ----------------------------
# Main
# ----------------------------
def main():
    eng_aug = pd.read_csv(ENG_AUG_CSV)
    eng_aug = ensure_columns(eng_aug, "eng_aug")

    backtrans = pd.read_csv(BACKTRANS_CSV)
    backtrans = ensure_columns(backtrans, "backtrans")

    merged = pd.concat([eng_aug, backtrans], ignore_index=True)
    before = len(merged)

    merged = deduplicate(merged)
    after = len(merged)

    merged.to_csv(OUT_CSV, index=False)

    print("Loaded eng_aug:", len(eng_aug))
    print("Loaded backtranslated:", len(backtrans))
    print(f"Merged rows: {before} → {after} after dedup")
    print("Label counts:\n", merged[LABEL_COL].value_counts())
    print("Augmented counts:\n", merged["augmented"].value_counts())
    print("Saved:", OUT_CSV)

if __name__ == "__main__":
    main()
