from pathlib import Path
import csv
import json
import time
import yaml
import pandas as pd
from openai import OpenAI
from tqdm import tqdm

# =========================
# PATHS
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent

DATA_CSV = PROJECT_ROOT / "data/dev_phase/subtask1/dev/eng.csv"
OUTPUT_CSV = PROJECT_ROOT / "pred.csv"

CONFIG_YAML = PROJECT_ROOT / "config_api.yaml"
CONFIG_JSON = PROJECT_ROOT / "config_api.json"

# =========================
# DATA COLUMNS
# =========================
ID_COL = "id"
TEXT_COL = "text"
LABEL_COL = "polarization"

# =========================
# MODEL CONFIG
# =========================
MODEL = "gpt-4.1-mini"
TEMPERATURE = 0.0

MAX_OUTPUT_TOKENS = 600
BATCH_SIZE = 200
SLEEP_BETWEEN_BATCHES = 0.5

DEFAULT_LABEL = 0


# =========================
# LOAD API KEY
# =========================
def load_api_key():
    if CONFIG_YAML.exists():
        with open(CONFIG_YAML, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)["openai_api_key"]

    if CONFIG_JSON.exists():
        with open(CONFIG_JSON, "r", encoding="utf-8") as f:
            return json.load(f)["openai_api_key"]

    raise FileNotFoundError("No config_api.yaml or config_api.json found")


# =========================
# PROMPT (POLAR-aligned, SIMPLE OUTPUT)
# =========================
SYSTEM_PROMPT = """You are a classifier for online polarization (POLAR).

Polarization means sharp division or hostility between social, political, or identity groups.
Label 1 ONLY when the text clearly contains inter-group antagonism or “us vs them” framing
(e.g., group blame, demonization, partisan/identity attacks, calls to oppose/hate a group).
Otherwise label 0.

Do not infer polarization from topic alone. If unsure, output 0.
"""


def build_user_prompt(rows):
    lines = [
        "For each item, output exactly one line in this format:",
        "id<TAB>polarization",
        "where polarization is a single digit: 0 or 1.",
        "Output ONLY these lines. No header. No extra text.",
        "",
    ]
    for rid, text in rows:
        lines.append(f"{rid}\t{text}")
    return "\n".join(lines)


def _response_text_any(response) -> str:
    """
    Robustly extract text from OpenAI Responses API across SDK versions.
    Prefer response.output_text when available, else walk response.output.
    """
    txt = getattr(response, "output_text", None)
    if isinstance(txt, str) and txt.strip():
        return txt.strip()

    out = getattr(response, "output", None)
    if not out:
        return ""

    chunks = []
    for msg in out:
        content = getattr(msg, "content", None) or []
        for part in content:
            t = getattr(part, "text", None)
            if isinstance(t, str) and t:
                chunks.append(t)
            else:
                try:
                    d = part.model_dump()
                    if isinstance(d, dict) and isinstance(d.get("text"), str):
                        chunks.append(d["text"])
                except Exception:
                    pass

    return "\n".join(chunks).strip()


# =========================
# OPENAI CALL (SIMPLE PARSING)
# =========================
def classify_batch(client, rows):
    response = client.responses.create(
        model=MODEL,
        temperature=TEMPERATURE,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(rows)},
        ],
    )

    raw = _response_text_any(response)
    preds = {}

    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        if "\t" not in line:
            continue

        rid, label = line.split("\t", 1)
        rid = rid.strip()
        label = label.strip()

        if label in ("0", "1"):
            preds[rid] = int(label)

    return preds


# =========================
# MAIN
# =========================
def main():
    client = OpenAI(api_key=load_api_key())

    df = pd.read_csv(DATA_CSV)

    if ID_COL not in df.columns or TEXT_COL not in df.columns:
        raise ValueError(f"Input CSV must contain columns: {ID_COL}, {TEXT_COL}")

    df[ID_COL] = df[ID_COL].astype(str)
    df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str)

    rows = list(zip(df[ID_COL].tolist(), df[TEXT_COL].tolist()))
    predictions = {}

    num_batches = (len(rows) + BATCH_SIZE - 1) // BATCH_SIZE
    for i in tqdm(range(0, len(rows), BATCH_SIZE), total=num_batches, desc="Classifying"):
        batch = rows[i: i + BATCH_SIZE]
        batch_preds = classify_batch(client, batch)

        # Ensure every id gets a prediction (default 0)
        for rid, _ in batch:
            predictions[rid] = int(batch_preds.get(rid, DEFAULT_LABEL))

        time.sleep(SLEEP_BETWEEN_BATCHES)

    # =========================
    # WRITE CSV (STRICT FORMAT)
    # =========================
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow([ID_COL, LABEL_COL])
        for rid in df[ID_COL].tolist():  # preserve input order
            writer.writerow([rid, predictions.get(rid, DEFAULT_LABEL)])

    print(f"Saved predictions to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
