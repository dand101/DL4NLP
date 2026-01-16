# import json
# from pathlib import Path
# from openai import OpenAI
#
# PROJECT_ROOT = Path(__file__).resolve().parent
# CONFIG_YAML = PROJECT_ROOT / "config_api.yaml"
# CONFIG_JSON = PROJECT_ROOT / "config_api.json"
#
# def load_api_key() -> str:
#     if CONFIG_YAML.exists():
#         import yaml
#         with open(CONFIG_YAML, "r", encoding="utf-8") as f:
#             return yaml.safe_load(f)["openai_api_key"]
#     if CONFIG_JSON.exists():
#         with open(CONFIG_JSON, "r", encoding="utf-8") as f:
#             return json.load(f)["openai_api_key"]
#     raise RuntimeError("No API key config found")
#
# client = OpenAI(api_key=load_api_key())
#
# INSTRUCTIONS = """
# Paraphrase the text without changing stance, sentiment, or intensity.
# Output ONLY the paraphrased text.
# """.strip()
#
# TEST_TEXT = "These people are destroying our country and nobody is stopping them."
#
# print("Sending test request...")
#
# resp = client.responses.create(
#     model="gpt-4.1-mini",
#     instructions=INSTRUCTIONS,
#     input=TEST_TEXT,
#     temperature=0.3,
#     max_output_tokens=100,
# )
#
# print("\nORIGINAL:")
# print(TEST_TEXT)
#
# print("\nPARAPHRASE:")
# print(resp.output_text)
import pandas as pd
df = pd.read_csv("data/dev_phase/subtask1/train/eng_aug.csv")
print(df["augmented"].value_counts())
