import os, re
import wikipediaapi
import nltk

nltk.download('punkt', quiet=True)

SAVE_DIR = "texts"
os.makedirs(SAVE_DIR, exist_ok=True)

topics = {
    "technology": [
        "Artificial intelligence",
        "Machine learning",
        "Computer network",
        "Cybersecurity",
        "Internet of things"
    ],
    "music": [
        "Classical music",
        "Jazz",
        "Rock music",
        "Hip hop music",
        "Electronic music"
    ],
    "animals": [
        "Mammal",
        "Bird",
        "Reptile",
        "Amphibian",
        "Fish"
    ]
}

wiki = wikipediaapi.Wikipedia(
    user_agent="DL4NLP-lab (https://github.com/dan; contact: student@example.com)",
    language="en"
)


def clean_text(text):
    text = re.sub(r"\[[0-9]+\]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def fetch_summary(title, max_paragraphs=3):
    page = wiki.page(title)
    if not page.exists():
        print(f"Page not found: {title}")
        return ""
    raw = page.text
    paragraphs = raw.split("\n")
    first_paras = [p for p in paragraphs if p.strip()][:max_paragraphs]
    return clean_text(" ".join(first_paras))


count = 0
for topic, titles in topics.items():
    for i, title in enumerate(titles, 1):
        print(f"Fetching [{topic}] → {title}")
        text = fetch_summary(title)
        if not text:
            continue
        filename = f"{topic}_{i}.txt"
        out_path = os.path.join(SAVE_DIR, filename)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
        count += 1

print(f"\n✅ Saved {count} documents in '{SAVE_DIR}'")

for name in sorted(os.listdir(SAVE_DIR))[:5]:
    path = os.path.join(SAVE_DIR, name)
    size = os.path.getsize(path)
    print(f"  {name} ({size} bytes)")
