import random
from collections import defaultdict
from typing import Dict, Set, Tuple
from nltk.corpus import wordnet as wn
import nltk

# ------------------------------
ROUNDS = 10
BASE_SCALE = 100

BONUS = {
    "synonyms": 0.30,
    "hypernyms": 0.22,
    "hyponyms": 0.22,
    "meronyms": 0.12,
    "antonyms": 0.10,
}

try:
    wn.ensure_loaded()
except LookupError:
    nltk.download("wordnet")
    nltk.download("omw-1.4")


# ------------------------------
def _lemma_str(n: str) -> str:
    return n.replace("_", " ").lower()


def _lemmas_for_synset(syn) -> Set[str]:
    return {_lemma_str(l.name()) for l in syn.lemmas()}


def max_noun_wup(a: str, b: str) -> float:
    best = 0.0
    s1 = wn.synsets(a, pos=wn.NOUN)
    s2 = wn.synsets(b, pos=wn.NOUN)
    if not s1 or not s2:
        return 0.0
    for x in s1:
        for y in s2:
            sim = x.wup_similarity(y)
            if sim is not None and sim > best:
                best = float(sim)
                if best == 1.0:
                    return 1.0
    return best


def gather_relations(w: str) -> Dict[str, Set[str]]:
    rel: Dict[str, Set[str]] = defaultdict(set)
    for syn in wn.synsets(w, pos=wn.NOUN):
        rel["synonyms"].update(_lemmas_for_synset(syn))
        for l in syn.lemmas():
            for a in l.antonyms():
                rel["antonyms"].add(_lemma_str(a.name()))
        for h in syn.hypernyms():
            rel["hypernyms"].update(_lemmas_for_synset(h))
        for h in syn.hyponyms():
            rel["hyponyms"].update(_lemmas_for_synset(h))
        for m in syn.part_meronyms() + syn.substance_meronyms() + syn.member_meronyms():
            rel["meronyms"].update(_lemmas_for_synset(m))
    s = w.lower()
    for k in rel:
        rel[k].discard(s)
    return rel


def relation_bonus_as_similarity(u_word: str, b_word: str) -> Tuple[float, str]:
    u, b = gather_relations(u_word), gather_relations(b_word)
    uw, bw = u_word.lower(), b_word.lower()
    for key, extra in BONUS.items():
        if bw in u.get(key, set()) or uw in b.get(key, set()):
            return extra, key
    return 0.0, ""


# ------------------------------
def random_single_word_noun(min_count: int = 1) -> str:
    if not hasattr(random_single_word_noun, "_pool"):
        pool, weights = [], []
        for syn in wn.all_synsets(pos='n'):
            for lem in syn.lemmas():
                name = lem.name()
                if "_" in name or not name.isalpha():
                    continue
                c = lem.count()
                if c >= min_count:
                    pool.append(name.lower())
                    weights.append(max(c, 1))
        if not pool:
            raise RuntimeError("No suitable single-word nouns found with current filters.")
        random_single_word_noun._pool = (pool, weights)
    pool, weights = random_single_word_noun._pool
    return random.choices(pool, weights=weights, k=1)[0]


def play_round(target: str, hint: str) -> Tuple[int, str]:
    sim = max_noun_wup(hint, target)
    bonus_sim, bonus_label = relation_bonus_as_similarity(hint, target)

    pts_base = int(round(BASE_SCALE * sim))
    pts_bonus = int(round(BASE_SCALE * bonus_sim))
    total_pts = pts_base + pts_bonus
    reason = bonus_label if bonus_label else "none"

    report = (
        f"TARGET='{target}'  |  your hint='{hint}'\n"
        f"WuP={sim:.2f}   bonus={bonus_sim:.2f} ({reason})\n"
        f"→ Points this round: {total_pts}  (Base={pts_base}  Bonus={pts_bonus})"
    )
    return total_pts, report


# ------------------------------ GUI ------------------------------
def launch_gui():
    import tkinter as tk
    from tkinter import ttk, messagebox

    root = tk.Tk()
    root.title("WordNet One-Word Game")

    score = 0
    round_idx = 1
    target = random_single_word_noun(min_count=3)

    info = ttk.Frame(root, padding=10)
    info.pack(side=tk.TOP, fill=tk.X)

    score_var = tk.StringVar(value=f"Score: {score}")
    round_var = tk.StringVar(value=f"Round: {round_idx}/{ROUNDS}")
    target_var = tk.StringVar(value=f"Target: {target}")

    ttk.Label(info, textvariable=score_var, font=("Segoe UI", 12, "bold")).pack(side=tk.LEFT, padx=(0, 16))
    ttk.Label(info, textvariable=round_var, font=("Segoe UI", 12)).pack(side=tk.LEFT, padx=(0, 16))
    ttk.Label(info, textvariable=target_var, font=("Segoe UI", 12)).pack(side=tk.LEFT)

    log_frame = ttk.LabelFrame(root, text="Details", padding=10)
    log_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
    log = tk.Text(log_frame, wrap="word", height=18, font=("Consolas", 10))
    log.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    log.insert("end", "Welcome! Type a related word and press Submit.\n")

    def append_log(text: str):
        log.insert("end", text + ("\n" if not text.endswith("\n") else ""))
        log.see("end")

    input_row = ttk.Frame(root, padding=10)
    input_row.pack(side=tk.BOTTOM, fill=tk.X)

    hint_var = tk.StringVar()
    hint_entry = ttk.Entry(input_row, textvariable=hint_var, width=40)
    hint_entry.pack(side=tk.LEFT, padx=(0, 10))
    hint_entry.focus()

    submit_btn = ttk.Button(input_row, text="Submit")
    submit_btn.pack(side=tk.LEFT)

    def refresh_labels():
        score_var.set(f"Score: {score}")
        round_var.set(f"Round: {round_idx}/{ROUNDS}")
        target_var.set(f"Target: {target}")

    def end_game():
        append_log("\nGAME OVER")
        append_log(f"Final score: {score} pts\n")
        submit_btn.config(state=tk.DISABLED)
        hint_entry.config(state=tk.DISABLED)
        messagebox.showinfo("Game Over", f"Final score: {score} pts")

    def on_submit():
        nonlocal score, round_idx, target
        hint = hint_var.get().strip().lower()
        if not hint:
            messagebox.showinfo("Hint required", "Please type a hint word.")
            return

        pts, report = play_round(target, hint)
        score += pts
        append_log(f"\n{report}\n")
        hint_var.set("")
        round_idx += 1

        if round_idx > ROUNDS:
            refresh_labels()
            end_game()
            return

        target = random_single_word_noun(min_count=3)
        refresh_labels()

    submit_btn.config(command=on_submit)
    root.bind("<Return>", lambda e: on_submit())

    refresh_labels()
    root.mainloop()


# ------------------------------ ------------------------------
def main_console():
    print("=== WordNet Random Single-Word Game ===")
    print("Each round you get one random single-word noun from WordNet.")
    print("Type a related word; you score based on Wu–Palmer (nouns) + relation bonuses.\n")

    score = 0
    for r in range(1, ROUNDS + 1):
        target = random_single_word_noun(min_count=3)
        print(f"\nRound {r}/{ROUNDS} — TARGET: {target}")
        hint = input("Your hint: ").strip().lower()
        if not hint:
            print("Empty hint → 0 pts.")
            continue
        pts, report = play_round(target, hint)
        score += pts
        print(report)
        print(f"Total score so far: {score}")

    print("\n=== GAME OVER ===")
    print(f"Final score: {score} pts")


# ------------------------------
if __name__ == "__main__":
    try:
        launch_gui()
    except Exception as e:
        print("GUI failed; falling back to console.")
        print("Reason:", e)
        main_console()
