import os, random, sys
from collections import defaultdict
from typing import Dict, Set, List, Tuple
from nltk.corpus import wordnet as wn
import nltk

# ------------------------------
BOARD_SIZE = 16
CLEAR_ZONE = 4
ROUNDS = 10
BASE_SCALE = 100

BONUS = {
    "synonyms": 0.20,
    "hypernyms": 0.12,
    "hyponyms": 0.12,
    "meronyms": 0.08,
    "antonyms": 0.06,
}

try:
    wn.ensure_loaded()
except LookupError:
    nltk.download("wordnet")
    nltk.download("omw-1.4")

# ---------------------------
WORD_BANK = [
    "dog", "cat", "lion", "tiger", "wolf", "fox", "fish", "bird", "hamster", "horse", "cow", "sheep",
    "river", "lake", "ocean", "cloud", "rain", "snow", "sun", "mountain", "forest", "tree", "flower", "leaf", "seed",
    "fruit",
    "car", "bus", "bicycle", "train", "plane", "boat", "ship", "motorcycle",
    "house", "apartment", "hotel", "museum", "school", "farm", "church", "hospital", "bank", "park",
    "book", "newspaper", "pencil", "pen", "table", "chair", "bed", "lamp", "bottle", "cup", "clock", "mirror",
    "phone", "computer", "keyboard", "guitar", "piano", "song", "radio", "television",
    "shoe", "coat", "hat", "shirt", "dress",
    "neighbor", "friend", "teacher", "doctor", "farmer", "pilot", "student",
]


# ------------------------------
def _lemma_str(n: str) -> str:
    return n.replace("_", " ").lower()


def _lemmas_for_synset(syn) -> Set[str]:
    return {_lemma_str(l.name()) for l in syn.lemmas()}


def noun_synsets(w: str):
    return [s for s in wn.synsets(w) if s.pos() == "n"]


def max_noun_wup(a: str, b: str) -> float:
    """Best Wu–Palmer across noun senses (0..1)."""
    best = 0.0
    s1, s2 = noun_synsets(a), noun_synsets(b)
    for x in s1:
        for y in s2:
            sim = x.wup_similarity(y)
            if sim is not None and sim > best:
                best = float(sim)
    return best


def gather_relations(w: str) -> Dict[str, Set[str]]:
    rel: Dict[str, Set[str]] = defaultdict(set)
    for syn in wn.synsets(w):
        if syn.pos() != "n":
            continue
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


# --------------------------------
def init_board_and_bag():
    bag = WORD_BANK[:]
    random.shuffle(bag)
    board, recycle = [], []
    while bag and len(board) < BOARD_SIZE:
        board.append(bag.pop())
    return board, bag, recycle


def refill_board(board: List[str], bag: List[str], recycle: List[str]):
    while len(board) < BOARD_SIZE:
        if not bag:
            random.shuffle(recycle)
            bag[:] = recycle
            recycle.clear()
            if not bag:
                return
        while bag and bag[-1] in board:
            random.shuffle(bag)
        if not bag:
            return
        board.append(bag.pop())


# --------------------------------
def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def draw_board_console(board: List[str], score: int, target_word: str):
    clear_screen()
    print(f"Score: {score}\n")
    cut = BOARD_SIZE - CLEAR_ZONE
    for i, w in enumerate(board):
        marker = "▶ " if w == target_word else "  "
        print(f"{marker}{w}")
        if i == cut - 1:
            print("  " + "-" * 26 + "  ⤵ cleared below")
    print("\nType a hint. NOTHING clears unless the TARGET lands below the line.\n")


# ----------------------------
def game_step(board: List[str], bag: List[str], recycle: List[str],
              target_word: str, score: int, hint: str) -> Tuple[List[str], List[str], List[str], str, int, str]:
    scored_all = []
    for w in board:
        sim = max_noun_wup(hint, w)
        bonus_sim, bonus_label = relation_bonus_as_similarity(hint, w)
        total = min(1.0, sim + bonus_sim)
        scored_all.append((w, sim, bonus_sim, bonus_label, total))

    scored_all.sort(key=lambda t: t[4])
    new_board = [t[0] for t in scored_all]

    cut = BOARD_SIZE - CLEAR_ZONE
    new_target_idx = new_board.index(target_word)

    report_lines = []
    if new_target_idx >= cut:
        cleared_slice = scored_all[-CLEAR_ZONE:]
        cleared_words = [w for (w, *_) in cleared_slice]

        gained_total = gained_base = gained_bonus = 0
        report_lines.append(f"\n🔥 TARGET '{target_word}' FELL BELOW THE LINE! Words cleared:\n")
        report_lines.append(f"{'Word':<12} {'WuP':<6} {'BonusType':<12} {'Bonus':<6} {'Pts':<6}")
        report_lines.append("-" * 50)

        for (_w, sim, bonus_sim, bonus_label, _tot) in cleared_slice:
            pts_base = int(round(BASE_SCALE * sim))
            pts_bonus = int(round(BASE_SCALE * bonus_sim))
            gained_total += pts_base + pts_bonus
            gained_base += pts_base
            gained_bonus += pts_bonus
            label = bonus_label if bonus_label else "-"
            report_lines.append(f"{_w:<12} {sim:<6.2f} {label:<12} {bonus_sim:<6.2f} {pts_base + pts_bonus:<6}")

        score += gained_total
        survivors = [w for w in new_board if w not in cleared_words]
        board = survivors
        recycle.extend(cleared_words)
        refill_board(board, bag, recycle)
        target_word = board[0] if board else ""

        report_lines.append("-" * 50)
        report_lines.append(f"🏆 Points gained: {gained_total}  (Base={gained_base}  Bonus={gained_bonus})")
        report_lines.append(f"💯 Total Score: {score}\n")

    else:
        board = new_board
        report_lines.append(f"\nTARGET stayed above line at position {new_target_idx + 1}/{BOARD_SIZE}. No clear.")
        for (w, sim, bonus_sim, bonus_label, _tot) in scored_all:
            if w == target_word:
                reason = bonus_label if bonus_label else "none"
                report_lines.append(f"🔎 Target '{w}': WuP={sim:.2f}, bonus={bonus_sim:.2f} ({reason})")
                break
        report_lines.append(f"💯 Total Score: {score}\n")

    return board, bag, recycle, target_word, score, "\n".join(report_lines)


# ----------------------------
def launch_gui():
    import tkinter as tk
    from tkinter import ttk, messagebox

    board, bag, recycle = init_board_and_bag()
    score = 0
    target_word = board[0]
    turn = 1

    root = tk.Tk()
    root.title("Semantris WordNet — NLTK")

    info_frame = ttk.Frame(root, padding=10)
    info_frame.pack(side=tk.TOP, fill=tk.X)

    score_var = tk.StringVar(value=f"Score: {score}")
    turn_var = tk.StringVar(value=f"Turn: {turn}/{ROUNDS}")
    target_var = tk.StringVar(value=f"Target: {target_word}")

    ttk.Label(info_frame, textvariable=score_var, font=("Segoe UI", 12, "bold")).pack(side=tk.LEFT, padx=(0, 15))
    ttk.Label(info_frame, textvariable=turn_var, font=("Segoe UI", 12)).pack(side=tk.LEFT, padx=(0, 15))
    ttk.Label(info_frame, textvariable=target_var, font=("Segoe UI", 12)).pack(side=tk.LEFT)

    main = ttk.Frame(root, padding=(10, 0, 10, 10))
    main.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    board_frame = ttk.LabelFrame(main, text="Board", padding=10)
    board_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)

    lb = tk.Listbox(board_frame, width=24, height=BOARD_SIZE, font=("Consolas", 12))
    lb.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)
    sb = ttk.Scrollbar(board_frame, orient="vertical", command=lb.yview)
    sb.pack(side=tk.LEFT, fill=tk.Y)
    lb.config(yscrollcommand=sb.set)

    cz_label = ttk.Label(board_frame, text=f"⤵ Clear zone: bottom {CLEAR_ZONE}", foreground="#666")
    cz_label.pack(side=tk.TOP, anchor="w", pady=(6, 0))

    log_frame = ttk.LabelFrame(main, text="Round details (same as console output)", padding=10)
    log_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
    log = tk.Text(log_frame, wrap="word", height=24, font=("Consolas", 10))
    log.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    log.insert("end", "Welcome! Type a hint and press Submit.\n")

    input_frame = ttk.Frame(root, padding=10)
    input_frame.pack(side=tk.BOTTOM, fill=tk.X)

    hint_var = tk.StringVar()
    hint_entry = ttk.Entry(input_frame, textvariable=hint_var, width=40)
    hint_entry.pack(side=tk.LEFT, padx=(0, 10))
    hint_entry.focus()

    submit_btn = ttk.Button(input_frame, text="Submit", command=lambda: on_submit())
    submit_btn.pack(side=tk.LEFT)

    def refresh_board():
        lb.delete(0, tk.END)
        cut = BOARD_SIZE - CLEAR_ZONE
        for i, w in enumerate(board):
            marker = "▶ " if w == target_word else "  "
            lb.insert(tk.END, f"{marker}{w}")
            if i >= cut:
                lb.itemconfig(tk.END, bg="#ffe6e6")
        score_var.set(f"Score: {score}")
        turn_var.set(f"Turn: {turn}/{ROUNDS}")
        target_var.set(f"Target: {target_word}")

    def append_log(text: str):
        log.insert("end", text + ("\n" if not text.endswith("\n") else ""))
        log.see("end")

    refresh_board()

    def draw_console_snapshot():
        draw_board_console(board, score, target_word)

    def on_submit():
        nonlocal board, bag, recycle, target_word, score, turn
        hint = hint_var.get().strip().lower()
        if not hint:
            messagebox.showinfo("Hint required", "Please type a hint word.")
            return

        draw_console_snapshot()
        print(f"Turn {turn}/{ROUNDS} — TARGET='{target_word}' — your hint: {hint}")

        board, bag, recycle, target_word, score, report = game_step(
            board, bag, recycle, target_word, score, hint
        )

        print(report)
        append_log(report)

        turn += 1
        hint_var.set("")
        refresh_board()

        if turn > ROUNDS:
            draw_console_snapshot()
            print("GAME OVER")
            print(f"Final score: {score} pts")
            append_log("\nGAME OVER\n")
            append_log(f"Final score: {score} pts\n")
            submit_btn.config(state=tk.DISABLED)
            hint_entry.config(state=tk.DISABLED)

    root.mainloop()


# -------------------------------
if __name__ == "__main__":
    try:
        launch_gui()
    except Exception as e:
        print("GUI failed to start; falling back to console mode.")
        print("Reason:", e)
        board, bag, recycle = init_board_and_bag()
        score = 0
        target_word = board[0]
        for turn in range(1, ROUNDS + 1):
            draw_board_console(board, score, target_word)
            hint = input(f"Turn {turn}/{ROUNDS} — TARGET='{target_word}' — your hint: ").strip().lower()
            if not hint:
                continue
            board, bag, recycle, target_word, score, report = game_step(
                board, bag, recycle, target_word, score, hint
            )
            print(report)
            input("Press Enter for next turn...")
        draw_board_console(board, score, target_word)
        print("GAME OVER")
        print(f"Final score: {score} pts")
