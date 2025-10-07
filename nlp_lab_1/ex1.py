from collections import defaultdict
import re

from nltk.corpus import wordnet as wn
import nltk

try:
    wn.ensure_loaded()
except LookupError:
    nltk.download("wordnet")


def lemmas_for_synset(synset):
    names = set()
    for lemma in synset.lemmas():
        name = lemma.name()
        name = name.replace("_", " ")
        name = name.lower()
        names.add(name)
    return names


def collect_related_words(word: str):
    related = defaultdict(set)

    synsets = wn.synsets(word)

    for syn in synsets:

        for name in lemmas_for_synset(syn):
            related["synonyms"].add(name)

        for lemma in syn.lemmas():
            for ant in lemma.antonyms():
                ant_name = ant.name().replace("_", " ").lower()
                related["antonyms"].add(ant_name)

        for hyper in syn.hypernyms():
            for name in lemmas_for_synset(hyper):
                related["hypernyms"].add(name)

        for hypo in syn.hyponyms():
            for name in lemmas_for_synset(hypo):
                related["hyponyms"].add(name)

        part_meros = syn.part_meronyms()
        subst_meros = syn.substance_meronyms()
        member_meros = syn.member_meronyms()
        all_meros = part_meros + subst_meros + member_meros

        for mero in all_meros:
            for name in lemmas_for_synset(mero):
                related["meronyms"].add(name)

    surface = word.lower()
    for key in related:
        if surface in related[key]:
            related[key].remove(surface)

    return synsets, related


def print_summary(word: str, synsets, related):
    print("\n=== WORDNET SUMMARY for:", word, "===")
    print("Languages available (lemma metadata):", wn.langs())
    print("\nAll synsets (senses) for this word:")
    print(synsets)
    print("-" * 60)

    def preview(label, key, limit=10):
        values = sorted(related.get(key, []))
        if values:
            print(f"{label}: " + ", ".join(values[:limit]))

    preview("Synonyms", "synonyms")
    preview("Antonyms", "antonyms")
    preview("Hypernyms (more general)", "hypernyms")
    preview("Hyponyms (more specific)", "hyponyms")
    preview("Meronyms (parts/members/substances)", "meronyms")


def print_per_sense_details(synsets):
    def synset_names(lst):
        return sorted(s.name() for s in lst)

    for syn in synsets:
        print("\n" + "*" * 20)
        print("synset name:", syn.name())
        print("synset def :", syn.definition())

        print("hypernyms:", synset_names(syn.hypernyms()))
        print("hyponyms :", synset_names(syn.hyponyms()))

        antonym_syn_names = set()
        for lemma in syn.lemmas():
            for ant in lemma.antonyms():
                antonym_syn_names.add(ant.synset().name())
        print("antonyms :", sorted(antonym_syn_names))

        mero_syns = syn.part_meronyms() + syn.substance_meronyms() + syn.member_meronyms()
        print("meronyms :", synset_names(mero_syns))


if __name__ == "__main__":
    word = input("Enter a word: ").strip() or "white"
    synsets, related = collect_related_words(word)

    if not synsets:
        print(f"No WordNet synsets found for '{word}'. Try another word.")
    else:
        print_summary(word, synsets, related)
        print_per_sense_details(synsets)
