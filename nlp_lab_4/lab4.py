import nltk
import spacy
from nltk import CFG
from spacy import displacy
import stanza


def compute_tree(grammar, sentences):
    sentences_split = [s.lower().split() for s in sentences]

    parser = nltk.ChartParser(grammar)

    for sent in sentences_split:
        print("Sentence:", " ".join(sent))
        for tree in parser.parse(sent):
            print(tree)
            tree.pretty_print()
            tree.draw()

def dependency_spacy(grammar, sentences):
    print("----- SPACY -----")
    nlp = spacy.load("en_core_web_sm")

    for sent in sentences:
        doc = nlp(sent)
        print(f"\nSentence: {sent}")
        print("Word".ljust(15), " Head".ljust(15), "Relation")
        print("-" * 45)
        for token in doc:
            print(f"{token.text:<15}  {token.head.text:<15} {token.dep_}")

    docs = [nlp(s) for s in sentences]
    displacy.serve(docs, style="dep")


def dependency_stanza(grammar, sentences):
    print("------ STANZA -------")
    nlp = stanza.Pipeline('en')
    docs = [nlp(s) for s in sentences]

    for i, doc in enumerate(docs, start=1):
        print(f"\nSentence {i}: {sentences[i-1]}")
        print(f"{'Word':<15} {'Head':<15} {'Relation'}")
        print("-" * 45)
        for sent in doc.sentences:
            for word in sent.words:
                head = sent.words[word.head - 1].text if word.head > 0 else "ROOT"
                print(f"{word.text:<15} {head:<15} {word.deprel}")


if __name__ == "__main__":
    grammar = CFG.fromstring("""
    S -> NP VP
    NP -> Det N | NP Conj NP | NP PP | Adj N | V N | Adj N Comp
    VP -> Aux V | Aux V Adj | V NP Comp | V NP
    PP -> P NP
    Comp -> Adv P NP
    Conj -> 'and'
    Det -> 'the' 
    Adj -> 'flying' | 'dangerous'
    N -> 'parents' | 'groom' | 'bride' | 'planes'
    Adj -> 'flying' | 'dangerous'
    V -> 'flying' | 'be' | 'loves'
    Aux -> 'can' | 'were'
    Adv -> 'more'
    P -> 'of' | 'than'
    """)

    sentences = [
    "Flying planes can be dangerous",
    "The parents of the bride and the groom were flying",
    "The groom loves dangerous planes more than the bride"
    ]

    compute_tree(grammar, sentences)
    dependency_stanza(grammar, sentences)
    dependency_spacy(grammar, sentences)

