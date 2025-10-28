import nltk


def transform_grammar(grammar):
    productions = []


    modified = False
    for p in grammar.productions():
        lhs, rhs = p.lhs(), p.rhs()
        prob = p.prob()

        if len(rhs) == 1 and isinstance(rhs[0], str):
            productions.append(p)
        elif len(rhs) == 1 and isinstance(rhs[0], nltk.Nonterminal):
            target = rhs[0]
            for q in grammar.productions(lhs=target):
                new_prob = prob * q.prob()
                productions.append(nltk.grammar.ProbabilisticProduction(lhs, q.rhs(), prob=new_prob))
                modified = True
        elif len(rhs) == 2 and all(isinstance(r, nltk.grammar.Nonterminal) for r in rhs):
            productions.append(p)
        elif len(rhs) > 2 and all(isinstance(x, nltk.Nonterminal) for x in rhs):
            new_productions = []
            current_lhs = lhs
            current_prob = prob
            symbols = list(rhs)
            while len(symbols) > 2:
                new_symbol = nltk.grammar.Nonterminal(f"New_{current_lhs}_{'_'.join(str(s) for s in symbols)}")
                new_productions.append(
                    nltk.grammar.ProbabilisticProduction(current_lhs, [symbols[0], new_symbol], prob=current_prob)
                )
                current_lhs = new_symbol
                symbols = symbols[1:]
                current_prob = 1.0
            new_productions.append(nltk.grammar.ProbabilisticProduction(current_lhs, symbols, prob=1.0))
            productions.extend(new_productions)
        elif len(rhs) >= 2:
            new_rhs = []
            for i, sym in enumerate(rhs):
                if isinstance(sym, str):
                    term_symbol = nltk.Nonterminal(f"T_{sym}_{i}")
                    productions.append(
                        nltk.grammar.ProbabilisticProduction(term_symbol, [sym], prob=1.0)
                    )
                    new_rhs.append(term_symbol)
                else:
                    new_rhs.append(sym)

            if len(new_rhs) == 2:
                productions.append(
                    nltk.grammar.ProbabilisticProduction(lhs, new_rhs, prob=prob)
                )
            else:
                new_productions = []
                current_lhs = lhs
                current_prob = prob
                symbols = list(rhs)
                while len(symbols) > 2:
                    new_symbol = nltk.grammar.Nonterminal(f"New_{current_lhs}_{'_'.join(str(s) for s in symbols)}")
                    new_productions.append(
                        nltk.grammar.ProbabilisticProduction(current_lhs, [symbols[0], new_symbol], prob=current_prob)
                    )
                    current_lhs = new_symbol
                    symbols = symbols[1:]
                    current_prob = 1.0
                new_productions.append(nltk.grammar.ProbabilisticProduction(current_lhs, symbols, prob=1.0))
                productions.extend(new_productions)
        else:
            raise ValueError(f"Unhandled rule: {p}")

    return productions, modified


def cnf(grammar):
    print()
    print("Grammar: ")
    for p in grammar.productions():
        print(p)

    productions, modified = transform_grammar(grammar)
    while modified:
        cnf = nltk.PCFG(grammar.start(), productions)
        productions, modified = transform_grammar(cnf)

    print()
    print("CNF grammar:")
    cnf = nltk.PCFG(grammar.start(), productions)
    for p in cnf.productions():
        print(p)

    grammar_noprob = nltk.grammar.CFG(grammar.start(), [nltk.Production(p.lhs(), p.rhs()) for p in grammar.productions()])
    try:
        grammar_noprob = grammar_noprob.chomsky_normal_form(flexible=False)
        print()
        print("CNF grammar no prob:")
        for p in grammar_noprob.productions():
            print(p)
    except:
        print("Could not determine CNF grammar.")





grammar = nltk.grammar.PCFG.fromstring("""
S -> NP VP [1.0]
NP -> Det Adj N [1.0]
Det -> 'the' [1.0]
Adj -> 'big' [1.0]
N -> 'dog' [1.0]
VP -> V [1.0]
V -> 'barks' [1.0]
""")

grammar1 = nltk.grammar.PCFG.fromstring("""
S -> 'the' N Adj [0.7]
S -> V 'dog' [0.3]
N -> 'cat' [1.0]
V -> 'chased' [1.0]
Adj -> 'fast' [1.0]
""")

grammar2 = nltk.grammar.PCFG.fromstring("""
S -> A B C D [1.0]
A -> 'x' [1.0]
B -> 'y' [1.0]
C -> 'z' [1.0]
D -> 'w' [1.0]
""")

grammar3 = nltk.grammar.PCFG.fromstring("""
S -> NP VP [1.0]

NP -> Det N [0.6] 
NP -> Det N PP [0.4]

VP -> V NP [0.7]
VP -> V NP PP [0.3]

PP -> P NP [1.0]

Det -> 'the' [1.0]
N -> 'dog' [0.3] | 'cat' [0.3] | 'telescope' [0.4]
V -> 'saw' [0.6] | 'liked' [0.4]
P -> 'with' [0.7] | 'by' [0.3]
""")

grammar4 = nltk.grammar.PCFG.fromstring("""
S -> VP [1.0]
VP -> V [0.8]
VP -> V NP [0.2]
V -> 'runs' [0.4]
V -> 'flies' [0.6]
NP -> 'bird' [1.0]
""")



grammar5 = nltk.grammar.PCFG.fromstring("""
S -> VP [1.0]
VP -> V [0.8]
VP -> V NP [0.2]
V -> NP Adj Adv [0.4]
V -> 'flies' [0.6]
NP -> 'bird' [1.0]
Adj -> 'dog' [1.0]
Adv -> 'saw' [0.6] | 'liked' [0.4]
""")


cnf(grammar)
cnf(grammar1)
cnf(grammar2)
cnf(grammar3)
cnf(grammar4)
cnf(grammar5)





