grammar = """
S: {<NP><VP>}
S: {<Aux><NP><VP>}
S: {<VP>}
NP: {<DT><NN>}
NP: {<DT><JJ><NN>}
NP: {<NP><PP>}
NP: {<PRP>}
VP: {<V><NP>}
VP: {<V>}
VP: {<V><NP><PP>}
VP: {<V><VP>}
VP: {<MD><VP}>
PP: {<IN><NP>}
ADJP: {<JJ>}
ADJP: {<JJ><ADJP>}
ADJP: {<RB><JJ>}
ADVP: {<RB>}
ADVP: {<RB><ADVP>}
NP: {<NP><CC><NP>}
VP: {<VP><CC><VP>}
"""
