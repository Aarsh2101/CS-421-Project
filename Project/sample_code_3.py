import nltk
from nltk.chunk import RegexpParser
from nltk.corpus import brown
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from cosines import cosine_similarity, sent_embedding
import spacy
import numpy as np
import os
nltk.download('brown')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

def general_scorer_gaussian_assumption(x, mean, stddev, min_score, max_score, reverse=False):
    if stddev == 0:
        return (min_score + max_score) / 2
    z_score = (x - mean) / stddev
    z_min, z_max = -3, 3
    score = (z_score - z_min) / (z_max - z_min) * (max_score - min_score) + min_score
    
    if reverse:
        score = max_score - score + min_score
    return np.clip(score, min_score, max_score)

# ----- DOES THE ESSAY ADDRESS THE TOPIC? ----- #
def get_topical_part(prompt):
    sentences = nltk.sent_tokenize(prompt)
    if len(sentences) < 3:
        return prompt  
    topic = sentences[1]
    return topic

def get_similarity_score(text1, text2):
    text1 = ' '.join([word for word in text1.split() if word.lower() not in nltk.corpus.stopwords.words('english')])
    text2 = ' '.join([word for word in text2.split() if word.lower() not in nltk.corpus.stopwords.words('english')])
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform([text1, text2])
    similarity = (tfidf_matrix * tfidf_matrix.T)[0,1]

    # Assign a score based on the similarity
    if similarity < 0.1:
        score = 1
    elif similarity < 0.2:
        score = 2
    elif similarity < 0.4:
        score = 3
    elif similarity < 0.6:
        score = 4
    else:
        score = 5

    return score

def address_topic(prompt, essay):
    topic = get_topical_part(prompt)
    score = get_similarity_score(topic, essay)
    return score

# ----- SYNTACTIC WELL-FORMEDNESS ----- #

def traverse_tree(tree):
    for subtree in tree:
        if type(subtree) == nltk.tree.Tree:
            parent = subtree.label()
            if parent not in parent_child_grandchild:
                parent_child_grandchild[parent] = dict()
            for child in subtree:
                if type(child) == nltk.tree.Tree:
                    if child.label() not in parent_child_grandchild[parent]:
                        parent_child_grandchild[parent][child.label()] = list()
                    for grandchild in child:
                        if type(grandchild) == nltk.tree.Tree:
                            if grandchild.label() not in parent_child_grandchild[parent][child.label()]:
                                parent_child_grandchild[parent][child.label()].append(grandchild.label())
                            # parent_child_grandchild[parent][child.label()].add(grandchild.label())
                        else:
                            if grandchild[1] not in parent_child_grandchild[parent][child.label()]:
                                parent_child_grandchild[parent][child.label()].append(grandchild[1])
                            # parent_child_grandchild[parent][child.label()].add(grandchild[1])
                else:
                    child = child[1]
                    parent_child_grandchild[parent][child] = list()
            traverse_tree(subtree)

TRAINING = False

grammar = r"""
    NP: {<DT>?<JJ.*>*<NN.*>+|<NNPS>+}        # Noun Phrase: optional determiner, any number of adjectives, singular or plural nouns (including proper nouns)
        {<PRP>|<PRP$>}                       # Personal pronouns or possessive pronouns
        {<NP><POS><NN.*>+|<NNP>+|<NNPS>+}    # Noun phrases that use possessive endings or include proper nouns (singular/plural)
        {<EX><VB.*>}                         # Existential there constructions, e.g., "There is" or "There are"
        {<NNP>+|<NNPS>+}                     # Proper noun sequences, potentially forming proper noun phrases (singular/plural)
    VP: {<MD>?<VB.*>+<RB.*>*}                # Verb Phrase: optional modal, one or more verbs (including past tense), any number of adverbs
        {<TO><VB>}                           # Infinitive verbs
        {<VBD><RB.*>*}                       # Past tense verb followed by any number of adverbs
        {<VBG><RB.*>*}                       # Present participle verb followed by any number of adverbs
        {<VBP><RB.*>*}                       # Non-3rd person singular present verb followed by any number of adverbs
        {<VBN><RB.*>*}                       # Past participle verb followed by any number of adverbs
        {<VBZ><RB.*>*}                       # 3rd person singular present verb followed by any number of adverbs
    ADJP: {<RB.*>*<JJ.*>+}                   # Adjective Phrase: any number of adverbs followed by one or more adjectives
        {<JJR>|<JJS>}                        # Comparative and Superlative adjectives directly
    ADVP: {<RB.*>+}                          # Adverb Phrase: one or more adverbs
        {<RBR>|<RBS>}                        # Comparative and superlative adverbs
    PP: {<IN><NP>}                           # Prepositional Phrase: preposition followed by a noun phrase
    CONJP: {<CC>}                            # Conjunction Phrase: coordinating conjunction
    INTJ: {<UH>}                             # Interjection
    DP: {<DT>|<PDT>|<WDT>|<EX>}              # Determiner Phrase: determiners, pre-determiners, wh-determiners, or existential "there"
    QP: {<CD><NNS|NN>?}                      # Quantifier Phrase: cardinal number followed optionally by plural or singular noun
    WHNP: {<WP>|<WP$>|<WRB>|<WDT>}           # WH Noun Phrase: wh-pronoun, possessive wh-pronoun, wh-adverb, or wh-determiner
    SYMP: {<SYM>}                            # Symbol Phrase: handling symbols
    CD-NP: {<CD><JJ.*>*<NN.*>+}              # Number followed by adjectives and nouns
    PAST-VP: {<VBD><RB.*>*}                  # Past tense verb followed by any number of adverbs
    PRES-VP: {<VBG><RB.*>*}                  # Present participle verb followed by any number of adverbs
    NON3RD-VP: {<VBP><RB.*>*}                # Non-3rd person singular present verb followed by any number of adverbs
    PERFECT-VP: {<VBN><RB.*>*}               # Past participle verb followed by any number of adverbs
    THIRD-PERSON-VP: {<VBZ><RB.*>*}          # 3rd person singular present verb followed by any number of adverbs
    LIST-NP: {<LS><NP>+}                     # List item markers followed by Noun Phrases, e.g., in enumerated lists
"""
cp = nltk.RegexpParser(grammar)
parent_child_grandchild = {}
sentences = brown.sents()

if TRAINING:
    for sentence in tqdm(sentences):
        tagged_tokens = nltk.pos_tag(sentence)
        tree = cp.parse(tagged_tokens)
        traverse_tree(tree)

    with open('parent_child.json', 'w') as f:
        json.dump(parent_child_grandchild, f)

with open('parent_child.json') as f:
    checker_dict = json.load(f)

def check_sentence_syntax(sentence, checker_dict):
    mistakes = 0
    cp = nltk.RegexpParser(grammar)
    # Get tokens from NLTK
    tagged_tokens = nltk.pos_tag(sentence)
    tree = cp.parse(tagged_tokens)
    for subtree in tree:
        if type(subtree) == nltk.tree.Tree:
            parent = subtree.label()
            for child in subtree:
                if type(child) == nltk.tree.Tree:
                    child_label = child.label()
                    for subsubtree in child:
                        if type(subsubtree) == nltk.tree.Tree:
                            subsubtree_label = subsubtree.label()     
                        else:
                            subsubtree_label = subsubtree[1]   
                            if parent in checker_dict.keys():
                                if child_label not in checker_dict[parent].keys():
                                    mistakes += 1 
                                    continue
                                else:
                                    if subsubtree_label not in checker_dict[parent][child_label]:
                                        mistakes += 1
                                        continue
                            else:
                                mistakes += 1
                                continue
    return mistakes


mistakes_per_essay = []
for filename in os.listdir(os.path.join("essays")):
    with open(os.path.join("essays", filename)) as file:
        text = file.read()
        sentences = nltk.sent_tokenize(text)
        mistakes = 0
        for sentence in sentences:
            tokens = nltk.word_tokenize(sentence)
            mistakes += check_sentence_syntax(tokens, checker_dict)
        mistakes_per_essay.append(mistakes)


def score_essay_syntax(text, min_score, max_score):
    sentences = nltk.sent_tokenize(text)
    mistakes = 0
    for sentence in sentences:
        tokens = nltk.word_tokenize(sentence)
        mistakes += check_sentence_syntax(tokens, checker_dict)
    mean = np.mean(mistakes_per_essay)
    stddev = np.std(mistakes_per_essay)
    score = general_scorer_gaussian_assumption(mistakes, mean, stddev, min_score, max_score, reverse=True)
    return score


# ----- ESSAY COHERENCE ----- #

nlp = spacy.load("en_core_web_sm")

incoherence_in_essays = []
# Load json file
with open('incoherence_in_essays.json', 'r') as f:
    incoherence_in_essays = json.load(f)
    f.close()


def get_all_sent_embeddings(essay):
    doc = nlp(essay)
    embeddings = []
    for sent in doc.sents:
        embeddings.append(sent_embedding(sent.text))
    return embeddings


def get_pairwise_cosine_similarities(essay):
    embeddings = get_all_sent_embeddings(essay)
    similarities = []
    for i in range(len(embeddings)-1):
        sim = cosine_similarity(embeddings[i], embeddings[i+1])
        similarities.append(sim)
    return similarities


def compute_coherence_changes(essay, threshold):
    similarities = get_pairwise_cosine_similarities(essay)
    changes = 0
    change_value = 0
    if len(similarities) == 0:
        return 0, 0
    for i in range(len(similarities)-1):
        if abs((similarities[i] - similarities[i+1])/similarities[i]) > threshold:
            changes += 1
            change_value += abs((similarities[i] - similarities[i+1])/similarities[i])
    return round(change_value, 4), round(changes, 4)


def score_by_essay_incoherence(text, min_score, max_score):
    incoherence, changes = compute_coherence_changes(text, threshold=0.1)
    if changes <= 1:
        return max_score
    mean = np.mean(incoherence_in_essays)
    stddev = np.std(incoherence_in_essays)
    score = general_scorer_gaussian_assumption(incoherence, mean, stddev, min_score, max_score, reverse=True)
    return score
