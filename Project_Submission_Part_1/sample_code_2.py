import spacy
import os
import numpy as np
import json
import nltk
from nltk.corpus import brown
from nltk import bigrams, ConditionalFreqDist
import pickle
import pandas as pd
from nltk.tokenize import sent_tokenize


# ----- GENERAL GAUSSIAN SCORER ----- #
def general_scorer_gaussian_assumption(x, mean, stddev, min_score, max_score, reverse=False):
    z_score = (x - mean) / stddev
    z_min, z_max = -3, 3

    score = (z_score - z_min) / (z_max - z_min) * (max_score - min_score) + min_score
    if reverse:
        score = max_score - score + min_score
    return np.clip(score, min_score, max_score)


# ----- SUBJECT VERB AGREEMENT ----- #
nlp = spacy.load("en_core_web_sm")

subject_verb_disagreements = {
    "NN" : ["VBP"],
    "NNP" : ["VBP"],
    "PRP" : ["VBP", "VBZ"],
    "NNS" : ["VBZ"],
    "NNPS" : ["VBZ"]
}

sub_verb_disagreements_list = []
with open('subject_verb_disagreements.json', 'r') as f:
    sub_verb_disagreements_list = json.load(f)
    f.close()

def subject_verb_disagree(subject, verb):
    if subject.tag_ in subject_verb_disagreements.keys():
        if verb.tag_ in subject_verb_disagreements[subject.tag_]:
            return True
    return False


def count_subject_verb_errors_fraction(text):
    doc = nlp(text)
    errors = 0
    # Get number of tokens in the text
    num_tokens = len([token for token in doc])
    for sent in doc.sents:
        for token in sent:
            if "VB" in token.tag_:
                subject = None
                for child in token.children:
                    if child.dep_ in ("nsubj", "nsubjpass"):
                        subject = child
                        break
                if subject and subject_verb_disagree(subject, token):
                    errors += 1
    return errors / num_tokens


def agreement(text, min_score, max_score):
    num_disagreements = count_subject_verb_errors_fraction(text)
    if num_disagreements < 2:
        return max_score
    mean = np.mean(sub_verb_disagreements_list)
    stddev = np.std(sub_verb_disagreements_list)
    score = general_scorer_gaussian_assumption(num_disagreements, mean, stddev, min_score, max_score, reverse=True)
    return score


# ----- VERB-TENSE DISAGREEMENTS ----- #
verb_tense_mistakes_list = []
with open('verb_tense_mistakes.json', 'r') as f:
    verb_tense_mistakes_list = json.load(f)
    f.close()

TRAINING = False

if TRAINING:
    sentences = brown.sents()

    tagged_sentences = [nltk.pos_tag(sentence) for sentence in sentences]

    tag_sequences = [[tag for word, tag in sentence] for sentence in tagged_sentences]
    tag_bigrams = [bigram for sent in tag_sequences for bigram in bigrams(sent)]
    cfd = ConditionalFreqDist(tag_bigrams)

    # Save the cfd object to a file
    with open('cfd.pkl', 'wb') as f:
        pickle.dump(cfd, f)

def tag_probability(prev_tag, current_tag):
    # Load the cfd object from the file
    with open('cfd.pkl', 'rb') as f:
        cfd = pickle.load(f)
        f.close()

    # Count of current tag following prev_tag
    current_tag_count = cfd[prev_tag][current_tag]
    # Total count of all tags following prev_tag
    total_count = sum(cfd[prev_tag].values())
    probability = current_tag_count / total_count if total_count > 0 else 0
    return probability


def verb_mistakes(text):
    sentences = sent_tokenize(text)
    mistakes = 0
    for sentence in sentences:
        tokens = nltk.word_tokenize(sentence)
        tagged = nltk.pos_tag(tokens)
        for i in range(1, len(tagged)):
            prev_tag = tagged[i-1][1]
            current_tag = tagged[i][1]
            if (prev_tag[0] == 'V' or current_tag[0] == 'V'):
                probability = tag_probability(prev_tag, current_tag)
                if probability < 0.05:
                    mistakes += 1
        percentage = mistakes / len(tagged)
    return round(percentage, 2)


def verbs(text, min_score, max_score):
    mistakes = verb_mistakes(text)
    mean = np.mean(verb_tense_mistakes_list)
    stddev = np.std(verb_tense_mistakes_list)
    score = general_scorer_gaussian_assumption(mistakes, mean, stddev, min_score, max_score, reverse=True)
    return score
