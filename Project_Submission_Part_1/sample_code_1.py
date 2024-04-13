import spacy
import os
import numpy as np
import json
import re
from spellchecker import SpellChecker
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer


# ----- GENERAL GAUSSIAN SCORER ----- #
def general_scorer_gaussian_assumption(x, mean, stddev, min_score, max_score, reverse=False):
    z_score = (x - mean) / stddev
    z_min, z_max = -3, 3

    score = (z_score - z_min) / (z_max - z_min) * (max_score - min_score) + min_score
    if reverse:
        score = max_score - score + min_score
    return np.clip(score, min_score, max_score)


# ----- SENTENCE COUNTING ----- #
nlp = spacy.load("en_core_web_sm")

sentence_counts_list = []
# Load json file
with open('sentence_counts.json', 'r') as f:
    sentence_counts_list = json.load(f)
    f.close()

def count_sentences_with_spacy(text):
    doc = nlp(text)
    sentences = list(doc.sents)
    sentence_count = len(sentences)

    return sentence_count


def num_sentences(text, sentence_counts, min_score, max_score):
    num_sentences = count_sentences_with_spacy(text)
    if num_sentences <= 10:
        return min_score
    else:
        sentence_counts = np.array(sentence_counts)
        sentence_counts = sentence_counts[sentence_counts > 10]
        mean = np.mean(sentence_counts)
        stddev = np.std(sentence_counts)
        score = general_scorer_gaussian_assumption(num_sentences, mean, stddev, min_score, max_score)
        return score


# ----- SPELLING MISTAKES ----- #
spelling_mistakes_list = []
with open('spelling_mistakes.json', 'r') as f:
    spelling_mistakes_list = json.load(f)
    f.close()

def decontracted(phrase):
    # specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase


def spell_check(text):
    spell = SpellChecker()
    lemmatizer = WordNetLemmatizer()
    text = decontracted(text)
    words = word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in words]
    misspelled_words = spell.unknown(words)
    percantage = len(misspelled_words) / len(words)
    return round(percantage, 2)


def spelling_mistakes(text, mistakes_list, min_score, max_score):
    mistakes = spell_check(text)
    mean = np.mean(mistakes_list)
    stddev = np.std(mistakes_list)
    score = general_scorer_gaussian_assumption(mistakes, mean, stddev, min_score, max_score)
    return score
