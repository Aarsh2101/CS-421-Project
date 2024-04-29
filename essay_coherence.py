from cosines import cosine_similarity, sent_embedding
import spacy
import numpy as np
import json

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
        return 0
    for i in range(len(similarities)-1):
        if abs((similarities[i] - similarities[i+1])/similarities[i]) > threshold:
            changes += 1
            change_value += abs((similarities[i] - similarities[i+1])/similarities[i])

    return change_value, changes


def general_scorer_gaussian_assumption(x, mean, stddev, min_score, max_score, reverse=False):
    z_score = (x - mean) / stddev
    z_min, z_max = -3, 3

    score = (z_score - z_min) / (z_max - z_min) * (max_score - min_score) + min_score
    if reverse:
        score = max_score - score + min_score
    return np.clip(score, min_score, max_score)


def score_by_essay_incoherence(text, min_score, max_score):
    incoherence, changes = compute_coherence_changes(text, threshold=0.1)
    if changes <= 1:
        return max_score
    mean = np.mean(incoherence_in_essays)
    stddev = np.std(incoherence_in_essays)
    score = general_scorer_gaussian_assumption(incoherence, mean, stddev, min_score, max_score, reverse=True)
    return score
