import nltk
from nltk.corpus import brown
from nltk import bigrams, ConditionalFreqDist
import pickle
import pandas as pd

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

# Define a function to calculate the probability of a tag given the previous tag
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
    from nltk.tokenize import sent_tokenize
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
    return round(percentage,2)

# import json
# df = pd.read_csv('essays_dataset/index.csv', sep=';')
# temp = []
# for filename in df['filename']:
#     with open('essays_dataset/essays/' + filename, 'r') as file:
#         text = file.read()
#         temp.append(verb_mistakes(text))
#     # break

# # Save the list to a JSON file
# with open('verb_tense_mistakes.json', 'w') as f:
#     json.dump(temp, f)