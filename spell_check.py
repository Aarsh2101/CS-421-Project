import re
from spellchecker import SpellChecker
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

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

# import json
# import pandas as pd
# df = pd.read_csv('essays_dataset/index.csv', sep=';')
# temp = []
# for filename in df['filename']:
#     with open('essays_dataset/essays/' + filename, 'r') as file:
#         text = file.read()
#         temp.append(spell_check(text))
#     # break
# # print(temp)
# # Save the list to a JSON file
# with open('spelling_mistakes.json', 'w') as f:
#     json.dump(temp, f)