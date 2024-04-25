import nltk
import spacy
import numpy as np

# nltk.download('punkt')

# Load SpaCy's English model with word vectors
nlp = spacy.load('en_core_web_sm')

def tokenize_and_vectorize(inp_str):
    # Tokenize the text
    tokens = nltk.word_tokenize(inp_str)

    # Get the vector representation of each word
    word_vectors = [nlp(word).vector for word in tokens]

    return word_vectors



def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    sim = 0.0

    # Copy your cosine_similarity code here
    numerator = np.dot(a, b)
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if numerator == 0:
        return 0
    sim = numerator / denominator

    return sim


def sent_embedding(user_input):
    # Write your code here:
    vectors = tokenize_and_vectorize(user_input)
    embedding = np.zeros(len(vectors[0]), )
    for vector in vectors:
        embedding += vector
    embedding /= len(vectors)

    return embedding


if __name__ == '__main__':
    # Define the two strings
    str1 = "I like to eat ice cream"
    str2 = "I like to eat chocolate"

    # Convert the strings to vectors
    vec1 = sent_embedding(str1)
    vec2 = sent_embedding(str2)

    # Calculate the cosine similarity
    sim = cosine_similarity(vec1, vec2)
    print(sim)