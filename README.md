# CS-421-Project

# Teammate 1: Aarsh Patel   Net ID: 659999805
# Teammate 2: Raj Mehta

# Project files & functions:


## sample_code_1.py:

### general_scorer_gaussian_assumption: 

#### Description: Calculates a score based on a Gaussian (normal) distribution assumption of input data. It converts a raw score (x) into a standardized score within a specified range.

#### Parameters: x (float): The raw score to be standardized.
mean (float): The mean of the distribution.
stddev (float): The standard deviation of the distribution.
min_score (int): The minimum possible score.
max_score (int): The maximum possible score.
reverse (bool, optional): If True, reverses the scoring direction.

#### Returns: float: A score normalized to the specified range and clipped to ensure it remains within the min_score and max_score bounds.

#### Detailed Behavior:
Standardization: Converts a raw score x into a z-score using the formula (x - mean) / stddev.
Normalization: Maps the z-score to a score within the specified range [min_score, max_score]. It linearly transforms z-scores between -3 and 3 to this range.
Reversal Option: If reverse is set to True, the direction of scoring is inverted, making higher raw scores correspond to lower normalized scores.
Clipping: Ensures the final score does not exceed the boundaries set by min_score and max_score.


### count_sentences_with_spacy: 

#### Description: Counts the number of sentences in a given text using the spacy library.

#### Parameters: text (str): Text to analyze.

#### Returns: int: The count of sentences in the text.

#### Detailed Behavior:
Text Processing: Uses the spacy library to parse the given text into a document object, which organizes the text into tokens and sentence structures.
Sentence Extraction: Extracts sentences from the document object and counts them.


### decontracted: 

#### Description: This function expands English contractions into their full form, which can be helpful for various natural language processing tasks that benefit from standardized text formats.

#### Parameters: phrase (str): A string containing English text with contractions.

#### Returns: str: The input string with all contractions expanded.

#### Detailed Behavior:
1. The function uses regular expressions to replace common English contractions.
2. Specific contractions handled include replacements for "won't" to "will not" and "can't" to "can not".
3. More general patterns cover other common contractions like "n't" (not), "'re" (are), "'s" (is), etc.


### num_sentences: 

#### Description: Evaluates the text based on the number of sentences, using a scoring system based on a Gaussian distribution of known sentence counts.

#### Parameters: text (str): Text to evaluate.
sentence_counts (list): Historical data of sentence counts.
min_score (int): Minimum score to assign.
max_score (int): Maximum score to assign.

#### Returns: float: A score reflecting the appropriateness of the sentence count in the text.

#### Detailed Behavior:
Initial Check: Directly returns min_score if the sentence count is 10 or less.
Data Filtering: Filters out sentence counts from historical data that are 10 or less (non-informative data).
Statistical Analysis: Calculates the mean and standard deviation of the filtered historical sentence counts.
Scoring: Applies the general_scorer_gaussian_assumption to the counted sentences, scoring them based on how they compare statistically to historical data.


### spell_check: 

#### Description: This function checks the spelling of words in a text and calculates the percentage of misspelled words. It first expands contractions using the decontracted function, then tokenizes the text, lemmatizes each word, and finally checks for spelling errors.

#### Parameters: text (str): A string of text in which to check spelling.

#### Returns: float: The percentage of misspelled words in the text, rounded to two decimal places.

#### Detailed Behavior:
1. The text is first decontracted to normalize contractions.
2. nltk.word_tokenize is used for tokenizing the string into words.
3. Each word is lemmatized using nltk.stem.WordNetLemmatizer to reduce it to its base or dictionary form.
4. spellchecker.SpellChecker is utilized to identify words not recognized by its dictionary.
5. The function calculates the percentage of words identified as misspelled compared to the total number of words.


### spelling_mistakes: 

#### Description: Evaluates the text based on the percentage of spelling mistakes, using a scoring system based on a Gaussian distribution of known spelling mistake rates.

#### Parameters: text (str): Text to evaluate.
mistakes_list (list): Historical data of spelling mistakes percentages.
min_score (int): Minimum score to assign.
max_score (int): Maximum score to assign.

#### Returns: float: A score reflecting the quality of spelling in the text.

#### Detailed Behavior:
1. The function uses regular expressions to replace common English contractions.
2. Specific contractions handled include replacements for "won't" to "will not" and "can't" to "can not".
3. More general patterns cover other common contractions like "n't" (not), "'re" (are), "'s" (is), etc.


## sample_code_2.py:

### agreement: 

### verbs: 