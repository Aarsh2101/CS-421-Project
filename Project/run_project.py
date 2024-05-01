from sample_code_1 import num_sentences, spelling_mistakes
from sample_code_2 import agreement, verbs
from sample_code_3 import address_topic
import pandas as pd

# MIN, MAX SCORES
min_score_sents, max_score_sents = 1, 5
min_score_spell, max_score_spell = 0, 4
min_score_agree, max_score_agree = 1, 5
min_score_verbs, max_score_verbs = 1, 5


def score_essay(prompt, essay):
    sents_score = num_sentences(essay, min_score_sents, max_score_sents)
    spell_score = spelling_mistakes(essay, min_score_spell, max_score_spell)
    agree_score = agreement(essay, min_score_agree, max_score_agree)
    verbs_score = verbs(essay, min_score_verbs, max_score_verbs)
    address_topic_score = address_topic(prompt, essay)
    final_score = 2*sents_score - spell_score + agree_score + verbs_score + 3*address_topic_score
    return sents_score, spell_score, agree_score, verbs_score, address_topic_score, final_score


if __name__ == '__main__':
    df = pd.read_csv('Project/index.csv', sep=';')
    for index, row in df.iterrows():
        prompt = row['prompt']
        with open('Project/essays/' + row['filename'], 'r') as file:
            essay = file.read()
            sents_score, spell_score, agree_score, verbs_score, address_topic_score, final_score = score_essay(prompt, essay)
        break
    print(f"Sentence Score: {round(sents_score,2)}")
    print(f"Spelling Score: {round(spell_score,2)}")
    print(f"Agreement Score: {round(agree_score,2)}")
    print(f"Verbs Score: {round(verbs_score,2)}")
    print(f"Address Topic Score: {round(address_topic_score,2)}")
    print(f"Final Score: {round(final_score,2)}")
