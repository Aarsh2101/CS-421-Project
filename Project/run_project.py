from sample_code_1 import num_sentences, spelling_mistakes
from sample_code_2 import agreement, verbs
from sample_code_3 import address_topic, score_by_essay_incoherence, score_essay_syntax
import pandas as pd
import os

# MIN, MAX SCORES
min_score_sents, max_score_sents = 1, 5
min_score_spell, max_score_spell = 0, 4
min_score_agree, max_score_agree = 1, 5
min_score_verbs, max_score_verbs = 1, 5
min_syntax_score, max_syntax_score = 1, 5
min_score_coherence, max_score_coherence = 1, 5

def score_essay(PATH_TO_ESSAY, prompt=None):
    with open(PATH_TO_ESSAY, 'r') as f:
        text = f.read()
        f.close()
    sents_score = num_sentences(text, min_score_sents, max_score_sents)
    spell_score = spelling_mistakes(text, min_score_spell, max_score_spell)
    agree_score = agreement(text, min_score_agree, max_score_agree)
    verbs_score = verbs(text, min_score_verbs, max_score_verbs)
    syntax_score = score_essay_syntax(text, min_syntax_score, max_syntax_score)
    address_topic_score = address_topic(prompt, text)
    coherence_score = score_by_essay_incoherence(text, min_score_coherence, max_score_coherence)
    final_score = 2*sents_score - spell_score + agree_score + verbs_score + 3*address_topic_score + coherence_score
    predicted_grade = 'high' if final_score >= 20 else 'low'
    return sents_score, spell_score, agree_score, verbs_score, syntax_score, address_topic_score, coherence_score, final_score, predicted_grade


if __name__ == '__main__':

    # IF YOU WANT TO SCORE A SINGLE ESSAY
    
    # PATH_TO_ESSAY = 'essays/746257.txt'
    # prompt = "Do you agree or disagree with the following statement? Most advertisements make products seem much better than they really are. Use specific reasons and examples to support your answer."
    # sents_score, spell_score, agree_score, verbs_score, syntax_score, address_topic_score, coherence_score, final_score, predicted_grade = score_essay(PATH_TO_ESSAY, prompt)
    # print(f"Essay: {row['filename']} \t Final Score: {round(final_score,2)} \t Predicted Class: '{predicted_grade}' \t Actual Class: '{row['grade']}'")


    # IF YOU WANT TO SCORE ALL ESSAYS IN THE DATASET
    df = pd.read_csv('index.csv', sep=';')
    for index, row in df.iterrows():
        prompt = row['prompt']
        PATH_TO_ESSAY = 'essays/' + row['filename']
        sents_score, spell_score, agree_score, verbs_score, syntax_score, address_topic_score, coherence_score, final_score, predicted_grade = score_essay(PATH_TO_ESSAY, prompt)
        # print(f"Sentence Score: {round(sents_score,2)}")
        # print(f"Spelling Score: {round(spell_score,2)}")
        # print(f"Agreement Score: {round(agree_score,2)}")
        # print(f"Verbs Score: {round(verbs_score,2)}")
        # print(f"Syntax Score: {round(syntax_score,2)}")
        # print(f"Address Topic Score: {round(address_topic_score,2)}")
        # print(f"Coherence Score: {round(coherence_score,2)}")
        # print(f"Final Score: {round(final_score,2)}")
        # print(f"Predicted Grade: '{predicted_grade}'")
        
        print(f"Essay: {row['filename']} \t Final Score: {round(final_score,2)} \t Predicted Class: '{predicted_grade}' \t Actual Class: '{row['grade']}'")
