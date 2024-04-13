from sample_code_1 import num_sentences, spelling_mistakes
from sample_code_2 import agreement, verbs


# MIN, MAX SCORES
min_score_sents, max_score_sents = 1, 5
min_score_spell, max_score_spell = 0, 4
min_score_agree, max_score_agree = 1, 5
min_score_verbs, max_score_verbs = 1, 5


def score_essay(PATH_TO_ESSAY):
    with open(PATH_TO_ESSAY, 'r') as f:
        text = f.read()
        f.close()
    sents_score = num_sentences(text, min_score_sents, max_score_sents)
    spell_score = spelling_mistakes(text, min_score_spell, max_score_spell)
    agree_score = agreement(text, min_score_agree, max_score_agree)
    verbs_score = verbs(text, min_score_verbs, max_score_verbs)

    final_score = 2*sents_score - spell_score + agree_score + verbs_score
    return final_score


if __name__ == '__main__':
    PATH_TO_ESSAY = 'essays_dataset/essays/essay_1.txt'
    print("Essay Score = ", round(score_essay(PATH_TO_ESSAY), 2))
