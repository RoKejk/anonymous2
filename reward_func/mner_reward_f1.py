import json
import re
from sklearn.metrics import f1_score

_SOLUTION_CLIP_CHARS = 1000


def extract_solution(solution_str):
    if len(solution_str) > _SOLUTION_CLIP_CHARS:
        solution_str = solution_str[-_SOLUTION_CLIP_CHARS:]

    pattern = re.compile(r'<entities>.+</entities>', re.DOTALL)
    solutions = re.findall(pattern, solution_str)
    if len(solutions) == 0:
        final_answer = None
    else:
        try:
            solution = re.sub(r'<entities>|</entities>', '', solutions[-1]).strip()
            final_answer = json.loads(solution)
        except:
            final_answer = None
    return final_answer


def compute_word_f1(ground_truth, raw_text, seperator, answer):
    if seperator == ' ':
        words = raw_text.split()
    else:
        words = list(raw_text)

    true_labels = ['O'] * len(words)
    pred_labels = ['O'] * len(words)

    for ent in ground_truth:
        if seperator == ' ':
            tokens = ent['text'].split()
        else:
            tokens = list(ent['text'])

        for i in range(len(words) - len(tokens) + 1):
            if words[i:i + len(tokens)] == tokens:
                for j in range(len(tokens)):
                    true_labels[i + j] = ent['type']

    for ent in answer:
        if seperator == ' ':
            tokens = ent['text'].split()
        else:
            tokens = list(ent['text'])

        for i in range(len(words) - len(tokens) + 1):
            if words[i:i + len(tokens)] == tokens:
                for j in range(len(tokens)):
                    pred_labels[i + j] = ent['type']

    return f1_score(true_labels, pred_labels, average='micro', zero_division=0)


def compute_score(solution_str, ground_truth):
    try:
        answer = extract_solution(solution_str)
        if answer is None or not isinstance(answer, list):
            # wrong format
            return -0.01
        elif len(answer) < 1:
            return 0.0
        else:
            ground_truth, raw_text, seperator = ground_truth['ground_truth'], ground_truth['text'], ground_truth['seperator']
            return compute_word_f1(ground_truth, raw_text, seperator, answer)
    except:
        return -0.01  # Any unexpected error


if __name__ == '__main__':
    pass
