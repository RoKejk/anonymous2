import json
import re

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
            print('format error')
            final_answer = None
    return final_answer


def compute_score(solution_str, ground_truth, score=1.0):
    answer = extract_solution(solution_str=solution_str)
    if answer is None:
        # format penalty
        return -0.01
    elif len(answer) < 1:
        return 0.0
    else:
        scores = []
        table = {i['text']: i['type'] for i in ground_truth}
        for entity in answer:
            if entity.get('text') and entity.get('type'):
                if entity['text'] in table:
                    if entity['type'] == table[entity['text']]:
                        scores.append(score)
                    else:
                        scores.append(0.5 * score)
                else:
                    scores.append(0.0)
            else:
                scores.append(-0.01)
        return sum(scores) / len(scores)
