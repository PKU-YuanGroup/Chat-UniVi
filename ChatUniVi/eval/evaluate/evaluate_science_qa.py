import argparse
import json
import os
import re
import random
import numpy as np


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-dir', type=str)
    parser.add_argument('--result-file', type=str)
    parser.add_argument('--output-file', type=str)
    parser.add_argument('--output-result', type=str)
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--options', type=list, default=["A", "B", "C", "D", "E"])
    return parser.parse_args()


def convert_caps(results):
    fakecaps = []
    for result in results:
        image_id = result['question_id']
        caption = result['text']
        fakecaps.append({"image_id": int(image_id), "caption": caption})
    return fakecaps


def get_pred_idx(prediction, choices, options):
    """
    Get the index (e.g. 2) from the prediction (e.g. 'C')
    """
    if prediction in options[:len(choices)]:
        return options.index(prediction)
    else:
        return random.choice(range(len(choices)))


if __name__ == "__main__":
    args = get_args()

    base_dir = args.base_dir
    split_indices = json.load(open(os.path.join(base_dir, "pid_splits.json")))[args.split]
    problems = json.load(open(os.path.join(base_dir, "problems.json")))
    predictions = [json.loads(line) for line in open(args.result_file)]
    predictions = {pred['question_id']: pred for pred in predictions}
    split_problems = {idx: problems[idx] for idx in split_indices}

    results = {'correct': [], 'incorrect': []}
    sqa_results = {}
    sqa_results['acc'] = None
    sqa_results['correct'] = None
    sqa_results['count'] = None
    sqa_results['results'] = {}
    sqa_results['outputs'] = {}

    sqa_results['NAT'] = []
    sqa_results['SOC'] = []
    sqa_results['LAN'] = []
    sqa_results['TXT'] = []
    sqa_results['IMG'] = []
    sqa_results['NO'] = []
    sqa_results['G1-6'] = []
    sqa_results['G7-12'] = []

    for prob_id, prob in split_problems.items():
        if prob_id not in predictions:
            continue
        pred = predictions[prob_id]
        pred_text = pred['text']

        pattern = re.compile(r'The answer is ([A-Z]).')
        res = pattern.findall(pred_text)
        if len(res) == 1:
            answer = res[0]  # 'A', 'B', ...
        else:
            answer = pred['pred']

        pred_idx = get_pred_idx(answer, prob['choices'], args.options)

        analysis = {
            'question_id': prob_id,
            'parsed_ans': answer,
            'ground_truth': args.options[prob['answer']],
            'question': pred['prompt'],
            'pred': pred_text,
            'is_multimodal': '<image>' in pred['prompt'],
        }

        sqa_results['results'][prob_id] = get_pred_idx(answer, prob['choices'], args.options)
        sqa_results['outputs'][prob_id] = pred_text

        if pred_idx == prob['answer']:
            results['correct'].append(analysis)
            cur_result = 1
        else:
            results['incorrect'].append(analysis)
            cur_result = 0

        if prob['subject'] == 'natural science':
            sqa_results['NAT'].append(cur_result)
        elif prob['subject'] == 'social science':
            sqa_results['SOC'].append(cur_result)
        elif prob['subject'] == 'language science':
            sqa_results['LAN'].append(cur_result)

        if prob['hint']:
            sqa_results['TXT'].append(cur_result)
        if prob['image']:
            sqa_results['IMG'].append(cur_result)
        if not prob['hint'] and not prob['image']:
            sqa_results['NO'].append(cur_result)

        if prob['grade'] in ['grade1', 'grade2', 'grade3', 'grade4', 'grade5', 'grade6']:
            sqa_results['G1-6'].append(cur_result)
        elif prob['grade'] in ['grade7', 'grade8', 'grade9', 'grade10', 'grade11', 'grade12']:
            sqa_results['G7-12'].append(cur_result)


    correct = len(results['correct'])
    total = len(results['correct']) + len(results['incorrect'])
    print(f'Total: {total}, Correct: {correct}, Accuracy: {correct / total * 100:.2f}%')

    print(f'Subject NAT: {len(sqa_results["NAT"])}, Correct: {sum(sqa_results["NAT"])}, Accuracy: {np.mean(sqa_results["NAT"]) * 100:.2f}%')
    print(f'Subject SOC: {len(sqa_results["SOC"])}, Correct: {sum(sqa_results["SOC"])}, Accuracy: {np.mean(sqa_results["SOC"]) * 100:.2f}%')
    print(f'Subject LAN: {len(sqa_results["LAN"])}, Correct: {sum(sqa_results["LAN"])}, Accuracy: {np.mean(sqa_results["LAN"]) * 100:.2f}%')

    print(f'Context Modality TXT: {len(sqa_results["TXT"])}, Correct: {sum(sqa_results["TXT"])}, Accuracy: {np.mean(sqa_results["TXT"]) * 100:.2f}%')
    print(f'Context Modality IMG: {len(sqa_results["IMG"])}, Correct: {sum(sqa_results["IMG"])}, Accuracy: {np.mean(sqa_results["IMG"]) * 100:.2f}%')
    print(f'Context Modality NO: {len(sqa_results["NO"])}, Correct: {sum(sqa_results["NO"])}, Accuracy: {np.mean(sqa_results["NO"]) * 100:.2f}%')

    print(f'Grade G1-6: {len(sqa_results["G1-6"])}, Correct: {sum(sqa_results["G1-6"])}, Accuracy: {np.mean(sqa_results["G1-6"]) * 100:.2f}%')
    print(f'Grade G7-12: {len(sqa_results["G7-12"])}, Correct: {sum(sqa_results["G7-12"])}, Accuracy: {np.mean(sqa_results["G7-12"]) * 100:.2f}%')

    sqa_results['acc'] = correct / total * 100
    sqa_results['correct'] = correct
    sqa_results['count'] = total

    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    with open(args.output_result, 'w') as f:
        json.dump(sqa_results, f, indent=2)