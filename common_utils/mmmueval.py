"""
MMMU evaluation — generic version.
Accepts a generate_fn(image, prompt, config) -> str callback.

Two functions:
- mmmu_infer: runs inference on MMMU and saves results
- mmmu_eval: evaluates results from a saved output file
"""

import json
import os
from pathlib import Path
from PIL import Image
import torch
from tqdm import tqdm
import random

from .data_utils import (
    save_json, load_yaml, construct_prompt, process_single_sample,
    CAT_SHORT2LONG, DOMAIN_CAT2SUB_CAT
)
from .eval_utils import (
    evaluate, parse_multi_choice_response, parse_open_response, calculate_ins_level_acc
)
from datasets import load_dataset, concatenate_datasets


def mmmu_infer(generate_fn, output_path, myconfig=None,
               data_path=None, config_path=None, split='validation'):
    """
    Run MMMU inference and save results.

    Args:
        generate_fn: callable(image, prompt: str, config: dict) -> str
        output_path: where to save results JSON
        myconfig: generation config dict (e.g. {"max_new_tokens": 128})
        data_path: path to MMMU datasets directory
        config_path: path to MMMU config YAML
        split: 'validation' or 'dev'
    """
    import sys
    if myconfig is None:
        myconfig = {"max_new_tokens": 128, "num_beams": 1, "do_sample": False}

    REPO_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(REPO_ROOT))
    from config import MMMU_DATASETS, MMMU_CONFIG

    data_path = data_path or MMMU_DATASETS
    config_path = config_path or MMMU_CONFIG

    config = load_yaml(config_path)
    for key, value in config.items():
        if key != 'eval_params' and type(value) == list:
            assert len(value) == 1, 'key {} has more than one value'.format(key)
            config[key] = value[0]

    sub_dataset_list = []
    for subject in CAT_SHORT2LONG.values():
        sub_dataset = load_dataset(data_path, subject, split=split)
        sub_dataset_list.append(sub_dataset)
    dataset = concatenate_datasets(sub_dataset_list)

    samples = []
    for sample in dataset:
        sample = process_single_sample(sample)
        sample = construct_prompt(sample, config)
        if sample['image']:
            sample['image'] = sample['image'].convert('RGB')
        samples.append(sample)

    out_samples = dict()
    with torch.no_grad():
        for sample in tqdm(samples, desc="  MMMU"):
            prompt = sample['final_input_prompt']
            image = sample['image']

            if image is not None:
                response = generate_fn(image, prompt, myconfig)
            else:  # multiple images
                if sample['question_type'] == 'multiple-choice':
                    all_choices = sample['all_choices']
                    response = random.choice(all_choices)
                else:
                    response = 'INVALID GENERATION FOR MULTIPLE IMAGE INPUTS'

            if sample['question_type'] == 'multiple-choice':
                pred_ans = parse_multi_choice_response(response, sample['all_choices'], sample['index2ans'])
            else:
                pred_ans = response
            out_samples[sample['id']] = pred_ans

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_json(output_path, out_samples)
    print(f"  MMMU inference saved → {output_path}")
    return output_path


def mmmu_eval(output_path, answer_path=None):
    """
    Evaluate MMMU results from a saved output file.

    Args:
        output_path: path to model predictions JSON
        answer_path: path to ground truth answers JSON

    Returns:
        dict with per-category and overall accuracy
    """
    if answer_path is None:
        answer_path = os.path.join(os.path.dirname(__file__), "answer_dict_val.json")

    output_dict = json.load(open(output_path))
    answer_dict = json.load(open(answer_path))

    # group by category
    output_dict_w_cat = {}
    for data_id, parsed_pred in output_dict.items():
        category = "_".join(data_id.split("_")[1:-1])
        if category not in output_dict_w_cat:
            output_dict_w_cat.update({category: {}})
        output_dict_w_cat[category].update({data_id: parsed_pred})

    answer_dict_w_cat = {}
    for data_id, parsed_pred in answer_dict.items():
        category = "_".join(data_id.split("_")[1:-1])
        if category not in answer_dict_w_cat:
            answer_dict_w_cat.update({category: {}})
        answer_dict_w_cat[category].update({data_id: parsed_pred})

    evaluation_result = {}

    for category in CAT_SHORT2LONG.values():
        print("Evaluating: {}".format(category))
        try:
            cat_outputs = output_dict_w_cat[category]
            cat_answers = answer_dict_w_cat[category]
        except KeyError:
            print("Skipping {} for not found".format(category))
            continue

        exampels_to_eval = []
        for data_id, parsed_pred in cat_outputs.items():
            question_type = cat_answers[data_id]['question_type']
            if question_type != 'multiple-choice':
                parsed_pred = parse_open_response(parsed_pred)
            else:
                parsed_pred = parsed_pred

            exampels_to_eval.append({
                "id": data_id,
                "question_type": question_type,
                "answer": cat_answers[data_id]['ground_truth'],
                "parsed_pred": parsed_pred
            })

        judge_dict, metric_dict = evaluate(exampels_to_eval)
        metric_dict.update({"num_example": len(exampels_to_eval)})
        evaluation_result[category] = metric_dict

    printable_results = {}
    for domain, in_domain_cats in DOMAIN_CAT2SUB_CAT.items():
        in_domain_cat_results = {}
        for cat_name in in_domain_cats:
            if cat_name in evaluation_result.keys():
                in_domain_cat_results[cat_name] = evaluation_result[cat_name]
        in_domain_ins_acc = calculate_ins_level_acc(in_domain_cat_results)
        in_domain_data_num = sum([cat_results['num_example'] for cat_results in in_domain_cat_results.values()])
        printable_results['Overall-' + domain] = {"num": int(in_domain_data_num),
                                                  "acc": round(in_domain_ins_acc, 3)}
        for cat_name, cat_results in in_domain_cat_results.items():
            printable_results[cat_name] = {"num": int(cat_results['num_example']),
                                           "acc": round(cat_results['acc'], 3)}

    all_ins_acc = calculate_ins_level_acc(evaluation_result)
    printable_results['Overall'] = {
        "num": sum([cat_results['num_example'] for cat_results in evaluation_result.values()]),
        "acc": round(all_ins_acc, 3)
    }

    return printable_results
