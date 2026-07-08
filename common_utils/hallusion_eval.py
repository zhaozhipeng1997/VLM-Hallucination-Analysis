#!/usr/bin/env python3
"""
HallusionBench — LLM-based evaluation (vLLM / OpenAI-compatible API).
Shared by all models. No model-specific dependencies.

Usage:
    from common_utils.hallusion_eval import hallusion_eval
    hallusion_eval(decode="beam", input_file_name="output.json", ip="127.0.0.1:8000", model="path/to/model")
"""

import json, os, time, uuid, concurrent.futures
from tqdm import tqdm
import requests


# ═══════════════════════════════════════════════════════════════════════════
#  Core: vLLM judge via OpenAI-compatible API
# ═══════════════════════════════════════════════════════════════════════════

def _call_llm_judge(ip: str, model: str, prompt: str, max_tokens: int = 512) -> str:
    """Send a single judge request to vLLM and return the response text."""
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    })
    url = f"http://{ip}/v1/chat/completions"
    resp = requests.post(url, headers={"Content-Type": "application/json"},
                         data=payload, timeout=60)
    resp.raise_for_status()
    data = json.loads(resp.text)
    content = data["choices"][0]["message"]["content"] or ""
    if not content:
        content = data["choices"][0]["message"].get("reasoning_content", "")
    return content.strip()


def evaluate_by_llm(ip, model, data, output_entry, correctness_entry,
                    load_json=False, save_json_path="./hallusion_output.json",
                    max_workers=10):
    """Judge correctness of model predictions against reference answers."""
    if load_json and os.path.exists(save_json_path):
        with open(save_json_path) as f:
            output = json.load(f)
    else:
        output = []

    samples_to_process = data[len(output):]
    session = requests.Session()

    def _process(sample):
        prompt = (
            "Imagine you are an intelligent teacher. Thoroughly read the question, "
            "reference answer and the prediction answer to ensure a clear understanding "
            "of the information provided. Assess the correctness of the predictions. "
            "If the prediction answer does not conflict with the reference answer, "
            'please generate "correct". If the prediction answer conflict with the '
            'reference answer, please generate "incorrect". If the prediction answer '
            'is unclear about the answer, please generate "unclear".\n\n'
            f"Question: {sample['question']}\n"
            f"Reference answer: {sample['gt_answer_details']}\n"
            f"Prediction answer: {sample[output_entry]}\n"
            "Output:"
        )
        for attempt in range(3):
            try:
                text = _call_llm_judge(ip, model, prompt)
                print("evaluate:", text)
                if "incorrect" in text.lower():
                    sample[correctness_entry] = "0"
                elif "correct" in text.lower():
                    sample[correctness_entry] = "1"
                else:
                    sample[correctness_entry] = "2"
                return sample
            except Exception as e:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    print(f"Failed after 3 attempts: {e}")
                    return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_process, s): s for s in samples_to_process}
        for future in tqdm(concurrent.futures.as_completed(futures),
                           total=len(samples_to_process), desc="  Hallusion Eval"):
            result = future.result()
            if result is not None:
                output.append(result)
                with open(save_json_path, 'w') as f:
                    json.dump(output, f)

    return output


def check_same_by_llm(ip, model, data, output_entry,
                      load_json=False, save_json_path="./hallusion_output.json",
                      max_workers=10):
    """Check consistency between two responses to the same question pair."""
    if load_json and os.path.exists(save_json_path):
        with open(save_json_path) as f:
            saved = json.load(f)
        for s in saved:
            for idx, sample in enumerate(data):
                if (sample.get("category") == s.get("category") and
                    sample.get("subcategory") == s.get("subcategory") and
                    sample.get("set_id") == s.get("set_id") and
                    sample.get("figure_id") == s.get("figure_id") and
                    sample.get("question_id") == s.get("question_id") and
                    "same" in s):
                    data[idx]["same"] = s["same"]
                    break

    # Build original response map (figure_id=0)
    orig = {}
    for r in data:
        if str(r.get("figure_id")) == "0":
            key = "_".join([r["category"], r["subcategory"], str(r["set_id"]), str(r["question_id"])])
            orig[key] = r[output_entry]

    to_process = [s for s in data if "same" not in s]
    if not to_process:
        return data

    session = requests.Session()

    def _process(sample):
        key = "_".join([sample["category"], sample["subcategory"],
                        str(sample["set_id"]), str(sample["question_id"])])
        r2 = orig.get(key, "")
        prompt = (
            "Imagine you are an intelligent teacher. Thoroughly read the two responses "
            "to two different questions. Assess the consistency of the information "
            "provided within those two responses. You do not know the specific questions, "
            "but you can assess the consistency among the two responses by checking for "
            'logical conflicts if both responses are correct. If response1 does not '
            'conflict with response2, please generate "same". Otherwise, generate "different".\n\n'
            f"response1: {sample[output_entry]}\n"
            f"response2: {r2}\n"
            "Output:"
        )
        for attempt in range(3):
            try:
                text = _call_llm_judge(ip, model, prompt)
                print("check:", text)
                sample["same"] = "1" if "same" in text.lower() else "0"
                return sample
            except Exception as e:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    print(f"Failed after 3 attempts: {e}")
                    return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_process, s): s for s in to_process}
        for future in tqdm(concurrent.futures.as_completed(futures),
                           total=len(to_process), desc="  Hallusion Check"):
            result = future.result()
            if result is not None:
                with open(save_json_path, 'w') as f:
                    json.dump(data, f)

    return data


# ═══════════════════════════════════════════════════════════════════════════
#  Top-level eval entry point (called by eval_offline.py)
# ═══════════════════════════════════════════════════════════════════════════

def hallusion_eval(decode: str, input_file_name: str, ip: str, model: str,
                   temp_dir: str = None, evaluation: str = None):
    """Run full HallusionBench evaluation: correctness check + consistency check."""
    if evaluation is not None:
        save_vd = evaluation.replace(".json", "_vd.json")
        save_vs = evaluation.replace(".json", "_vs.json")
    elif temp_dir is not None:
        os.makedirs(temp_dir, exist_ok=True)
        suffix = uuid.uuid4().hex[:8]
        save_vd = f"{temp_dir}/hallusion_eval_vd_{decode}_{suffix}.json"
        save_vs = f"{temp_dir}/hallusion_eval_vs_{decode}_{suffix}.json"
    else:
        from pathlib import Path
        temp_dir = str(Path(__file__).resolve().parents[1] / "hallusion_bench_temp")
        os.makedirs(temp_dir, exist_ok=True)
        suffix = uuid.uuid4().hex[:8]
        save_vd = f"{temp_dir}/hallusion_eval_vd_{decode}_{suffix}.json"
        save_vs = f"{temp_dir}/hallusion_eval_vs_{decode}_{suffix}.json"

    with open(input_file_name) as f:
        datas = json.load(f)

    data_vd = [d for d in datas if d.get("category") == "VD"]
    data_vs = [d for d in datas if d.get("category") == "VS"]

    model_output_entry = "model_prediction"
    correctness_entry = "gpt4v_output_gpt_check"

    # VD: evaluate correctness
    data_vd = evaluate_by_llm(ip, model, data_vd, model_output_entry,
                              correctness_entry, save_json_path=save_vd)
    # VD: check consistency
    data_vd = check_same_by_llm(ip, model, data_vd, model_output_entry,
                                save_json_path=save_vd)

    # VS: evaluate correctness
    data_vs = evaluate_by_llm(ip, model, data_vs, model_output_entry,
                              correctness_entry, save_json_path=save_vs)
    # VS: check consistency
    data_vs = check_same_by_llm(ip, model, data_vs, model_output_entry,
                                save_json_path=save_vs)

    print("##### LLM Evaluate #####")
    print(f"  VD: {len(data_vd)} samples → {save_vd}")
    print(f"  VS: {len(data_vs)} samples → {save_vs}")

    # ── Full HallusionBench metrics ──
    from pathlib import Path as _Path
    _llava_utils = _Path(__file__).resolve().parents[1] / "llava" / "utils"
    import sys as _sys
    _sys.path.insert(0, str(_llava_utils))
    from hallusionbench_utils import (
        assign_correctness, get_eval_pair_all, get_eval_fig,
        get_eval_all, get_eval_pair_easy, get_eval_pair_hard, yes_ratio_stats,
    )

    correctness_entry = "gpt4v_output_gpt_check"
    all_data = assign_correctness(data_vd + data_vs, correctness_entry)

    q_pair = get_eval_pair_all(all_data, correctness_entry)
    fig = get_eval_fig(all_data)
    q_all = get_eval_all(all_data, correctness_entry)
    easy = get_eval_pair_easy(all_data)
    hard = get_eval_pair_hard(all_data)
    yes = yes_ratio_stats(all_data)

    wrong = q_pair["wrong"]
    lh_ratio = round(q_pair["LH"] / wrong * 100, 1) if wrong else 0
    vi_ratio = round(q_pair["VI"] / wrong * 100, 1) if wrong else 0
    mixed_ratio = round(q_pair["Mix"] / wrong * 100, 1) if wrong else 0

    results = {
        "evaluation": evaluation or save_vd,
        "qAcc": round(q_pair["correct"] / q_pair["total"] * 100, 2) if q_pair["total"] else 0,
        "fAcc": round(fig["score"] * 100, 2),
        "aAcc": round(q_all["correct"] / q_all["total"] * 100, 2) if q_all["total"] else 0,
        "easy_aAcc": round(easy["correct"] / easy["total"] * 100, 2) if easy["total"] else 0,
        "hard_aAcc": round(hard["correct"] / hard["total"] * 100, 2) if hard["total"] else 0,
        "YesNo_diff": round(yes["diff"], 4),
        "YesNo_fp_ratio": round(yes["fp"], 4),
        "consistency_correct": fig["correct"],
        "consistency_inconsistent": fig["inconsistent"],
        "consistency_wrong": fig["wrong"],
        "LH_ratio": lh_ratio,
        "VI_ratio": vi_ratio,
        "Mixed_ratio": mixed_ratio,
        "LH": q_pair["LH"],
        "VI": q_pair["VI"],
        "Mixed": q_pair["Mix"],
        "wrong_total": wrong,
        "details": {
            "q_pair": q_pair, "fig": fig, "q_all": q_all,
            "easy": easy, "hard": hard, "yes_ratio": yes,
        },
    }

    print(f"  qAcc={results['qAcc']:.1f}%  fAcc={results['fAcc']:.1f}%  aAcc={results['aAcc']:.1f}%")
    print(f"  easy={results['easy_aAcc']:.1f}%  hard={results['hard_aAcc']:.1f}%")
    print(f"  YesNo diff={results['YesNo_diff']:.4f}  fp_ratio={results['YesNo_fp_ratio']:.4f}")
    print(f"  Consistency: correct={results['consistency_correct']}  inconsistent={results['consistency_inconsistent']}  wrong={results['consistency_wrong']}")
    print(f"  LH={lh_ratio:.1f}% ({q_pair['LH']}/{wrong})  VI={vi_ratio:.1f}% ({q_pair['VI']}/{wrong})  Mixed={mixed_ratio:.1f}% ({q_pair['Mix']}/{wrong})")

    return results
