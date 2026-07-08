"""
POPE evaluation — generic version.
Accepts a generate_fn(image, prompt, config) -> str callback so any model can plug in.
"""

import torch
from tqdm import tqdm
from PIL import Image

from .pope_loader import POPEDataSet


def recorder(out, pred_list):
    """Determine if a response contains negation words."""
    NEG_WORDS = ["No", "not", "no", "NO"]

    if isinstance(out, str):
        lines = out.split('\n')
    else:
        lines = out

    has_neg = False
    for line in lines:
        clean_line = line.split(':', 1)[-1].strip()
        clean_line = clean_line.replace('.', '').replace(',', '')
        words = clean_line.lower().split()
        if any(word in [w.lower() for w in NEG_WORDS] for word in words) or \
           any(word.endswith("n't") for word in words):
            has_neg = True
            break

    pred_list.append(0 if has_neg else 1)
    return pred_list


def print_acc(pred_list, label_list):
    """Compute POPE metrics."""
    pos, neg = 1, 0
    yes_ratio = pred_list.count(1) / len(pred_list)

    TP = TN = FP = FN = 0
    for pred, label in zip(pred_list, label_list):
        if pred == pos and label == pos:
            TP += 1
        elif pred == pos and label == neg:
            FP += 1
        elif pred == neg and label == neg:
            TN += 1
        elif pred == neg and label == pos:
            FN += 1

    precision = float(TP) / float(TP + FP) if (TP + FP) > 0 else 0
    recall = float(TP) / float(TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    acc = (TP + TN) / (TP + TN + FP + FN)

    return acc, precision, recall, f1, yes_ratio


def pope_eval(generate_fn, pope_path, data_path, pope_config=None):
    """
    Run POPE evaluation using a generic generate function.

    Args:
        generate_fn: callable(image: PIL.Image, prompt: str, config: dict) -> str
        pope_path: path to the POPE JSON file
        data_path: path to COCO val2014 images
        pope_config: dict with generation parameters (e.g. {"max_new_tokens": 64})

    Returns:
        (acc, precision, recall, f1, yes_ratio)
    """
    if pope_config is None:
        pope_config = {"max_new_tokens": 64, "num_beams": 1, "do_sample": False}

    pope_dataset = POPEDataSet(pope_path=pope_path, data_path=data_path)
    pope_loader = torch.utils.data.DataLoader(
        pope_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        persistent_workers=True,
        drop_last=False,
    )

    pred_list, label_list = [], []
    for data in tqdm(pope_loader, desc="  POPE"):
        image_path = data["image_path"][0]
        image = Image.open(image_path).convert('RGB')
        query = data["query"][0] if isinstance(data["query"], list) else data["query"]
        label = data["label"]
        label_list.append(int(label.cpu().detach()[0].item()))

        response = generate_fn(image, query, pope_config)
        pred = 1 if "yes" in response.lower() else 0
        pred_list.append(pred)

    if len(pred_list) == 0:
        return 0, 0, 0, 0, 0

    acc, precision, recall, f1, yes_ratio = print_acc(pred_list, label_list)
    print(f"    Acc={acc:.3f} Precision={precision:.3f} Recall={recall:.3f} F1={f1:.3f} Yes%={yes_ratio:.3f}")
    return acc, precision, recall, f1, yes_ratio
