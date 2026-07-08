"""
HallusionBench inference — generic version.
Accepts a generate_fn(image, prompt, config) -> str callback.
"""

import json
import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def hallusion_infer(generate_fn, output_path, decode="beam"):
    """
    Run HallusionBench inference.

    Args:
        generate_fn: callable(image, prompt: str, config: dict) -> str
        output_path: where to save the output JSON
        decode: label for generation config (e.g. "beam")
    """
    import sys
    REPO_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(REPO_ROOT))
    from config import HALLUSIONBENCH_JSON, HALLUSIONBENCH_IMAGES

    model_output_entry = "model_prediction"

    with open(HALLUSIONBENCH_JSON) as json_file:
        datas = json.load(json_file)

    config = {"max_new_tokens": 1024, "num_beams": 1}

    for sample in tqdm(datas, desc="  HallusionBench"):
        if sample['filename'] is not None:
            if sample['filename'].startswith("./"):
                img_path = os.path.join(HALLUSIONBENCH_IMAGES, sample['filename'].lstrip("./"))
            else:
                img_path = sample['filename']
            image = Image.open(img_path).convert('RGB')
        else:
            image = None

        question = sample['question']

        if image is not None:
            response = generate_fn(image, question, config)
        else:
            # Text-only questions — pass a blank image with just the prompt
            response = generate_fn(None, question, config)

        sample[model_output_entry] = response

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(datas, f, indent=2)

    print(f"  HallusionBench inference saved → {output_path}")
    return output_path
