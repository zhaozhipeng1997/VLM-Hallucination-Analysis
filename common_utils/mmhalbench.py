"""
MMHalBench inference — generic version.
Accepts a generate_fn(image, prompt, config) -> str callback.
"""

import json
import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def load_image(image_file):
    """Load an MMHalBench image."""
    import sys
    REPO_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(REPO_ROOT))
    from config import MMHAL_IMAGES

    if image_file.startswith('http') or image_file.startswith('https'):
        image_file = image_file.split('/')[-1]
    return Image.open(os.path.join(MMHAL_IMAGES, image_file)).convert('RGB')


def mmhalbench_infer(generate_fn, output_path, decode="beam"):
    """
    Run MMHalBench inference.

    Args:
        generate_fn: callable(image, prompt: str, config: dict) -> str
        output_path: where to save the output JSON
        decode: label for generation config
    """
    # Resolve input file relative to common_utils
    inputfile = os.path.join(os.path.dirname(__file__), "response_template.json")
    json_data = json.load(open(inputfile, 'r'))

    config = {"max_new_tokens": 128, "num_beams": 1}

    for idx, line in tqdm(enumerate(json_data), total=len(json_data), desc="  MMHalBench"):
        image_src = line['image_src']
        image = load_image(image_src)
        question = line['question']

        response = generate_fn(image, question, config)
        line['model_answer'] = response

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(json_data, f, indent=2)

    print(f"  MMHalBench inference saved → {output_path}")
    return output_path
