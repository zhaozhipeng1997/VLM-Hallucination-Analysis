"""
CHAIR inference — generic version.
Accepts a generate_fn(image, prompt, config) -> str callback.
"""

import os
import json
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def chair_infer(generate_fn, output_path, decode="beam", num_samples=100,
                data_path=None, annotations_path=None):
    """
    Run CHAIR inference: generate captions for COCO images.

    Args:
        generate_fn: callable(image, prompt: str, config: dict) -> str
        output_path: where to save the JSONL results
        decode: label for output file naming (e.g. "beam")
        num_samples: max number of COCO images to process
        data_path: path to COCO val2014 images directory
        annotations_path: path to instances_val2014.json
    """
    import sys
    # Resolve default paths from config if not provided
    if data_path is None or annotations_path is None:
        REPO_ROOT = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(REPO_ROOT))
        from config import COCO_VAL2014, COCO_ANNOTATIONS_ROOT
        data_path = data_path or COCO_VAL2014
        annotations_path = annotations_path or f"{COCO_ANNOTATIONS_ROOT}/instances_val2014.json"

    img_files = os.listdir(data_path)

    with open(annotations_path, 'r') as f:
        lines = f.readlines()
    coco_anns = json.loads(lines[0])
    img_dict = {}

    categories = coco_anns["categories"]
    category_dict = {int(c["id"]): c["name"] for c in categories}

    for img_info in coco_anns["images"]:
        img_dict[img_info["id"]] = {"name": img_info["file_name"], "anns": []}

    for ann_info in coco_anns["annotations"]:
        img_dict[ann_info["image_id"]]["anns"].append(
            category_dict[ann_info["category_id"]]
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    chair_config = {"max_new_tokens": 128, "num_beams": 3, "do_sample": False}
    results = []

    for img_id in tqdm(range(min(len(img_files), num_samples)), desc="  CHAIR"):
        img_file = img_files[img_id]
        img_id_int = int(img_file.split(".jpg")[0][-6:])
        image_path = os.path.join(data_path, img_file)

        try:
            caption = generate_fn(
                Image.open(image_path).convert('RGB'),
                "Please describe this image in detail.",
                chair_config
            )
            results.append({"image_id": img_id_int, "caption": caption})
        except Exception as e:
            print(f"  [WARN] {img_file}: {e}")
            continue

    with open(output_path, 'w') as f:
        for r in results:
            json.dump(r, f)
            f.write('\n')

    print(f"  Generated {len(results)} captions → {output_path}")
    return output_path
