from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
HOME_ROOT = Path(os.getenv("DATA_ROOT", "."))



LLAVA_15_7B_HF = str(HOME_ROOT / "llava-1.5-7b-hf") # https://huggingface.co/liuhaotian/llava-v1.5-7b
LLAVA_BENCH_IN_THE_WILD = str(HOME_ROOT / "llava-bench-in-the-wild") # https://huggingface.co/datasets/lmms-lab/llava-bench-in-the-wild, case study data

INSTRUCTBLIP_VICUNA_7B = str(HOME_ROOT / "instructblip-vicuna-7b") #https://huggingface.co/Salesforce/instructblip-vicuna-7b

# ============= New models (extension) =============
QWEN25VL_7B = str(HOME_ROOT / "Qwen2.5-VL-7B-Instruct")  # https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
MINICPMV26_8B = str(HOME_ROOT / "MiniCPM-V-2_6")  # https://huggingface.co/openbmb/MiniCPM-V-2_6
INTERNVL35_8B = str(HOME_ROOT / "InternVL3_5-8B")  # https://huggingface.co/OpenGVLab/InternVL3_5-8B-HF

COCO_VAL2014 = str(HOME_ROOT / "COCO2014/val2014")
COCO_ANNOTATIONS_ROOT = str(HOME_ROOT / "COCO2014/annotations_trainval2014" / "annotations")
COCO_VAL2014_ANNOTATIONS = str(
    HOME_ROOT / "COCO2014/annotations_trainval2014" / "annotations" / "instances_val2014.json"
)

POPE_RANDOM_JSON = str(HOME_ROOT / "pope_coco" / "coco_pope_random.json")
POPE_POPULAR_JSON = str(HOME_ROOT / "pope_coco" / "coco_pope_popular.json")
POPE_ADVERSARIAL_JSON = str(HOME_ROOT / "pope_coco" / "coco_pope_adversarial.json")
POPE_PATH = str(HOME_ROOT / "pope_coco")

MMHAL_IMAGES = str(HOME_ROOT / "MMHal-Bench" / "images")
MMHAL_IMG_DIR = str(HOME_ROOT / "MMHal-Bench" / "images")
MMHAL_JSON = str(HOME_ROOT / "MMHal-Bench" / "mmhal_bench.json")

HALLUSIONBENCH_JSON = str(HOME_ROOT / "HallusionBench" / "HallusionBench.json")
HALLUSIONBENCH_IMAGES = str(HOME_ROOT / "HallusionBench" / "hallusion_bench")
HALLUSIONBENCH_DIR = str(HOME_ROOT / "HallusionBench")

MMMU_DATASETS = str(HOME_ROOT / "MMMU_Datasets")
MMMU_CONFIG = str(HOME_ROOT / "MMMU" / "mmmu" / "configs" / "llava1.5.yaml")

# ============= Output Directories =============
# Created relative to cwd so each model gets its own set of output dirs
# (run_all.sh cd's into each model subdirectory before running)

# Experiment results and outputs
RESULTS_DIR = str(Path.cwd() / "results")
CASE_STUDY_DIR = str(Path.cwd() / "case_study")
INFER_DIR = str(Path.cwd() / "infer_dir")
MMMU_OUT_DIR = str(Path.cwd() / "mmmu_out")
CHAIR_OUT_DIR = str(Path.cwd() / "chair_out")
HALLUSION_BENCH_TEMP_DIR = str(Path.cwd() / "hallusion_bench_temp")


def ensure_output_dirs():
    """Automatically create all output directories if they don't exist."""
    import os
    for dir_path in [RESULTS_DIR, CASE_STUDY_DIR, INFER_DIR, MMMU_OUT_DIR, CHAIR_OUT_DIR, HALLUSION_BENCH_TEMP_DIR]:
        os.makedirs(dir_path, exist_ok=True)
