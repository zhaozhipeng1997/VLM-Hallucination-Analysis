"""
common_utils — shared evaluation utilities for all VLM models.

Each evaluation module accepts a generate_fn(image, prompt, config) -> str
callback so any model (LLaVA, InstructBLIP, InternVL, MiniCPM, Qwen, etc.)
can plug in its own generation logic.
"""

from .pope_loader import POPEDataSet
from .pope_eval import pope_eval, print_acc, recorder
from .chair_infer import chair_infer
from .chair_eval import chair_eval
from .hallusionbench import hallusion_infer
from .mmhalbench import mmhalbench_infer
from .mmmueval import mmmu_infer, mmmu_eval
from .data_utils import (
    CAT_SHORT2LONG, DOMAIN_CAT2SUB_CAT,
    save_json, load_yaml, construct_prompt, process_single_sample
)
from .eval_utils import (
    evaluate, parse_multi_choice_response, parse_open_response,
    calculate_ins_level_acc
)
