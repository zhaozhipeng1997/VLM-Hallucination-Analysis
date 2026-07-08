#!/usr/bin/env python3
"""
Mechanistic Attribution — Updated Runner
===============================================
Supports three analysis modes:
  --mode dynamic        Dynamic circuit discovery (per-token head tracking)
  --mode encoding       Encoding vs arbitration decomposition
  --mode cross          Cross-architecture invariant circuit comparison
  --mode full           All three (requires multiple models)

Usage:
    python run_attribution.py --model qwen2.5-vl --mode dynamic --num_samples 10
    python run_attribution.py --model all --mode full --num_samples 15
    python run_attribution.py --model qwen2.5-vl --mode encoding --image_path cat.jpg
"""

import os, sys, json, argparse
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from config import (
    LLAVA_15_7B_HF, QWEN25VL_7B, MINICPMV26_8B, INTERNVL35_8B,
    INSTRUCTBLIP_VICUNA_7B,
    COCO_VAL2014, RESULTS_DIR, ensure_output_dirs,
)
from dynamic_circuit import (
    dynamic_circuit_discovery,
    cross_architecture_comparison,
    encoding_vs_arbitration_decomposition,
)

# ═══════════════════════════════════════════════════════════════════════════
#  Model Registry
# ═══════════════════════════════════════════════════════════════════════════

MODELS = {
    "llava-1.5": {
        "name": "LLaVA-1.5",
        "path": LLAVA_15_7B_HF,
        "model_cls": "LlavaForConditionalGeneration",
        "proc_cls": "AutoProcessor",
        "gen_module": "llava.utils.causal_generator_optimized",
        "gen_cls": "OptimizedCausalLlavaGenerator",
        "num_layers": 32, "num_heads": 32,
        "projector": "mlp",
    },
    "qwen2.5-vl": {
        "name": "Qwen2.5-VL",
        "path": QWEN25VL_7B,
        "model_cls": "Qwen2_5_VLForConditionalGeneration",
        "proc_cls": "AutoProcessor",
        "gen_module": "qwen2_5_vl.utils.causal_generator_optimized_qwen",
        "gen_cls": "OptimizedCausalQwenGenerator",
        "num_layers": 28, "num_heads": 28,
        "projector": "mlp",
    },
    "minicpm-v2.6": {
        "name": "MiniCPM-V2.6",
        "path": MINICPMV26_8B,
        "model_cls": "AutoModel",
        "proc_cls": "AutoTokenizer",
        "gen_module": "minicpmv.utils.causal_generator_optimized_minicpm",
        "gen_cls": "OptimizedCausalMiniCPMGenerator",
        "num_layers": 28, "num_heads": 28,
        "projector": "perceiver",
        "trust_remote_code": True,
    },
    "internvl3.5": {
        "name": "InternVL3.5",
        "path": INTERNVL35_8B,
        "model_cls": "AutoModel",
        "proc_cls": "AutoTokenizer",
        "gen_module": "internvl3_5.utils.causal_generator_optimized_internvl",
        "gen_cls": "OptimizedCausalInternVLGenerator",
        "num_layers": 36, "num_heads": 32,
        "projector": "pixelshuffle",
        "trust_remote_code": True,
    },
    "instructblip": {
        "name": "InstructBLIP",
        "path": INSTRUCTBLIP_VICUNA_7B,
        "model_cls": "InstructBlipForConditionalGeneration",
        "proc_cls": "InstructBlipProcessor",
        "gen_module": "instructblip.utils.causal_generator_optimized",
        "gen_cls": "OptimizedCausalInstructBlipGenerator",
        "num_layers": 32, "num_heads": 32,
        "projector": "mlp",
    },
}


def load_model_and_generator(model_key: str):
    """Load model + processor + generator class for a given model key."""
    import importlib, torch

    cfg = MODELS[model_key]
    path = cfg["path"]

    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")

    # Import model class
    cls_name = cfg["model_cls"]
    if cls_name == "custom":
        raise NotImplementedError(f"Custom loading for {model_key} not implemented in runner")
    import transformers
    model_cls = getattr(transformers, cls_name)

    # Import processor
    proc_name = cfg["proc_cls"]
    processor_cls = getattr(transformers, proc_name)

    # Load
    load_kwargs = {
        "torch_dtype": torch.float16,
        "attn_implementation": "eager",
        "device_map": "auto",
    }
    if cfg.get("trust_remote_code"):
        load_kwargs["trust_remote_code"] = True

    print(f"  Loading {cfg['name']} ({cfg['projector']} projector)...")
    model = model_cls.from_pretrained(path, **load_kwargs).eval()
    proc_kwargs = {"trust_remote_code": True} if cfg.get("trust_remote_code") else {}

    # Qwen2.5-VL: limit visual tokens to prevent OOM (matching baseline.sh).
    if model_key == "qwen2.5-vl":
        min_pixels = 256 * 28 * 28
        max_pixels = 512 * 28 * 28
        proc_kwargs["min_pixels"] = min_pixels
        proc_kwargs["max_pixels"] = max_pixels
        print(f"  Qwen2.5-VL visual token limits: {min_pixels} → {max_pixels} px")

    processor = processor_cls.from_pretrained(path, **proc_kwargs)

    # Import generator
    gen_mod = importlib.import_module(cfg["gen_module"])
    generator_cls = getattr(gen_mod, cfg["gen_cls"])

    return model, processor, generator_cls, cfg


# ═══════════════════════════════════════════════════════════════════════════
#  Mode: Dynamic Circuit Discovery
# ═══════════════════════════════════════════════════════════════════════════

def run_dynamic_discovery(model_key: str, num_samples: int, output_dir: Path):
    """Run dynamic circuit discovery for a single model."""
    cfg = MODELS[model_key]
    model, processor, gen_cls, _ = load_model_and_generator(model_key)

    model_dir = output_dir / model_key / "dynamic"
    model_dir.mkdir(parents=True, exist_ok=True)

    img_files = sorted(os.listdir(COCO_VAL2014))[:num_samples]
    all_results = []

    for img_file in tqdm(img_files, desc=f"  [{cfg['name']}] Dynamic circuit"):
        image_path = os.path.join(COCO_VAL2014, img_file)
        try:
            result = dynamic_circuit_discovery(
                model=model, processor=processor,
                generator_class=gen_cls,
                image=image_path,
                prompt="Please describe this image in detail.",
                num_layers=cfg["num_layers"],
                num_heads=cfg["num_heads"],
                max_new_tokens=64,
            )
            if result:
                result['image'] = img_file
                all_results.append(result)
        except Exception as e:
            print(f"    [WARN] {img_file}: {e}")
            continue

    # Aggregate across samples
    if not all_results:
        print("  No results collected.")
        return None

    # Combine per-token head deltas (pad to max length)
    max_tokens = max(r['per_token_head_deltas'].shape[0] for r in all_results)
    nl, nh = cfg["num_layers"], cfg["num_heads"]
    combined_deltas = np.zeros((len(all_results), max_tokens, nl, nh))
    combined_layer_dom = np.zeros((len(all_results), max_tokens, nl))

    for i, r in enumerate(all_results):
        t = r['per_token_head_deltas'].shape[0]
        combined_deltas[i, :t, :, :] = r['per_token_head_deltas']
        combined_layer_dom[i, :t, :] = r['per_layer_dominance']

    # Mean across samples (masked by actual length)
    mean_per_token = np.zeros((max_tokens, nl, nh))
    counts = np.zeros(max_tokens)
    for i, r in enumerate(all_results):
        t = r['per_token_head_deltas'].shape[0]
        mean_per_token[:t] += combined_deltas[i, :t]
        counts[:t] += 1
    counts[counts == 0] = 1
    mean_per_token /= counts[:, None, None]

    # Find top heads overall
    overall_importance = mean_per_token.mean(axis=0)  # (nl, nh)
    top_heads = []
    for l in range(nl):
        for h in range(nh):
            top_heads.append((l, h, float(overall_importance[l, h])))
    top_heads.sort(key=lambda x: x[2], reverse=True)

    # Circuit regime analysis
    all_regimes = []
    for r in all_results:
        all_regimes.extend(r.get('circuit_regimes', []))
    all_transitions = []
    for r in all_results:
        all_transitions.extend(r.get('circuit_transitions', []))

    # Flatten: accumulate per-token deltas across all samples
    all_delta_values = []
    for r in all_results:
        dt = r.get('delta_t')
        if dt:
            # May be a list of floats, or list of dicts from older compat CodeGenTokenSource attrs
            for v in dt:
                if isinstance(v, (int, float)):
                    all_delta_values.append(v)
                elif isinstance(v, dict):
                    all_delta_values.append(v.get('ate', 0.0))
                else:
                    all_delta_values.append(float(v))
    mean_delta_t = float(np.mean(all_delta_values)) if all_delta_values else 0.0

    # Save
    summary = {
        'model': cfg['name'],
        'projector_type': cfg['projector'],
        'num_layers': nl, 'num_heads': nh,
        'num_samples': len(all_results),
        'mean_tokens': float(np.mean([r['per_token_head_deltas'].shape[0] for r in all_results])),
        'top_20_heads': [(int(l), int(h), float(s)) for l, h, s in top_heads[:20]],
        'circuit_regimes': [{
            'type': 'aggregate',
            'regime_distribution': {},
        }],
        'n_transitions': len(all_transitions),
        'mean_delta_t': mean_delta_t,
    }

    # Regime distribution across samples
    regime_layers = defaultdict(int)
    for r in all_results:
        for reg in r.get('circuit_regimes', []):
            regime_layers[reg['dominant_layer']] += 1
    summary['circuit_regimes'][0]['regime_distribution'] = {
        str(k): v for k, v in sorted(regime_layers.items())
    }

    # Save detailed data
    np.savez_compressed(
        model_dir / "dynamic_circuit_data.npz",
        mean_per_token=mean_per_token,
        overall_importance=overall_importance,
        combined_deltas=combined_deltas,
        combined_layer_dom=combined_layer_dom,
    )

    with open(model_dir / "dynamic_circuit_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"  → {model_dir}/")
    print(f"    Top heads: L{top_heads[0][0]}H{top_heads[0][1]} ({top_heads[0][2]:.4f})")
    print(f"    Circuit regimes: {dict(sorted(regime_layers.items())[:5])}")

    del model, processor
    torch.cuda.empty_cache()
    return summary


# ═══════════════════════════════════════════════════════════════════════════
#  Mode: Encoding vs Arbitration
# ═══════════════════════════════════════════════════════════════════════════

def run_encoding_arbitration(model_key: str, num_samples: int,
                              output_dir: Path, image_path: str = None):
    """Run encoding vs arbitration decomposition."""
    cfg = MODELS[model_key]
    model, processor, gen_cls, _ = load_model_and_generator(model_key)

    model_dir = output_dir / model_key / "encoding_arbitration"
    model_dir.mkdir(parents=True, exist_ok=True)

    if image_path:
        paths = [image_path]
    else:
        img_files = sorted(os.listdir(COCO_VAL2014))[:num_samples]
        paths = [os.path.join(COCO_VAL2014, f) for f in img_files]

    all_classifications = []
    all_summaries = []

    for path in tqdm(paths, desc=f"  [{cfg['name']}] Encoding/Arbitration"):
        try:
            result = encoding_vs_arbitration_decomposition(
                model=model, processor=processor,
                generator_class=gen_cls,
                image=path,
                prompt="Please describe this image in detail.",
                num_layers=cfg["num_layers"],
                num_heads=cfg["num_heads"],
                max_new_tokens=64,
            )
            if result:
                result['image'] = os.path.basename(path)
                all_classifications.extend(result['per_token_classification'])
                all_summaries.append(result['summary'])
        except Exception as e:
            print(f"    [WARN] {path}: {e}")
            continue

    # Aggregate
    encoding_rate = np.mean([s['encoding_failure_rate'] for s in all_summaries])
    arbitration_rate = np.mean([s['arbitration_failure_rate'] for s in all_summaries])
    grounded_rate = np.mean([s['grounded_rate'] for s in all_summaries])

    summary = {
        'model': cfg['name'],
        'num_samples': len(all_summaries),
        'mean_encoding_failure_rate': float(encoding_rate),
        'mean_arbitration_failure_rate': float(arbitration_rate),
        'mean_grounded_rate': float(grounded_rate),
        'total_tokens_analyzed': len(all_classifications),
    }

    with open(model_dir / "encoding_arbitration_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    with open(model_dir / "per_token_classifications.jsonl", 'w') as f:
        for c in all_classifications:
            json.dump(c, f)
            f.write('\n')

    print(f"  → {model_dir}/")
    print(f"    Encoding failure: {encoding_rate*100:.1f}%")
    print(f"    Arbitration failure: {arbitration_rate*100:.1f}%")
    print(f"    Grounded: {grounded_rate*100:.1f}%")

    del model, processor
    torch.cuda.empty_cache()
    return summary


# ═══════════════════════════════════════════════════════════════════════════
#  Mode: Full (all three analyses + cross-architecture)
# ═══════════════════════════════════════════════════════════════════════════

def run_full_pipeline(model_keys: list, num_samples: int, output_dir: Path):
    """Run all three analyses across all models and compare."""
    all_dynamic = {}
    all_encoding = {}

    for key in model_keys:
        if key in ("instructblip", "minicpm-v2.6"):
            print(f"\n  Skipping {key} (custom loading required — run separately)")
            continue

        try:
            print(f"\n{'='*60}")
            print(f"  {MODELS[key]['name']}")
            print(f"{'='*60}")

            # Dynamic circuit
            dyn = run_dynamic_discovery(key, num_samples, output_dir)
            if dyn:
                all_dynamic[key] = {
                    'data': dyn,
                    'projector_type': MODELS[key]['projector'],
                }

            # Encoding vs arbitration
            enc = run_encoding_arbitration(key, num_samples, output_dir)
            if enc:
                all_encoding[key] = enc

        except Exception as e:
            print(f"  [ERROR] {key}: {e}")
            import traceback
            traceback.print_exc()

    # Cross-architecture comparison
    if len(all_dynamic) >= 2:
        print("\n" + "=" * 60)
        print("  Cross-Architecture Comparison")
        print("=" * 60)

        cross_input = {}
        for key, val in all_dynamic.items():
            dyn_data = val['data']
            # Reconstruct head deltas from saved data
            data_path = output_dir / key / "dynamic" / "dynamic_circuit_data.npz"
            if data_path.exists():
                loaded = np.load(data_path)
                cross_input[MODELS[key]['name']] = {
                    'per_token_head_deltas': loaded['mean_per_token'],
                    'num_layers': MODELS[key]['num_layers'],
                    'num_heads': MODELS[key]['num_heads'],
                    'projector_type': MODELS[key]['projector'],
                }

        cross_result = cross_architecture_comparison(cross_input)

        cross_dir = output_dir / "cross_architecture"
        cross_dir.mkdir(parents=True, exist_ok=True)

        with open(cross_dir / "cross_architecture.json", 'w') as f:
            # Convert numpy values for JSON serialization
            serializable = {
                'universal_heads': cross_result.get('universal_heads', []),
                'model_names': cross_result.get('model_names', []),
                'circuit_similarity': cross_result.get('circuit_similarity', []),
            }
            json.dump(serializable, f, indent=2)

        print(f"  Universal heads found: {len(cross_result.get('universal_heads', []))}")
        if cross_result.get('circuit_similarity'):
            sim = np.array(cross_result['circuit_similarity'])
            names = cross_result.get('model_names', [])
            print(f"  Pairwise circuit similarity matrix:")
            for i, n1 in enumerate(names):
                for j, n2 in enumerate(names):
                    if i < j:
                        print(f"    {n1} ↔ {n2}: {sim[i,j]:.3f}")


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Mechanistic Attribution v2")
    parser.add_argument("--model", type=str, default="qwen2.5-vl",
                        choices=["all"] + list(MODELS.keys()),
                        help="Model to analyze")
    parser.add_argument("--mode", type=str, default="dynamic",
                        choices=["dynamic", "encoding", "cross", "full"],
                        help="Analysis mode")
    parser.add_argument("--num_samples", type=int, default=10,
                        help="Number of images to analyze")
    parser.add_argument("--image_path", type=str, default=None,
                        help="Single image for encoding mode (overrides num_samples)")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    ensure_output_dirs()

    output_dir = Path(args.output_dir) if args.output_dir else (
        REPO_ROOT / "results" / "attribution_v2"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Mechanistic Attribution v2")
    print(f"  Mode: {args.mode}  |  Model: {args.model}")
    print(f"  Samples: {args.num_samples}")
    print("=" * 60)

    if args.model == "all" and args.mode == "full":
        model_keys = ["llava-1.5", "qwen2.5-vl", "internvl3.5"]
        run_full_pipeline(model_keys, args.num_samples, output_dir)
    elif args.mode == "dynamic":
        run_dynamic_discovery(args.model, args.num_samples, output_dir)
    elif args.mode == "encoding":
        run_encoding_arbitration(args.model, args.num_samples, output_dir, args.image_path)
    elif args.mode == "cross":
        # Load pre-computed dynamic results
        cross_input = {}
        for key in MODELS:
            data_path = output_dir / key / "dynamic" / "dynamic_circuit_data.npz"
            if data_path.exists():
                loaded = np.load(data_path)
                cross_input[MODELS[key]['name']] = {
                    'per_token_head_deltas': loaded['mean_per_token'],
                    'num_layers': MODELS[key]['num_layers'],
                    'num_heads': MODELS[key]['num_heads'],
                    'projector_type': MODELS[key]['projector'],
                }
        if len(cross_input) >= 2:
            result = cross_architecture_comparison(cross_input)
            cross_dir = output_dir / "cross_architecture"
            cross_dir.mkdir(parents=True, exist_ok=True)
            with open(cross_dir / "cross_architecture.json", 'w') as f:
                json.dump({
                    'universal_heads': result.get('universal_heads', []),
                    'model_names': result.get('model_names', []),
                    'circuit_similarity': result.get('circuit_similarity', []),
                }, f, indent=2)
            print(f"  Universal heads: {len(result.get('universal_heads', []))}")
        else:
            print("  Need at least 2 models with pre-computed data. Run --mode dynamic first.")

    print("\nDone.")


if __name__ == "__main__":
    main()
