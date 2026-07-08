#!/usr/bin/env python3
"""
Experiment 2: Taxonomy-Guided Intervention Comparison (v2 — Weight Modification)
=================================================================================
Compare three intervention strategies based on the encoding-arbitration taxonomy.
Uses DIRECT WEIGHT MODIFICATION of o_proj layers rather than forward hooks,
ensuring α=1.0 baseline is mathematically identical to no-intervention.

Key insight: in MHA, head h contributes to o_proj via columns [h*hd:(h+1)*hd].
Scaling those columns by α is equivalent to scaling head h's output contribution.

  1. ENCODING BOOST:   scale o_proj columns for encoding-regime heads by α > 1
  2. ARBITRATION SUPPRESS: scale o_proj columns for arbitration-regime heads by α < 1
  3. UNIFIED: scale all top-10 head columns by α > 1
  4. COMBINED: encoding boost + arbitration suppress simultaneously

Usage:
    python mechanistic_analysis/exp2_taxonomy_intervention.py --model llava-1.5 --num_images 200

Requires: GPU with ~24GB VRAM, COCO val2014 dataset.
"""

import argparse, os, sys, json, tempfile, copy
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from PIL import Image as PILImage
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from config import (
    LLAVA_15_7B_HF, QWEN25VL_7B, INTERNVL35_8B,
    COCO_VAL2014, COCO_VAL2014_ANNOTATIONS,
)
from mechanistic_analysis.run_attribution import load_model_and_generator

OUTPUT_DIR = REPO_ROOT / "results" / "paper4_revision" / "taxonomy_intervention"
FIG_DIR = OUTPUT_DIR / "figures"

MODELS = {
    "llava-1.5":   {"name": "LLaVA-1.5",   "num_layers": 32, "num_heads": 32,
                     "enc_cutoff": 8, "arb_dominant": True},
    "qwen2.5-vl":  {"name": "Qwen2.5-VL",  "num_layers": 28, "num_heads": 28,
                     "enc_cutoff": 7, "arb_dominant": True},
    "internvl3.5": {"name": "InternVL3.5",  "num_layers": 36, "num_heads": 32,
                     "enc_cutoff": 9, "arb_dominant": False},
}

# Per-model failure rates (from paper Table 1)
TAXONOMY_RATES = {
    "llava-1.5":   {"enc": 0.139, "arb": 0.860, "grd": 0.001},
    "qwen2.5-vl":  {"enc": 0.114, "arb": 0.876, "grd": 0.010},
    "internvl3.5": {"enc": 0.297, "arb": 0.306, "grd": 0.397},
}

# For the single alpha sweep this is the cleanup:
#   encoding_boost  at baseline only → one unified baseline (no hooks) + α=2.0 weight-modified
#   arbitration_suppress   → common baseline + α=0.2 weight-modified
#   unified         → common baseline + α=2.0 weight-modified
#   combined         → common baseline + (2.0,0.2) weight-modified
#   random_control     → common baseline + random heads at α=2.0

INTERVENTION_SPECS_ORDERED = [
    ("encoding_boost",      "Encoding Boost",       1.0, 1.0),
    ("encoding_boost",      "Encoding Boost",       2.0, 1.0),
    ("arbitration_suppress","Arbitration Suppress", 1.0, 1.0),
    ("arbitration_suppress","Arbitration Suppress", 1.0, 0.2),
    ("unified",             "Unified",              1.0, 1.0),
    ("unified",             "Unified",              2.0, 1.0),
    ("combined",            "Combined",             1.0, 1.0),
    ("combined",            "Combined",             2.0, 0.2),
    ("random_control",      "Random Control",       1.0, 1.0),
    ("random_control",      "Random Control",       2.0, 1.0),
]


def ensure_dirs():
    for d in [OUTPUT_DIR, FIG_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def load_cached_top_heads(model_key: str) -> dict:
    sum_path = (REPO_ROOT / "results" / "attribution_v2" / model_key /
                "dynamic" / "dynamic_circuit_summary.json")
    if not sum_path.exists():
        print(f"  [WARN] No cached attribution data for {model_key}")
        return None
    with open(sum_path) as f:
        data = json.load(f)
    top_20 = data.get('top_20_heads', [])
    if len(top_20) < 10:
        print(f"  [WARN] Only {len(top_20)} heads in top_20")
        return None
    cfg = MODELS[model_key]
    cutoff = cfg["enc_cutoff"]
    enc_heads = [(l, h, s) for l, h, s in top_20 if l < cutoff]
    arb_heads = [(l, h, s) for l, h, s in top_20 if l >= cutoff]
    print(f"  Top-20 heads: enc={len(enc_heads)}, arb={len(arb_heads)}")
    return {
        'top_20': top_20,
        'encoding_heads': [(l, h) for l, h, _ in enc_heads[:10]],
        'arbitration_heads': [(l, h) for l, h, _ in arb_heads[:10]],
        'all_top10': [(l, h) for l, h, _ in top_20[:10]],
    }


def find_o_proj_modules(model) -> dict:
    """Map layer_idx -> (o_proj_module, num_heads)."""
    layer_to_module = {}
    for name, module in model.named_modules():
        if ('self_attn.o_proj' in name or 'self_attn.wo' in name or
            'attention.o_proj' in name) and 'vision' not in name.lower():
            parts = name.split('.')
            for i, p in enumerate(parts):
                if p in ('layers', 'layer', 'model.layers',
                         'language_model.model.layers') and i + 1 < len(parts):
                    try:
                        lidx = int(parts[i + 1])
                        layer_to_module[lidx] = module
                        break
                    except ValueError:
                        pass
    return layer_to_module


def get_head_dim_from_o_proj(o_proj_module, nh: int) -> int:
    """Head dim = in_features / num_heads.
    o_proj.weight shape is (out_features, in_features) = (hidden_dim, nh * hd).
    So hd = in_features // nh = weight.shape[1] // nh.
    """
    return o_proj_module.weight.shape[1] // nh


def scale_head_columns(o_proj_module, head_idx: int, alpha: float,
                       head_dim: int, nh: int):
    """
    Scale columns [head_idx*head_dim : (head_idx+1)*head_dim] of o_proj.weight
    and corresponding bias entries by `alpha / old_alpha`.

    o_proj maps (nh * hd) → hidden_dim, so its input dimension is nh * hd.
    head h corresponds to columns [h*hd : (h+1)*hd] of weight.

    At α=1.0 this is a no-op: scale factor = 1.0 / 1.0 = 1.0.
    """
    # o_proj.weight shape: (hidden_dim, nh * hd)
    start = head_idx * head_dim
    end = start + head_dim
    o_proj_module.weight.data[:, start:end] *= alpha
    if o_proj_module.bias is not None:
        # Actually o_proj bias is output-side (hidden_dim), not input-side.
        # Scaling head columns in weight handles the linear combination.
        # No bias adjustment needed for a per-head scaling.
        pass


class WeightRestorer:
    """Context manager that saves o_proj weights and restores on exit."""
    def __init__(self, modules):
        self.modules = modules
        self.backups = {}

    def __enter__(self):
        for lidx, mod in self.modules.items():
            self.backups[lidx] = mod.weight.data.clone()
        return self

    def __exit__(self, *args):
        for lidx, mod in self.modules.items():
            if lidx in self.backups:
                mod.weight.data.copy_(self.backups[lidx])


def run_model(model_key: str, num_images: int = 200):
    """Main experiment."""
    ensure_dirs()
    cfg = MODELS[model_key]
    nh = cfg["num_heads"]
    rates = TAXONOMY_RATES[model_key]

    print(f"\n{'='*70}")
    print(f"  Experiment 2 v2: Taxonomy-Guided Intervention — {cfg['name']}")
    print(f"  Method: DIRECT WEIGHT MODIFICATION (no forward hooks)")
    print(f"  Taxonomy: enc={rates['enc']*100:.0f}% arb={rates['arb']*100:.0f}% grd={rates['grd']*100:.0f}%")
    print(f"{'='*70}")

    # ── Load heads ──
    head_data = load_cached_top_heads(model_key)
    if not head_data:
        return None

    # ── Build random control heads ──
    rng = np.random.RandomState(42)
    all_possible = [(l, h) for l in range(cfg['num_layers'])
                    for h in range(cfg['num_heads'])]
    random_10 = [all_possible[i] for i in rng.choice(len(all_possible), 10, replace=False)]
    print(f"  Random control heads: {[f'L{l}H{h}' for l,h in random_10[:5]]}...")

    # ── Load model ──
    model, processor, gen_cls, _ = load_model_and_generator(model_key)
    layer_to_module = find_o_proj_modules(model)
    if len(layer_to_module) == 0:
        print("  [ERROR] No o_proj modules found")
        return None

    hd = get_head_dim_from_o_proj(list(layer_to_module.values())[0], nh)
    print(f"  Head dim: {hd}, o_proj input dim: {nh * hd}")

    # ── Images ──
    img_files_all = sorted(os.listdir(COCO_VAL2014))
    img_files = [os.path.join(COCO_VAL2014, f)
                 for f in img_files_all[:num_images]]
    print(f"  Images: {len(img_files)}")

    # ── CHAIR evaluator ──
    coco_ann_path = COCO_VAL2014_ANNOTATIONS
    if not coco_ann_path or not Path(coco_ann_path).exists():
        print("  [SKIP] COCO annotations not available")
        return None
    coco_annot_dir = str(Path(coco_ann_path).parent)
    from common_utils.chair_eval import CHAIR as CHAIREvaluator
    chair_eval = CHAIREvaluator(coco_annot_dir)

    # ── Shared baseline: ONE generation pass with NO weight modifications ──
    print(f"\n  ── Shared Baseline (no weight modification) ──")
    generator = gen_cls(model=model, processor=processor)
    prompt = "Please describe this image in detail."
    baseline_captions = []
    for img_path in tqdm(img_files, desc="    Baseline generation"):
        try:
            pil_img = PILImage.open(img_path).convert("RGB")
            outputs = generator.generate(
                image=pil_img, prompt=prompt,
                max_new_tokens=64, num_beams=1, do_sample=False, use_cache=False,
            )
            decoded = processor.decode(outputs.sequences[0], skip_special_tokens=True)
            if prompt in decoded:
                decoded = decoded.split(prompt)[-1].strip()
            baseline_captions.append(decoded)
        except Exception as e:
            print(f"      [WARN] {os.path.basename(img_path)}: {e}")
            baseline_captions.append("")

    # Compute baseline CHAIR
    base_list = []
    for img_path, cap in zip(img_files, baseline_captions):
        if not cap or len(cap.strip()) < 3:
            continue
        img_name = Path(img_path).name
        try:
            img_id = int(img_name.split('_')[-1].replace('.jpg', '').replace('.jpeg', ''))
        except ValueError:
            continue
        base_list.append({'image_id': img_id, 'caption': cap.strip()})

    baseline_chairs = baseline_chairi = 0.0
    baseline_n = len(base_list)
    if baseline_n >= 10:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tf:
            json.dump(base_list, tf)
            tmp_path = tf.name
        chair_out = chair_eval.compute_chair(tmp_path,
                                              image_id_key='image_id',
                                              caption_key='caption')
        os.unlink(tmp_path)
        baseline_chairs = float(chair_out.get('overall_metrics', {}).get('CHAIRs', 0))
        baseline_chairi = float(chair_out.get('overall_metrics', {}).get('CHAIRi', 0))

    print(f"    Baseline: CHAIRs={baseline_chairs*100:.1f}%  CHAIRi={baseline_chairi*100:.1f}%  "
          f"(N={baseline_n})")

    # ── Function to generate with weight-modified model ──
    def generate_with_weight_mod(enc_heads_list, enc_alpha, arb_heads_list, arb_alpha):
        """Modify o_proj weights, generate captions, restore weights."""
        results = []

        # Collect ALL o_proj modules we need to touch
        all_modules = set()
        head_scales = defaultdict(list)  # layer_idx -> [(head_idx, scale_factor)]

        for l, h in enc_heads_list:
            if l in layer_to_module:
                all_modules.add(l)
                head_scales[l].append((h, enc_alpha))
        for l, h in arb_heads_list:
            if l in layer_to_module:
                all_modules.add(l)
                head_scales[l].append((h, arb_alpha))

        # Save and modify weights
        backups = {}
        try:
            for lidx in all_modules:
                mod = layer_to_module[lidx]
                backups[lidx] = mod.weight.data.clone()
                for head_idx, scale in head_scales[lidx]:
                    start = head_idx * hd
                    end = start + hd
                    mod.weight.data[:, start:end] *= scale

            # Generate
            for img_path in tqdm(img_files, desc=f"      Gen (enc={enc_alpha}, arb={arb_alpha})"):
                try:
                    pil_img = PILImage.open(img_path).convert("RGB")
                    outputs = generator.generate(
                        image=pil_img, prompt=prompt,
                        max_new_tokens=64, num_beams=1, do_sample=False, use_cache=False,
                    )
                    decoded = processor.decode(outputs.sequences[0], skip_special_tokens=True)
                    if prompt in decoded:
                        decoded = decoded.split(prompt)[-1].strip()
                    results.append(decoded)
                except Exception as e:
                    results.append("")
        finally:
            # Restore
            for lidx in backups:
                layer_to_module[lidx].weight.data.copy_(backups[lidx])

        return results

    # ── Run all intervention specs ──
    all_records = []

    for sname, slabel, enc_alpha, arb_alpha in INTERVENTION_SPECS_ORDERED:
        # Determine which heads to target
        if sname == "random_control":
            enc_heads_list = random_10
            arb_heads_list = []
        elif sname == "encoding_boost":
            enc_heads_list = head_data['encoding_heads']
            arb_heads_list = []
        elif sname == "arbitration_suppress":
            enc_heads_list = []
            arb_heads_list = head_data['arbitration_heads']
        elif sname == "combined":
            enc_heads_list = head_data['encoding_heads']
            arb_heads_list = head_data['arbitration_heads']
        else:  # unified
            enc_heads_list = head_data['all_top10']
            arb_heads_list = []

        # At alpha=1.0, use shared baseline (already computed, no weight mod needed)
        if abs(enc_alpha - 1.0) < 1e-6 and abs(arb_alpha - 1.0) < 1e-6:
            chair_s, chair_i = baseline_chairs, baseline_chairi
            label = f"{sname}_baseline"
            print(f"\n    [{slabel}] baseline (shared) → CHAIRs={chair_s*100:.1f}%  "
                  f"CHAIRi={chair_i*100:.1f}%")
        else:
            label = f"{sname}_enc{enc_alpha}_arb{arb_alpha}"
            print(f"\n    [{slabel}] enc_α={enc_alpha} arb_α={arb_alpha}")

            captions = generate_with_weight_mod(
                enc_heads_list, enc_alpha, arb_heads_list, arb_alpha)

            # Compute CHAIR
            valid = []
            for img_path, cap in zip(img_files, captions):
                if not cap or len(cap.strip()) < 3:
                    continue
                img_name = Path(img_path).name
                try:
                    img_id = int(img_name.split('_')[-1].replace('.jpg', '').replace('.jpeg', ''))
                except ValueError:
                    continue
                valid.append({'image_id': img_id, 'caption': cap.strip()})

            if len(valid) >= 5:
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tf:
                    json.dump(valid, tf)
                    tmp_path = tf.name
                chair_out = chair_eval.compute_chair(tmp_path,
                                                      image_id_key='image_id',
                                                      caption_key='caption')
                os.unlink(tmp_path)
                chair_s = float(chair_out.get('overall_metrics', {}).get('CHAIRs', 0))
                chair_i = float(chair_out.get('overall_metrics', {}).get('CHAIRi', 0))
                n_valid = len(valid)
            else:
                chair_s = chair_i = None
                n_valid = len(valid)

            print(f"      CHAIRs={chair_s*100:.1f}%  CHAIRi={chair_i*100:.1f}%  "
                  f"(N={n_valid})" if chair_s is not None else f"      N={n_valid} (too few)")

        all_records.append({
            'strategy': sname,
            'label_strategy': slabel,
            'run_label': label,
            'enc_alpha': enc_alpha,
            'arb_alpha': arb_alpha,
            'CHAIRs': chair_s,
            'CHAIRi': chair_i,
            'N_valid': baseline_n,
        })

    del model, processor
    torch.cuda.empty_cache()

    # ── Summary ──
    # Group by strategy, compute reduction vs this strategy's own baseline
    strategy_results = {}
    for sname in set(r['strategy'] for r in all_records):
        s_records = [r for r in all_records if r['strategy'] == sname]
        baseline_r = next((r for r in s_records
                          if abs(r['enc_alpha'] - 1.0) < 1e-6
                          and abs(r['arb_alpha'] - 1.0) < 1e-6), None)
        inter_r = next((r for r in s_records
                       if not (abs(r['enc_alpha'] - 1.0) < 1e-6
                               and abs(r['arb_alpha'] - 1.0) < 1e-6)), None)

        if baseline_r and inter_r and baseline_r['CHAIRs'] is not None and inter_r['CHAIRs'] is not None:
            reduction = (baseline_r['CHAIRs'] - inter_r['CHAIRs']) * 100
            strategy_results[sname] = {
                'baseline_CHAIRs': baseline_r['CHAIRs'],
                'intervention_CHAIRs': inter_r['CHAIRs'],
                'baseline_CHAIRi': baseline_r['CHAIRi'],
                'intervention_CHAIRi': inter_r['CHAIRi'],
                'reduction_pp': reduction,
                'enc_alpha': inter_r['enc_alpha'],
                'arb_alpha': inter_r['arb_alpha'],
            }
        else:
            strategy_results[sname] = {'status': 'insufficient_data'}

    print(f"\n  {'='*65}")
    print(f"  RESULTS — {cfg['name']}")
    print(f"  Shared baseline: CHAIRs={baseline_chairs*100:.1f}%  CHAIRi={baseline_chairi*100:.1f}%")
    print(f"  {'Strategy':<25} {'α_enc':<8} {'α_arb':<8} {'CHAIRs':<10} {'Reduction':<12}")
    print(f"  {'─'*65}")
    # Print all unique strategies
    printed = set()
    for sname in set(r['strategy'] for r in all_records):
        sr = strategy_results.get(sname, {})
        if 'reduction_pp' in sr:
            slabel = next(r['label_strategy'] for r in all_records if r['strategy'] == sname)
            print(f"  {slabel:<25} {sr['enc_alpha']:<8.1f} {sr['arb_alpha']:<8.1f} "
                  f"{sr['intervention_CHAIRs']*100:<5.1f}%     {sr['reduction_pp']:+.1f} pp")

    # Check prediction
    arb_red = strategy_results.get('arbitration_suppress', {}).get('reduction_pp', 0)
    enc_red = strategy_results.get('encoding_boost', {}).get('reduction_pp', 0)
    unif_red = strategy_results.get('unified', {}).get('reduction_pp', 0)
    rand_red = strategy_results.get('random_control', {}).get('reduction_pp', 0)
    combined_red = strategy_results.get('combined', {}).get('reduction_pp', 0)

    if cfg['arb_dominant']:
        pred_confirmed = arb_red > enc_red
    else:
        pred_confirmed = combined_red > max(enc_red, arb_red)

    print(f"\n  TAXONOMY CHECK: arb_suppress={arb_red:+.1f}pp, enc_boost={enc_red:+.1f}pp, "
          f"unified={unif_red:+.1f}pp, random={rand_red:+.1f}pp, combined={combined_red:+.1f}pp")
    print(f"  Prediction {'✓ CONFIRMED' if pred_confirmed else '✗ NOT confirmed'}")

    # Save
    summary = {
        'model': cfg['name'],
        'method': 'weight_modification',
        'num_images': num_images,
        'shared_baseline_CHAIRs': baseline_chairs,
        'shared_baseline_CHAIRi': baseline_chairi,
        'taxonomy_rates': rates,
        'arbitration_dominant': cfg['arb_dominant'],
        'all_records': all_records,
        'strategy_results': {k: {kk: vv for kk, vv in v.items() if not isinstance(vv, (np.ndarray,))}
                            for k, v in strategy_results.items()},
        'prediction_confirmed': pred_confirmed,
        'head_counts': {
            'encoding': len(head_data['encoding_heads']),
            'arbitration': len(head_data['arbitration_heads']),
            'random': len(random_10),
        },
    }

    out_path = OUTPUT_DIR / f"taxonomy_intervention_{model_key}.json"
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  → {out_path}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Exp 2 v2: Taxonomy-Guided Intervention")
    parser.add_argument('--model', default='llava-1.5',
                       choices=['llava-1.5', 'qwen2.5-vl', 'internvl3.5', 'all'])
    parser.add_argument('--num_images', type=int, default=200)
    args = parser.parse_args()

    ensure_dirs()
    models_to_run = list(MODELS.keys()) if args.model == 'all' else [args.model]

    for mkey in models_to_run:
        try:
            run_model(mkey, args.num_images)
        except Exception as e:
            print(f"  [FATAL] {mkey}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nDone. Results: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
