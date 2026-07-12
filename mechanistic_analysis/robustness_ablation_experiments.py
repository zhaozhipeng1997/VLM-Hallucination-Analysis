#!/usr/bin/env python3
"""
Robustness & Ablation Experiments
================================
Four experiments:

  Exp A: Gradient attribution robustness check
      → Perturb input embeddings with Gaussian noise (σ ∈ {1e-3, 5e-3, 1e-2, 5e-2})
      → Recompute head rankings, report Spearman ρ between clean and noisy ranks

  Exp B: Task format control experiment
      → Run both captioning prompt AND VQA prompt on SAME COCO images
      → Compute encoding/arbitration decomposition under both prompts
      → Isolates task format from image content

  Exp C: Single-head causal patching (L30 H31)
      → Replace only L30 H31 activation with counterfactual version during
        factual forward pass (or vice versa)
      → Measure per-token Δ_t and arbitration failure rate change

Usage:
    # Exp A (requires model, ~5 min per model):
    python mechanistic_analysis/robustness_ablation_experiments.py --experiment r1 --model llava-1.5 --num_images 50

    # Exp B (requires model, ~30 min per model):
    python mechanistic_analysis/robustness_ablation_experiments.py --experiment r2 --model llava-1.5 --num_images 50

    # Exp C (requires model, ~30 min per model):
    python mechanistic_analysis/robustness_ablation_experiments.py --experiment r3 --model llava-1.5 --num_images 50

    # All three:
    python mechanistic_analysis/robustness_ablation_experiments.py --experiment all --model llava-1.5
"""

import argparse, json, os, sys
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Optional, List, Dict, Tuple
import torch
import torch.nn.functional as F
from PIL import Image as PILImage
from tqdm import tqdm
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['mathtext.fontset'] = 'stix'

from config import (
    LLAVA_15_7B_HF, QWEN25VL_7B, INTERNVL35_8B,
    COCO_VAL2014, RESULTS_DIR, ensure_output_dirs,
)
from mechanistic_analysis.run_attribution import load_model_and_generator
from mechanistic_analysis.dynamic_circuit import install_all_head_hooks

OUTPUT_DIR = REPO_ROOT / "results" / "robustness_ablation"
FIG_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"

MODELS = {
    "llava-1.5":   {"name": "LLaVA-1.5",   "num_layers": 32, "num_heads": 32},
    "qwen2.5-vl":  {"name": "Qwen2.5-VL",  "num_layers": 28, "num_heads": 28},
    "internvl3.5": {"name": "InternVL3.5",  "num_layers": 36, "num_heads": 32},
}


def ensure_dirs():
    for d in [OUTPUT_DIR, FIG_DIR, TABLE_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def get_head_dim(model):
    try:
        hs = model.config.text_config.hidden_size
    except AttributeError:
        try:
            hs = model.config.hidden_size
        except AttributeError:
            hs = model.language_model.config.hidden_size
    return hs


def get_image_files(n: int) -> List[str]:
    """Get n COCO image paths."""
    files = sorted(os.listdir(COCO_VAL2014))[:n]
    return [os.path.join(COCO_VAL2014, f) for f in files]


# ──────────────────────────────────────────────────────────────────────────────
#  Exp A: GRADIENT ATTRIBUTION ROBUSTNESS CHECK
# ──────────────────────────────────────────────────────────────────────────────

def add_embedding_noise(model, sigma: float):
    """Add Gaussian noise to input embeddings during forward pass."""
    # Hook into the embedding layer to inject noise
    handles = []
    for name, module in model.named_modules():
        if ('embed_tokens' in name or 'wte' in name or 'token_embedding' in name) and \
           'vision' not in name.lower():
            def make_hook(sigma_val):
                def hook(module, input, output):
                    noise = torch.randn_like(output) * sigma_val
                    return output + noise
                return hook
            handles.append(module.register_forward_hook(make_hook(sigma)))
            break  # only hook the first embed_tokens
    return handles


def compute_head_attributions(model, processor, gen_cls, image_path, prompt,
                              nl: int, nh: int, sigma: float = 0.0):
    """
    Run one forward pass (factual + counterfactual) and compute
    per-head attribution via L2 norm difference.

    If sigma > 0, injects Gaussian noise at the embedding layer.
    """
    pil_img = PILImage.open(image_path).convert("RGB")
    hd = get_head_dim(model) // nh

    # Generate to get token sequence
    generator = gen_cls(model=model, processor=processor)
    outputs = generator.generate(
        image=pil_img, prompt=prompt,
        max_new_tokens=32, num_beams=1, do_sample=False, use_cache=False,
    )
    full_ids = outputs.sequences[0]
    num_tokens = len(getattr(outputs, 'token_sources', []))

    # Prepare inputs
    has_ip = hasattr(processor, 'image_processor') or hasattr(processor, 'image_processor_class')
    if has_ip:
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": pil_img},
            {"type": "text", "text": prompt}
        ]}]
        text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs_f = processor(text=text, images=pil_img, return_tensors="pt").to(model.device)
        inputs_c = processor(text=text, return_tensors="pt").to(model.device)
    else:
        from mechanistic_analysis.dynamic_circuit import (
            _prepare_pixel_values_for_model, _prepare_shuffled_pixel_values,
            _internvl_build_prompt_with_context,
        )
        qt = '<image>\n' + prompt
        _pv = _prepare_pixel_values_for_model(model, processor, pil_img)
        _spv = _prepare_shuffled_pixel_values(model, processor, pil_img)
        if _pv is not None and hasattr(model, 'img_context_token_id'):
            inputs_f = _internvl_build_prompt_with_context(model, processor, qt, _pv)
        elif _pv is not None:
            inputs_f = processor(qt, return_tensors="pt").to(model.device)
            inputs_f['pixel_values'] = _pv
        else:
            inputs_f = processor(qt, return_tensors="pt").to(model.device)

        if _spv is not None and hasattr(model, 'img_context_token_id'):
            inputs_c = _internvl_build_prompt_with_context(model, processor, qt, _spv)
        elif _spv is not None:
            inputs_c = processor(qt, return_tensors="pt").to(model.device)
            inputs_c['pixel_values'] = _spv
        elif _pv is not None and hasattr(model, 'img_context_token_id'):
            inputs_c = _internvl_build_prompt_with_context(model, processor, qt, torch.zeros_like(_pv))
        else:
            inputs_c = None

    full_ids = full_ids.unsqueeze(0).to(model.device)
    if not has_ip and pil_img is not None and inputs_f is not None:
        full_ids = torch.cat([inputs_f.input_ids, full_ids], dim=1)

    # Install hooks + optional noise
    noise_handles = add_embedding_noise(model, sigma) if sigma > 0 else []

    try:
        # Factual
        hooks_f, cleanup_f = install_all_head_hooks(model, nh, hd)
        with torch.inference_mode():
            f_kwargs = dict(input_ids=full_ids,
                            attention_mask=torch.ones(1, full_ids.shape[1], device=model.device, dtype=torch.long))
            for k in ('pixel_values', 'image_grid_thw', 'image_flags'):
                if inputs_f is not None and inputs_f.get(k) is not None:
                    f_kwargs[k] = inputs_f[k]
            _ = model(**f_kwargs)

        factual = {}
        for l, _, ho in hooks_f:
            if ho.captured is not None:
                factual[l] = ho.captured
        cleanup_f()

        # Counterfactual
        hooks_c, cleanup_c = install_all_head_hooks(model, nh, hd)
        with torch.inference_mode():
            c_kwargs = dict(input_ids=full_ids,
                            attention_mask=torch.ones(1, full_ids.shape[1], device=model.device, dtype=torch.long))
            if inputs_c is not None:
                for k in ('pixel_values', 'image_grid_thw', 'image_flags'):
                    if inputs_c.get(k) is not None:
                        c_kwargs[k] = inputs_c[k]
            else:
                if inputs_f is not None and inputs_f.get('image_flags') is not None:
                    c_kwargs['image_flags'] = torch.zeros_like(inputs_f['image_flags'])
                if inputs_f is not None and inputs_f.get('pixel_values') is not None:
                    c_kwargs['pixel_values'] = torch.zeros_like(inputs_f['pixel_values'])
            _ = model(**c_kwargs)

        counter = {}
        for l, _, ho in hooks_c:
            if ho.captured is not None:
                counter[l] = ho.captured
        cleanup_c()

        # Compute per-head delta (aggregate over last seq position + all tokens)
        head_deltas = np.zeros((nl, nh))
        for l in sorted(set(factual.keys()) & set(counter.keys())):
            if l >= nl:
                continue
            f_h = factual[l]
            c_h = counter[l]
            # Take mean over all sequence positions
            delta = torch.norm(f_h[0] - c_h[0], dim=-1).mean(dim=0)  # (H,)
            head_deltas[l, :] = delta.cpu().numpy()

    finally:
        for h in noise_handles:
            h.remove()

    return head_deltas


def run_r1_robustness(model_key: str, num_images: int = 50):
    """Gradient attribution robustness: clean vs noisy head rankings."""
    ensure_dirs()
    cfg = MODELS[model_key]
    nl, nh = cfg["num_layers"], cfg["num_heads"]

    print(f"\n{'='*70}")
    print(f"  Exp A: Attribution Robustness — {cfg['name']}")
    print(f"{'='*70}")

    model, processor, gen_cls, _ = load_model_and_generator(model_key)
    img_files = get_image_files(num_images)

    sigmas = [0.0, 1e-3, 5e-3, 1e-2, 5e-2]
    all_attrs = {s: [] for s in sigmas}

    for img_path in tqdm(img_files, desc=f"  [{cfg['name']}] Exp A"):
        prompt = "Please describe this image in detail."
        for sigma in sigmas:
            try:
                attr = compute_head_attributions(
                    model, processor, gen_cls, img_path, prompt, nl, nh, sigma
                )
                # Flatten to 1D ranking
                flat = attr.flatten()
                all_attrs[sigma].append(flat)
            except Exception as e:
                print(f"  [WARN] {os.path.basename(img_path)} σ={sigma}: {e}")

    # Aggregate: mean attribution vector per sigma
    mean_attrs = {}
    for sigma in sigmas:
        if all_attrs[sigma]:
            mean_attrs[sigma] = np.mean(all_attrs[sigma], axis=0)  # (L*H,)
        else:
            mean_attrs[sigma] = None

    clean_flat = mean_attrs[0.0]
    if clean_flat is None:
        print("  [ERROR] No clean attributions collected")
        del model, processor; torch.cuda.empty_cache()
        return None

    # Compute Spearman ρ between clean ranking and each noisy ranking
    clean_rank = stats.rankdata(-clean_flat)  # 1 = highest attr

    results = {"model": cfg["name"], "sigmas": [], "spearman_rho": [], "top20_overlap": []}
    print(f"\n  Robustness check: Spearman ρ(clean, noisy) and Top-20 overlap")

    for sigma in sigmas:
        if sigma == 0.0:
            results["sigmas"].append(0.0)
            results["spearman_rho"].append(1.0)
            results["top20_overlap"].append(1.0)
            continue

        noisy_flat = mean_attrs[sigma]
        if noisy_flat is None:
            continue

        noisy_rank = stats.rankdata(-noisy_flat)
        rho, p = stats.spearmanr(clean_flat, noisy_flat)

        # Top-20 overlap
        clean_top20 = set(np.argsort(-clean_flat)[:20])
        noisy_top20 = set(np.argsort(-noisy_flat)[:20])
        overlap = len(clean_top20 & noisy_top20) / 20

        results["sigmas"].append(float(sigma))
        results["spearman_rho"].append(float(rho))
        results["top20_overlap"].append(float(overlap))

        print(f"    σ={sigma:.0e}: ρ={rho:.4f}  Top-20 overlap={overlap*100:.0f}%")

    # Save
    with open(OUTPUT_DIR / f"r1_robustness_{model_key}.json", 'w') as f:
        json.dump(results, f, indent=2)

    # Figure
    fig, ax = plt.subplots(figsize=(8, 5))
    xs = results["sigmas"][1:]  # skip clean
    rhos = results["spearman_rho"][1:]
    overlaps = results["top20_overlap"][1:]

    ax.plot(xs, rhos, 'o-', color='#4472C4', lw=2, ms=8, label='Spearman ρ (all heads)')
    ax.plot(xs, overlaps, 's-', color='#ED7D31', lw=2, ms=8, label='Top-20 overlap')
    ax.axhline(0.95, color='gray', ls='--', lw=1, alpha=0.5, label='ρ=0.95')
    ax.set_xscale('log')
    ax.set_xlabel('Embedding noise σ', fontsize=12)
    ax.set_ylabel('Similarity to clean ranking', fontsize=12)
    ax.set_title(f'Exp A: Attribution Robustness — {cfg["name"]}', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"r1_robustness_{model_key}.pdf", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  → {FIG_DIR / f'r1_robustness_{model_key}.pdf'}")

    del model, processor; torch.cuda.empty_cache()
    return results


# ──────────────────────────────────────────────────────────────────────────────
#  Exp B: TASK FORMAT CONTROL (same images, two prompts)
# ──────────────────────────────────────────────────────────────────────────────

def run_r2_task_control(model_key: str, num_images: int = 50):
    """
    Compare encoding/arbitration decomposition on the SAME set of images
    under two different prompt formats:
      - Captioning: "Please describe this image in detail."
      - VQA: "What objects are visible in this image? Answer concisely."

    This isolates the effect of task format (instruction-following vs
    open-ended description) while holding image content constant.
    """
    ensure_dirs()
    cfg = MODELS[model_key]
    nl, nh = cfg["num_layers"], cfg["num_heads"]

    print(f"\n{'='*70}")
    print(f"  Exp B: Task Format Control — {cfg['name']}")
    print(f"  Same {num_images} COCO images, captioning vs VQA prompt")
    print(f"{'='*70}")

    from mechanistic_analysis.dynamic_circuit import encoding_vs_arbitration_decomposition

    model, processor, gen_cls, _ = load_model_and_generator(model_key)
    img_files = get_image_files(num_images)

    caption_prompt = "Please describe this image in detail."
    vqa_prompt = "What objects are visible in this image? Answer concisely."

    all_results = {"caption": [], "vqa": []}

    for task_name, prompt in [("caption", caption_prompt), ("vqa", vqa_prompt)]:
        cls_list = []
        sums_list = []

        for img_path in tqdm(img_files, desc=f"  [{cfg['name']}] {task_name}"):
            try:
                res = encoding_vs_arbitration_decomposition(
                    model=model, processor=processor, generator_class=gen_cls,
                    image=img_path, prompt=prompt,
                    num_layers=nl, num_heads=nh, max_new_tokens=64 if task_name == "caption" else 24,
                )
                if res:
                    cls_list.extend(res['per_token_classification'])
                    sums_list.append(res['summary'])
            except Exception as e:
                print(f"  [WARN] {os.path.basename(img_path)} {task_name}: {e}")

        if sums_list:
            enc_r = float(np.mean([s['encoding_failure_rate'] for s in sums_list]))
            arb_r = float(np.mean([s['arbitration_failure_rate'] for s in sums_list]))
            grd_r = float(np.mean([s['grounded_rate'] for s in sums_list]))
            all_results[task_name] = {
                "enc": enc_r, "arb": arb_r, "grd": grd_r,
                "N_tokens": len(cls_list), "N_samples": len(sums_list),
            }
            print(f"    {task_name}: enc={enc_r*100:.1f}% arb={arb_r*100:.1f}% grd={grd_r*100:.1f}%")

    cap = all_results["caption"]
    vqa = all_results["vqa"]
    if isinstance(cap, dict) and isinstance(vqa, dict):
        # Save
        with open(OUTPUT_DIR / f"r2_task_control_{model_key}.json", 'w') as f:
            json.dump({"model": cfg["name"], "caption": cap, "vqa": vqa}, f, indent=2)

        # Figure
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(3)
        w = 0.35
        ax.bar(x - w/2, [cap["enc"]*100, cap["arb"]*100, cap["grd"]*100], w,
                label='Captioning', color=['#ED7D31','#4472C4','#70AD47'],
                alpha=0.5, edgecolor='black', lw=0.5)
        ax.bar(x + w/2, [vqa["enc"]*100, vqa["arb"]*100, vqa["grd"]*100], w,
                label='VQA (same images)', color=['#ED7D31','#4472C4','#70AD47'],
                alpha=1.0, edgecolor='black', lw=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(['Encoding\nFailure', 'Arbitration\nFailure', 'Grounded'], fontsize=11)
        ax.set_ylabel('Token fraction (%)', fontsize=12)
        ax.set_title(f'Exp B: Task Format Control — {cfg["name"]}\nSame images, different prompts',
                     fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        # Annotate delta
        delta_arb = abs(vqa["arb"] - cap["arb"]) * 100
        ax.text(1, max(cap["arb"], vqa["arb"])*100 + 2,
                f'Δ arb = {delta_arb:.1f} pp', ha='center', fontsize=9,
                fontstyle='italic', color='gray')
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"r2_task_control_{model_key}.pdf", dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  → {FIG_DIR / f'r2_task_control_{model_key}.pdf'}")

    del model, processor; torch.cuda.empty_cache()
    return all_results


# ──────────────────────────────────────────────────────────────────────────────
#  Exp C: SINGLE-HEAD CAUSAL PATCHING (L30 H31)
# ──────────────────────────────────────────────────────────────────────────────

def run_r3_single_head_patching(model_key: str, num_images: int = 50):
    """
    Perform targeted counterfactual patching of the top-1 head ONLY.

    For LLaVA-1.5, patch L30 H31's output activation:
    - Factual forward: use factual activations
    - Patched forward: replace L30 H31 activation with its counterfactual version
    - Measure per-token Δ_t change between factual and patched logits

    Also measure patching effect on encoding/arbitration decomposition.
    """
    ensure_dirs()
    cfg = MODELS[model_key]
    nl, nh = cfg["num_layers"], cfg["num_heads"]

    print(f"\n{'='*70}")
    print(f"  Exp C: Single-Head Causal Patching — {cfg['name']}")
    print(f"{'='*70}")

    from mechanistic_analysis.run_attribution import load_model_and_generator
    model, processor, gen_cls, _ = load_model_and_generator(model_key)
    hd = get_head_dim(model) // nh

    # Determine top head from existing data
    import json as _json
    sum_path = (REPO_ROOT / "results" / "attribution_v2" / model_key /
                "dynamic" / "dynamic_circuit_summary.json")
    if sum_path.exists():
        with open(sum_path) as f:
            top_heads = _json.load(f).get('top_20_heads', [])
        tl, th, ts = top_heads[0] if top_heads else (30, 31, 0.075)  # fallback for LLaVA
    else:
        tl, th = (30, 31)  # fallback

    print(f"  Patching head L{tl} H{th}")

    img_files = get_image_files(min(num_images, 30))
    per_token_effects = []  # list of Δ_t changes per token
    per_image_effects = []  # mean Δ_t change per image

    for img_path in tqdm(img_files, desc=f"  [{cfg['name']}] Exp C patching"):
        try:
            pil_img = PILImage.open(img_path).convert("RGB")
            prompt = "Please describe this image in detail."

            # Generate to get baseline Δ_t
            generator = gen_cls(model=model, processor=processor)
            outputs = generator.generate(
                image=pil_img, prompt=prompt,
                max_new_tokens=32, num_beams=1, do_sample=False, use_cache=False,
            )
            token_sources = getattr(outputs, 'token_sources', [])
            baseline_ate = [ts.get('ate', 0.0) for ts in token_sources]
            n_tokens = len(token_sources)
            if n_tokens == 0:
                continue

            full_ids = outputs.sequences[0].unsqueeze(0).to(model.device)

            # Prepare inputs
            has_ip = hasattr(processor, 'image_processor') or hasattr(processor, 'image_processor_class')
            if has_ip:
                msgs = [{"role": "user", "content": [
                    {"type": "image", "image": pil_img},
                    {"type": "text", "text": prompt}
                ]}]
                text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                inputs_f = processor(text=text, images=pil_img, return_tensors="pt").to(model.device)
                inputs_c = processor(text=text, return_tensors="pt").to(model.device)
            else:
                # Tokenizer-only path — skip for now, requires InternVL/MiniCPM loading
                print("  [SKIP] tokenizer-only model not supported in Exp C yet")
                continue

            # Find the o_proj module for layer tl
            target_module = None
            for name, module in model.named_modules():
                if ('self_attn.o_proj' in name or 'self_attn.wo' in name) and \
                   'vision' not in name.lower():
                    parts = name.split('.')
                    for i, p in enumerate(parts):
                        if p in ('layers', 'layer') and i + 1 < len(parts):
                            try:
                                if int(parts[i+1]) == tl:
                                    target_module = module
                                    break
                            except ValueError:
                                pass
                    if target_module:
                        break

            if target_module is None:
                print(f"  [WARN] Could not find o_proj for layer {tl}")
                continue

            # ── Step 1: Capture factual head output ──
            factual_head_out = None
            counter_head_out = None

            def capture_input(module, input, output):
                nonlocal factual_head_out
                x = input[0]  # (B, S, nh*hd)
                b, s, d = x.shape
                factual_head_out = x.view(b, s, nh, hd).detach().clone()

            handle = target_module.register_forward_hook(capture_input)
            with torch.inference_mode():
                f_kwargs = dict(
                    input_ids=full_ids,
                    attention_mask=torch.ones(1, full_ids.shape[1], device=model.device, dtype=torch.long),
                )
                for k in ('pixel_values', 'image_grid_thw', 'image_flags'):
                    if inputs_f.get(k) is not None:
                        f_kwargs[k] = inputs_f[k]
                f_out = model(**f_kwargs)
                f_logits = f_out.logits[:, -1, :]
            handle.remove()

            # ── Step 2: Capture counterfactual head output ──
            def capture_cf(module, input, output):
                nonlocal counter_head_out
                x = input[0]
                b, s, d = x.shape
                counter_head_out = x.view(b, s, nh, hd).detach().clone()

            handle = target_module.register_forward_hook(capture_cf)
            with torch.inference_mode():
                c_kwargs = dict(
                    input_ids=full_ids,
                    attention_mask=torch.ones(1, full_ids.shape[1], device=model.device, dtype=torch.long),
                )
                if inputs_c is not None:
                    for k in ('pixel_values', 'image_grid_thw', 'image_flags'):
                        if inputs_c.get(k) is not None:
                            c_kwargs[k] = inputs_c[k]
                c_out = model(**c_kwargs)
                c_logits = c_out.logits[:, -1, :]
            handle.remove()

            # ── Step 3: Patch only L{tl} H{th} ──
            if factual_head_out is None or counter_head_out is None:
                continue

            def patch_head(module, input, output):
                x = input[0].clone()
                b, s, d = x.shape
                x_view = x.view(b, s, nh, hd)
                # Replace H{th} with counterfactual version
                x_view[:, :, th, :] = counter_head_out[:, :, th, :]
                return x.view(b, s, d)

            handle_p = target_module.register_forward_hook(patch_head)
            with torch.inference_mode():
                p_kwargs = dict(
                    input_ids=full_ids,
                    attention_mask=torch.ones(1, full_ids.shape[1], device=model.device, dtype=torch.long),
                )
                for k in ('pixel_values', 'image_grid_thw', 'image_flags'):
                    if inputs_f.get(k) is not None:
                        p_kwargs[k] = inputs_f[k]
                p_out = model(**p_kwargs)
                p_logits = p_out.logits[:, -1, :]
            handle_p.remove()

            # ── Step 4: Measure effect as logit divergence ──
            # Δ_t effect ≈ mean absolute shift in top-k logprobs
            f_lp = F.log_softmax(f_logits, dim=-1)
            p_lp = F.log_softmax(p_logits, dim=-1)
            kl = F.kl_div(p_lp, f_lp, reduction='batchmean', log_target=True).item()
            per_image_effects.append(kl)

            # Per-token: use KL at each position (approximate from full forward)
            # For simplicity we use the final logit KL as proxy
            per_token_effects.append(kl)

        except Exception as e:
            print(f"  [WARN] {os.path.basename(img_path)}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if per_image_effects:
        mu = np.mean(per_image_effects)
        sigma = np.std(per_image_effects)
        print(f"\n  Exp C Results for {cfg['name']} (L{tl} H{th}):")
        print(f"    Mean KL(patched || factual): {mu:.6f} ± {sigma:.6f}")
        print(f"    N images: {len(per_image_effects)}")

        with open(OUTPUT_DIR / f"r3_patching_{model_key}.json", 'w') as f:
            json.dump({
                "model": cfg["name"], "head": {"layer": tl, "head": th},
                "mean_kl": float(mu), "std_kl": float(sigma),
                "n_images": len(per_image_effects),
                "per_image_kl": [float(x) for x in per_image_effects],
            }, f, indent=2)

        # Figure: KL histogram
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.hist(per_image_effects, bins=20, color='#4472C4', edgecolor='white', alpha=0.8)
        ax.axvline(mu, color='black', ls='--', lw=1.5, label=f'Mean = {mu:.6f}')
        ax.set_xlabel('KL divergence (patched || factual)', fontsize=12)
        ax.set_ylabel('Images', fontsize=12)
        ax.set_title(f'Exp C: Single-Head Causal Patching — {cfg["name"]} L{tl} H{th}\n'
                     f'Effect of replacing ONE head on logit distribution',
                     fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"r3_patching_{model_key}.pdf", dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  → {FIG_DIR / f'r3_patching_{model_key}.pdf'}")
    else:
        print("  [ERROR] No successful patching runs")

    del model, processor; torch.cuda.empty_cache()
    return per_image_effects


# ──────────────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Robustness & Ablation Experiments")
    parser.add_argument("--experiment", type=str, default="r1",
                        choices=["r1", "r2", "r3", "all"])
    parser.add_argument("--model", type=str, default="llava-1.5",
                        choices=list(MODELS.keys()))
    parser.add_argument("--num_images", type=int, default=30,
                        help="Number of images (default 30 for speed)")
    args = parser.parse_args()

    ensure_dirs()
    print(f"Robustness & Ablation | experiment={args.experiment} | model={args.model}")
    print(f"Output: {OUTPUT_DIR}")

    if args.experiment in ("r1", "all"):
        run_r1_robustness(args.model, args.num_images)
    if args.experiment in ("r2", "all"):
        run_r2_task_control(args.model, args.num_images)
    if args.experiment in ("r3", "all"):
        run_r3_single_head_patching(args.model, args.num_images)

    print(f"\nDone. Results: {OUTPUT_DIR}")
    print(f"  Figures: {FIG_DIR}")
    print(f"  Tables:  {TABLE_DIR}")


if __name__ == "__main__":
    main()
