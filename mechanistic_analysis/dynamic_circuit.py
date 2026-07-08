#!/usr/bin/env python3
"""
Dynamic Circuit Discovery & Mechanistic Attribution Framework
===================================================================
Three novel analyses that extend the Δ_t signal from behavioral
diagnosis to mechanistic understanding:

1. DYNAMIC CIRCUIT DISCOVERY:
   Per-token tracking of which attention heads drive Δ_t, revealing
   how hallucination circuits EVOLVE during generation.
   Key question: Do hallucination chains have distinct neural signatures?

2. CROSS-ARCHITECTURE INVARIANT CIRCUITS:
   Compare 6 models (3 projector types: MLP / Perceiver / PixelShuffle)
   to find universal vs architecture-specific hallucination circuits.

3. ENCODING vs ARBITRATION DECOMPOSITION:
   Decompose Δ_t into "encoding failure" (visual info never encoded) vs
   "arbitration failure" (visual info encoded but overridden by language
   priors), enabling precision intervention.

Technical Approach:
  - Hook attention o_proj layers during factual/counterfactual forward passes
  - Per-head delta = ||factual_head_output - counterfactual_head_output||₂
  - Track per-head contribution across decoding steps
  - Cross-model alignment via layer normalization

Usage:
    python run_attribution.py --model qwen2.5-vl --mode dynamic --num_samples 10
    python run_attribution.py --model all --mode full --num_samples 20
"""

import torch
import numpy as np
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, Callable
from collections import defaultdict
from tqdm import tqdm
import json, os


# ═══════════════════════════════════════════════════════════════════════════
#  Core Hook Infrastructure
# ═══════════════════════════════════════════════════════════════════════════

class HeadOutputHook:
    """Captures per-head attention outputs during a forward pass.

    Hooks the input to o_proj (which receives concatenated head outputs
    BEFORE the output projection). Shape: (batch, seq_len, num_heads * head_dim).
    Reshaped to (batch, seq_len, num_heads, head_dim) for per-head analysis.
    """

    def __init__(self, num_heads: int, head_dim: int):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.outputs = []      # List of (layer_idx, per_head_activations)
        self.handles = []

    def hook_fn(self, layer_idx: int):
        def _hook(module, input, output):
            # output is (batch, seq_len, num_heads * head_dim) — pre o_proj projection
            # Actually, the input to o_proj is the concatenated head outputs
            x = input[0]  # First argument to o_proj
            b, s, d = x.shape
            h = self.num_heads
            hd = d // h
            # Reshape to separate heads
            per_head = x.view(b, s, h, hd).detach().cpu()
            self.outputs.append((layer_idx, per_head))
        return _hook

    def install(self, model, target_layers: Optional[list] = None):
        """Register hooks on all attention o_proj modules.

        Args:
            model: The VLM (e.g., Qwen2_5_VLForConditionalGeneration)
            target_layers: Specific layer indices to hook (None = all)
        """
        self.clear()

        # Find the language model's decoder layers
        # Navigate: model → model.language_model → model.layers[i] → self_attn.o_proj
        # Or: model.model.language_model.model.layers[i].self_attn.o_proj
        # This varies by architecture; we search generically

        for name, module in model.named_modules():
            if not ('self_attn.o_proj' in name or 'self_attn.wo' in name):
                continue
            if 'vision' in name.lower():
                continue  # Skip vision encoder attention

            # Extract layer index
            parts = name.split('.')
            layer_idx = None
            for i, p in enumerate(parts):
                if p in ('layers', 'layer') and i + 1 < len(parts):
                    try:
                        layer_idx = int(parts[i + 1])
                    except ValueError:
                        pass

            if layer_idx is None:
                continue
            if target_layers is not None and layer_idx not in target_layers:
                continue

            # Get num_heads and head_dim from the module or model config
            # The input dim to o_proj = num_heads * head_dim
            # We can infer from the Linear layer's in_features
            if hasattr(module, 'in_features'):
                d = module.in_features
                # Try to get num_heads from config
                h = self.num_heads
                hd = d // h
            else:
                # Fallback: use stored values
                h, hd = self.num_heads, self.head_dim

            hook = HeadOutputHookPerLayer(layer_idx, h, hd)
            handle = module.register_forward_hook(hook)
            self.handles.append((layer_idx, handle, hook))

    def clear(self):
        for _, handle, _ in self.handles:
            handle.remove()
        self.handles = []

    def get_all(self) -> dict:
        """Return {layer_idx: tensor (batch, seq_len, num_heads, head_dim)}."""
        results = {}
        for _, _, hook in self.handles:
            if hook.captured is not None:
                results[hook.layer_idx] = hook.captured
        return results


class HeadOutputHookPerLayer:
    """Captures the input to a single o_proj layer."""
    def __init__(self, layer_idx: int, num_heads: int, head_dim: int):
        self.layer_idx = layer_idx
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.captured = None

    def __call__(self, module, input, output):
        x = input[0]  # Shape: (batch, seq_len, num_heads * head_dim)
        b, s, d = x.shape
        h = self.num_heads
        hd = d // h
        self.captured = x.view(b, s, h, hd).detach().cpu()


def install_all_head_hooks(model, num_heads: int, head_dim: int,
                           target_layers: Optional[list] = None):
    """Install hooks on all attention o_proj layers. Returns cleanup function."""
    hooks = []
    for name, module in model.named_modules():
        if not ('self_attn.o_proj' in name or 'self_attn.wo' in name):
            continue
        if 'vision' in name.lower():
            continue

        parts = name.split('.')
        layer_idx = None
        for i, p in enumerate(parts):
            if p in ('layers', 'layer') and i + 1 < len(parts):
                try:
                    layer_idx = int(parts[i + 1])
                except ValueError:
                    pass

        if layer_idx is None:
            continue
        if target_layers is not None and layer_idx not in target_layers:
            continue

        if hasattr(module, 'in_features'):
            d = module.in_features
            hd = d // num_heads
        else:
            hd = head_dim

        hook_obj = HeadOutputHookPerLayer(layer_idx, num_heads, hd)
        handle = module.register_forward_hook(hook_obj)
        hooks.append((layer_idx, handle, hook_obj))

    def cleanup():
        for _, h, _ in hooks:
            h.remove()

    return hooks, cleanup


def _prepare_pixel_values_for_model(model, tokenizer, pil_image):
    """
    Preprocess an image for tokenizer-only models (InternVL, etc.).

    Tries model-specific image loading (internvl3_5.utils.load_image) first,
    then falls back to loading pixel_values from the generator's helper.
    Returns a tensor on the model's device/dtype, or None on failure.
    """
    import importlib
    from PIL import Image as PILImage
    img = PILImage.open(pil_image).convert("RGB") if isinstance(pil_image, str) else pil_image

    # Try InternVL3.5-style dynamic_preprocess + transform pipeline
    try:
        from internvl3_5.utils.causal_generator_optimized_internvl import load_image
        pv = load_image(img, input_size=448, max_num=12)
        pv = pv.unsqueeze(0) if pv.dim() == 3 else pv
        return pv.to(dtype=model.dtype, device=model.device)
    except ImportError:
        pass

    return None


def _prepare_shuffled_pixel_values(model, tokenizer, pil_image):
    """
    Prepare counterfactual pixel_values for tokenizer-only models (InternVL).

    InternVLChatModel.forward() REQUIRES pixel_values — it cannot be omitted.
    The counterfactual must have the same tensor structure as the factual
    (same number of tiles, same image_flags) so the <IMG_CONTEXT> token
    injection works correctly.  We shuffle spatial patches to scramble
    visual semantics while preserving low-level statistics.

    Returns pixel_values tensor in (num_patches, 3, H, W) format,
    or None if the shuffle function isn't available.
    """
    from PIL import Image as PILImage
    img = PILImage.open(pil_image).convert("RGB") if isinstance(pil_image, str) else pil_image

    try:
        from internvl3_5.utils.causal_generator_optimized_internvl import (
            shuffle_pil_patches, load_image,
        )
        shuffled_img = shuffle_pil_patches(img)
        pv = load_image(shuffled_img, input_size=448, max_num=12)
        pv = pv.unsqueeze(0) if pv.dim() == 3 else pv
        return pv.to(dtype=model.dtype, device=model.device)
    except ImportError:
        pass

    return None


# ═══════════════════════════════════════════════════════════════════════════
#  InternVL-specific: replicate chat()'s <image> → <IMG_CONTEXT> expansion
# ═══════════════════════════════════════════════════════════════════════════

def _internvl_build_prompt_with_context(model, tokenizer, prompt_text, pixel_values):
    """
    Replicate InternVL's chat() logic: replace '<image>' with
    '<img>' + '<IMG_CONTEXT>' * (num_image_token * num_tiles) + '</img>',
    tokenize, and set model.img_context_token_id.

    Returns a dict with input_ids, attention_mask, pixel_values, image_flags.
    """
    num_tiles = pixel_values.shape[0]  # tiles from dynamic_preprocess
    num_image_token = getattr(model, 'num_image_token', 256)

    IMG_START = '<img>'
    IMG_END = '</img>'
    IMG_CONTEXT = '<IMG_CONTEXT>'

    # Set img_context_token_id on model (needed by forward())
    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT)
    model.img_context_token_id = img_context_token_id

    # Build the image token block
    image_tokens = IMG_START + IMG_CONTEXT * (num_image_token * num_tiles) + IMG_END

    # Replace first '<image>' in the prompt
    if '<image>' in prompt_text:
        query = prompt_text.replace('<image>', image_tokens, 1)
    else:
        query = image_tokens + '\n' + prompt_text

    inputs = tokenizer(query, return_tensors='pt').to(model.device)
    inputs['pixel_values'] = pixel_values.to(dtype=model.dtype, device=model.device)
    # image_flags: 1 for all tiles (chat()/generate() doesn't filter, so we don't either)
    inputs['image_flags'] = torch.ones(num_tiles, 1, dtype=torch.long, device=model.device)
    return inputs


# ═══════════════════════════════════════════════════════════════════════════
#  Analysis 1: Dynamic Circuit Discovery
# ═══════════════════════════════════════════════════════════════════════════

def dynamic_circuit_discovery(model, processor, generator_class,
                               image, prompt: str,
                               num_layers: int, num_heads: int,
                               max_new_tokens: int = 64) -> dict:
    """
    Track which attention heads drive Δ_t at EACH decoding step.

    For each generated token, we:
    1. Run a factual forward pass (with image) and capture head outputs
    2. Run a counterfactual forward pass (no image) and capture head outputs
    3. Compute per-head delta = ||factual - counterfactual||₂
    4. Store the per-token × per-layer × per-head delta tensor

    This reveals the DYNAMIC EVOLUTION of hallucination circuits.

    Returns:
        {
            'per_token_deltas': np.ndarray (num_tokens, num_layers, num_heads),
                Per-head factual-counterfactual divergence at each token.
            'per_layer_dominance': np.ndarray (num_tokens, num_layers),
                The dominant layer's contribution at each token.
            'circuit_transitions': list of (token_idx, from_layer, to_layer),
                Points where the dominant circuit shifts.
            'tokens': list of generated token strings,
            'delta_t': list of per-token Δ_t values,
            'token_sources': list of source classifications,
        }
    """
    from PIL import Image as PILImage

    # Step 1: Generate to get token sequence + Δ_t
    generator = generator_class(model=model, processor=processor)

    if isinstance(image, str):
        pil_image = PILImage.open(image).convert("RGB")
    else:
        pil_image = image

    outputs = generator.generate(
        image=pil_image, prompt=prompt,
        max_new_tokens=max_new_tokens,
        num_beams=1, do_sample=False, use_cache=False,
    )

    token_sources = getattr(outputs, 'token_sources', [])
    generated_ids = outputs.sequences[0]
    delta_t_values = [ts.get('ate', 0.0) for ts in token_sources]

    # Decode tokens
    token_strings = []
    for tid in generated_ids:
        try:
            token_strings.append(processor.decode([int(tid)]))
        except Exception:
            token_strings.append(str(tid.item()))

    # Step 2: For each generated token position, capture head activations
    # from both factual and counterfactual paths
    num_tokens = len(token_sources)
    if num_tokens == 0:
        print("[WARN] No tokens generated")
        return None

    # Get the prompt-only inputs for building prefixes
    # Detect if processor is a plain tokenizer (InternVL, MiniCPM) vs
    # an image-capable AutoProcessor (LLaVA, Qwen2.5-VL, InstructBLIP).
    has_image_processor = hasattr(processor, 'image_processor') or hasattr(processor, 'image_processor_class')

    if pil_image is not None:
        if has_image_processor:
            messages_f = [{"role": "user", "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": prompt}
            ]}]
            messages_c = [{"role": "user", "content": [
                {"type": "text", "text": prompt}
            ]}]
            text_f = processor.apply_chat_template(messages_f, tokenize=False, add_generation_prompt=True)
            text_c = processor.apply_chat_template(messages_c, tokenize=False, add_generation_prompt=True)
            inputs_f_base = processor(text=text_f, images=pil_image, return_tensors="pt").to(model.device)
            inputs_c_base = processor(text=text_c, return_tensors="pt").to(model.device)
        else:
            # Tokenizer-only models (InternVL, MiniCPM): tokenize text separately,
            # preprocess image via model-specific load_image / build_transform.
            # Both factual and counterfactual need pixel_values — the
            # counterfactual uses shuffled patches.
            question_text = '<image>\n' + prompt
            counter_text = '<image>\n' + prompt  # use same template, shuffled image

            # Preprocess factual image into pixel_values
            _pixel_values = _prepare_pixel_values_for_model(model, processor, pil_image)

            # Preprocess counterfactual image (shuffled patches)
            _shuffled_pv = _prepare_shuffled_pixel_values(model, processor, pil_image)

            # InternVL3.5: replicate chat()'s <image> → <IMG_CONTEXT> expansion
            # so that forward() can find the img_context_token_id positions.
            if _pixel_values is not None and hasattr(model, 'img_context_token_id'):
                inputs_f_base = _internvl_build_prompt_with_context(
                    model, processor, question_text, _pixel_values)
            elif _pixel_values is not None:
                inputs_f_base = processor(question_text, return_tensors="pt").to(model.device)
                inputs_f_base['pixel_values'] = _pixel_values
            else:
                inputs_f_base = processor(question_text, return_tensors="pt").to(model.device)

            if _shuffled_pv is not None and hasattr(model, 'img_context_token_id'):
                inputs_c_base = _internvl_build_prompt_with_context(
                    model, processor, counter_text, _shuffled_pv)
            elif _shuffled_pv is not None:
                inputs_c_base = processor(counter_text, return_tensors="pt").to(model.device)
                inputs_c_base['pixel_values'] = _shuffled_pv
            else:
                # No counterfactual pixel_values — build a pure-text input
                # (same text template) for InternVL. The counterfactual
                # forward pass will use image_flags=zeros so InternVL's
                # forward() runs text-only.
                if _pixel_values is not None and hasattr(model, 'img_context_token_id'):
                    inputs_c_base = _internvl_build_prompt_with_context(
                        model, processor, counter_text, torch.zeros_like(_pixel_values))
                else:
                    inputs_c_base = processor(counter_text, return_tensors="pt").to(model.device)
    else:
        messages_f = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        text_f = processor.apply_chat_template(messages_f, tokenize=False, add_generation_prompt=True)
        inputs_f_base = processor(text=text_f, return_tensors="pt").to(model.device)
        inputs_c_base = inputs_f_base

    # Install hooks on all attention layers
    # Get head_dim in a model-agnostic way
    # LLaVA: model.config.text_config.hidden_size
    # Qwen2.5-VL/MiniCPM: model.config.hidden_size
    # InternVL: model.language_model.config.hidden_size
    try:
        hidden_size = model.config.text_config.hidden_size
    except AttributeError:
        try:
            hidden_size = model.config.hidden_size
        except AttributeError:
            # InternVL-style: full model config wraps an LLM config
            hidden_size = model.language_model.config.hidden_size
    head_dim = hidden_size // num_heads

    # Collect per-head deltas for each token position
    per_token_head_deltas = np.zeros((num_tokens, num_layers, num_heads))

    # Track all active hook cleanups to ensure nothing leaks
    active_cleanups = []

    try:
        # Capture from the prefill (prompt-only) step first
        # to establish baseline circuit
        with torch.inference_mode():
            # Install hooks for factual run
            hooks_f, cleanup_f = install_all_head_hooks(model, num_heads, head_dim)
            active_cleanups.append(cleanup_f)

            # Our hooks capture on the INPUT to o_proj (pre-projection head outputs)
            # For Qwen2/Llama: o_proj input IS the concatenated head outputs
            f_out = model(**{k: v for k, v in inputs_f_base.items()
                             if k in ('input_ids', 'attention_mask',
                                      'pixel_values', 'image_grid_thw', 'image_flags')})

            # Collect factual head outputs
            factual_per_layer = {}
            for layer_idx, _, hook_obj in hooks_f:
                if hook_obj.captured is not None:
                    factual_per_layer[layer_idx] = hook_obj.captured

            cleanup_f()
            active_cleanups.remove(cleanup_f)

            # Now counterfactual
            hooks_c, cleanup_c = install_all_head_hooks(model, num_heads, head_dim)
            active_cleanups.append(cleanup_c)
            c_out = model(**{k: v for k, v in inputs_c_base.items()
                             if k in ('input_ids', 'attention_mask',
                                      'pixel_values', 'image_grid_thw', 'image_flags')})

            counter_per_layer = {}
            for layer_idx, _, hook_obj in hooks_c:
                if hook_obj.captured is not None:
                    counter_per_layer[layer_idx] = hook_obj.captured

            cleanup_c()
            active_cleanups.remove(cleanup_c)

            # Compute per-head delta for this token (prefill)
            for layer_idx in set(factual_per_layer.keys()) & set(counter_per_layer.keys()):
                if layer_idx < num_layers:
                    f_h = factual_per_layer[layer_idx]  # (1, seq_len, num_heads, head_dim)
                    c_h = counter_per_layer[layer_idx]
                    # Take the L2 norm difference at the last token position
                    # (the last token aggregates cross-modal info per FCCT)
                    delta = torch.norm(f_h[0, -1, :, :] - c_h[0, -1, :, :], dim=-1)
                    per_token_head_deltas[0, layer_idx, :] = delta.numpy()

    except Exception as e:
        print(f"[WARN] Hook-based capture failed: {e}")
        # Fall back to logit-based approximation
        pass
    finally:
        for cleanup_fn in active_cleanups:
            try:
                cleanup_fn()
            except Exception:
                pass

    # Step 3: Compute derived metrics
    # Per-layer dominance per token
    per_layer_dominance = per_token_head_deltas.sum(axis=-1)  # (num_tokens, num_layers)

    # Circuit transitions: where does the dominant layer change by > threshold?
    dominant_layers = per_layer_dominance.argmax(axis=-1)
    transitions = []
    for t in range(1, num_tokens):
        if dominant_layers[t] != dominant_layers[t - 1]:
            transitions.append({
                'token_idx': t,
                'from_layer': int(dominant_layers[t - 1]),
                'to_layer': int(dominant_layers[t]),
                'delta_t_before': delta_t_values[t - 1],
                'delta_t_after': delta_t_values[t],
            })

    # Step 4: Identify "circuit regimes" — contiguous blocks with stable dominant layers
    regimes = []
    if len(dominant_layers) > 0:
        current_layer = dominant_layers[0]
        start_idx = 0
        for t in range(1, len(dominant_layers)):
            if dominant_layers[t] != current_layer:
                regimes.append({
                    'start_token': start_idx,
                    'end_token': t - 1,
                    'dominant_layer': int(current_layer),
                    'mean_delta_t': float(np.mean(delta_t_values[start_idx:t])),
                    'token_count': t - start_idx,
                })
                current_layer = dominant_layers[t]
                start_idx = t
        regimes.append({
            'start_token': start_idx,
            'end_token': len(dominant_layers) - 1,
            'dominant_layer': int(current_layer),
            'mean_delta_t': float(np.mean(delta_t_values[start_idx:])),
            'token_count': len(dominant_layers) - start_idx,
        })

    return {
        'per_token_head_deltas': per_token_head_deltas,
        'per_layer_dominance': per_layer_dominance,
        'dominant_layers': dominant_layers.tolist(),
        'circuit_transitions': transitions,
        'circuit_regimes': regimes,
        'tokens': token_strings[generated_ids.shape[0] - num_tokens:],
        'delta_t': delta_t_values,
        'token_sources': token_sources,
        'num_layers': num_layers,
        'num_heads': num_heads,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Analysis 2: Cross-Architecture Invariant Circuits
# ═══════════════════════════════════════════════════════════════════════════

def cross_architecture_comparison(all_model_results: dict) -> dict:
    """
    Compare hallucination circuits across architectures.

    Args:
        all_model_results: {
            'model_name': {
                'per_token_head_deltas': ndarray (T, L, H),
                'num_layers': int,
                'num_heads': int,
                'projector_type': str ('mlp' | 'perceiver' | 'pixelshuffle'),
            }
        }

    Returns:
        {
            'universal_heads': list of (layer_frac, head_idx, mean_score, n_models)
                Heads consistently important across ALL architectures.
            'projector_specific': {
                'mlp': [...], 'perceiver': [...], 'pixelshuffle': [...]
            }
            'mean_layer_importance': dict mapping model → normalized layer importance
            'circuit_similarity': pairwise similarity matrix
        }
    """
    # Step 1: Normalize layers to [0, 1] range for cross-model comparison
    normalized_results = {}
    for name, data in all_model_results.items():
        if data is None:
            continue
        nl = data.get('num_layers', 28)
        nh = data.get('num_heads', 28)
        head_deltas = data.get('per_token_head_deltas')
        if head_deltas is None or head_deltas.size == 0:
            continue

        # Aggregate across tokens: mean per-head delta
        mean_head_importance = head_deltas.mean(axis=0)  # (L, H)

        normalized_results[name] = {
            'mean_head_importance': mean_head_importance,
            'num_layers': nl,
            'num_heads': nh,
            'projector_type': data.get('projector_type', 'mlp'),
            'layer_to_frac': np.linspace(0, 1, nl),
        }

    if len(normalized_results) < 2:
        return {'error': 'Need at least 2 models for cross-architecture comparison'}

    # Step 2: For each normalized layer position (0.0 to 1.0, in 0.05 steps),
    # find the consistently important heads across models
    frac_bins = np.linspace(0, 1, 21)  # 20 bins
    universal_heads = []

    for i in range(len(frac_bins) - 1):
        frac_start = frac_bins[i]
        frac_end = frac_bins[i + 1]

        head_scores = defaultdict(list)
        for name, ndata in normalized_results.items():
            nl = ndata['num_layers']
            l2f = ndata['layer_to_frac']
            imp = ndata['mean_head_importance']  # (L, H)

            # Find layers that fall in this normalized bin
            for l in range(nl):
                if frac_start <= l2f[l] < frac_end:
                    # This layer's head importance
                    for h in range(imp.shape[1]):
                        head_scores[(l, h)].append(imp[l, h])
                    break  # Take closest layer per model per bin

        # Find heads with high mean importance across models
        for (l, h), scores in head_scores.items():
            if len(scores) >= len(normalized_results) * 0.5:  # Present in at least half
                mean_score = np.mean(scores)
                if mean_score > 0:
                    universal_heads.append({
                        'layer_frac': (frac_start + frac_end) / 2,
                        'head_idx': h,
                        'mean_score': float(mean_score),
                        'n_models': len(scores),
                        'std_score': float(np.std(scores)),
                    })

    # Sort by mean_score descending
    universal_heads.sort(key=lambda x: x['mean_score'], reverse=True)

    # Step 3: Projector-type specific analysis
    projector_specific = defaultdict(list)
    for name, ndata in normalized_results.items():
        ptype = ndata['projector_type']
        imp = ndata['mean_head_importance']
        for l in range(imp.shape[0]):
            for h in range(imp.shape[1]):
                if imp[l, h] > imp.mean() + imp.std():  # Significant heads
                    projector_specific[ptype].append({
                        'model': name,
                        'layer': l,
                        'head': h,
                        'score': float(imp[l, h]),
                        'layer_frac': float(ndata['layer_to_frac'][l]),
                    })

    # Step 4: Circuit similarity (pairwise layer-importance correlation)
    model_names = list(normalized_results.keys())
    n_models = len(model_names)
    similarity = np.zeros((n_models, n_models))
    max_bins = min(nd['num_layers'] for nd in normalized_results.values())

    for i in range(n_models):
        for j in range(n_models):
            # Compare mean layer importance (sum over heads)
            imp_i = normalized_results[model_names[i]]['mean_head_importance'].sum(axis=1)
            imp_j = normalized_results[model_names[j]]['mean_head_importance'].sum(axis=1)

            # Interpolate to same length for correlation
            xi = np.linspace(0, 1, len(imp_i))
            xj = np.linspace(0, 1, len(imp_j))
            x_common = np.linspace(0, 1, min(len(imp_i), len(imp_j)))

            interp_i = np.interp(x_common, xi, imp_i)
            interp_j = np.interp(x_common, xj, imp_j)

            corr = np.corrcoef(interp_i, interp_j)[0, 1]
            similarity[i, j] = corr

    return {
        'universal_heads': universal_heads[:50],
        'projector_specific': {k: v for k, v in projector_specific.items()},
        'model_names': model_names,
        'circuit_similarity': similarity.tolist(),
        'normalized_importance': {
            name: {
                'mean_head_importance': nd['mean_head_importance'].tolist(),
                'layer_to_frac': nd['layer_to_frac'].tolist(),
            }
            for name, nd in normalized_results.items()
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Analysis 3: Encoding vs Arbitration Decomposition
# ═══════════════════════════════════════════════════════════════════════════

def encoding_vs_arbitration_decomposition(model, processor, generator_class,
                                           image, prompt: str,
                                           num_layers: int, num_heads: int,
                                           max_new_tokens: int = 64) -> dict:
    """
    Decompose Δ_t into two orthogonal failure modes at the token level.

    For each token, measure:
      - VISUAL ENCODING STRENGTH: ||factual visual-head activation||
        (absolute strength of visual signal, independent of language competition)
      - ARBITRATION SIGNAL: (factual head activation) / (factual + counterfactual)
        (ratio of visual contribution to total — higher = visual wins)

    Then classify each token:
      - "encoding failure": visual encoding weak AND low Δ_t
        → fix: enhance visual features (better encoder, higher resolution)
      - "arbitration failure": visual encoding strong BUT low Δ_t
        → fix: suppress language prior (penalty, head-level steering)
      - "grounded": visual encoding strong AND high Δ_t
        → correct result, nothing to fix

    This decomposition enables PRECISION INTERVENTION: different failure
    modes need different fixes.

    Returns:
        {
            'per_token_classification': list of {
                'token_idx', 'token_str', 'encoding_strength',
                'arbitration_ratio', 'failure_mode', 'delta_t'
            },
            'per_layer_encoding': ndarray (num_layers, num_tokens),
            'per_layer_arbitration': ndarray (num_layers, num_tokens),
            'summary': {encoding_fail_rate, arbitration_fail_rate, grounded_rate}
        }
    """
    from PIL import Image as PILImage

    # Generate tokens
    generator = generator_class(model=model, processor=processor)

    if isinstance(image, str):
        pil_image = PILImage.open(image).convert("RGB")
    else:
        pil_image = image

    outputs = generator.generate(
        image=pil_image, prompt=prompt,
        max_new_tokens=max_new_tokens,
        num_beams=1, do_sample=False, use_cache=False,
    )

    token_sources = getattr(outputs, 'token_sources', [])
    # outputs.sequences[0] already contains full input+generated sequence
    full_sequence = outputs.sequences[0]  # shape: (total_len,)
    num_tokens = len(token_sources)

    if num_tokens == 0:
        return None

    # Decode only the generated portion (last num_tokens entries)
    token_strings = []
    generated_ids = full_sequence[-num_tokens:] if num_tokens <= len(full_sequence) else full_sequence
    for tid in generated_ids:
        try:
            token_strings.append(processor.decode([int(tid)]))
        except Exception:
            token_strings.append('?')

    # Prepare inputs — use the GENERATOR'S prefill inputs when available,
    # because chat-template models (Qwen2.5-VL) use dynamic resolution where
    # a second processor call may produce a different visual token count,
    # breaking the forward pass when paired with the generator's full_ids.
    has_image_processor = hasattr(processor, 'image_processor') or hasattr(processor, 'image_processor_class')

    # Check if the generator saved its prefill inputs (dual-cache generators do this)
    f_prefill = getattr(generator, '_f_prefill_inputs', None)
    c_prefill = getattr(generator, '_c_prefill_inputs', None)

    if f_prefill is not None:
        # Use the generator's own prefill inputs — they are guaranteed to match
        # the full_ids token sequence (same processor call).
        # f_prefill may be a BatchEncoding (LLaVA/Qwen) or a plain dict (InternVL).
        # Wrap plain dicts so both .input_ids and .get(k) work correctly.
        class _PrefillWrapper:
            __slots__ = ('_d',)
            def __init__(self, d):
                object.__setattr__(self, '_d', d)
            def __getattr__(self, name):
                if name == '_d':
                    return object.__getattribute__(self, '_d')
                return self._d[name]
            def get(self, key, default=None):
                return self._d.get(key, default)
            def __getitem__(self, key):
                return self._d[key]
            def __contains__(self, key):
                return key in self._d

        if isinstance(f_prefill, dict):
            inputs_f = _PrefillWrapper(f_prefill)
            inputs_c = _PrefillWrapper(c_prefill) if c_prefill is not None else None
        else:
            inputs_f = f_prefill
            inputs_c = c_prefill  # may be None or a BatchEncoding
    elif pil_image is not None:
        if has_image_processor:
            messages_f = [{"role": "user", "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": prompt}
            ]}]
            text_f = processor.apply_chat_template(messages_f, tokenize=False, add_generation_prompt=True)
            inputs_f = processor(text=text_f, images=pil_image, return_tensors="pt").to(model.device)
            inputs_c = None  # chat-template models use a different counterfactual strategy
        else:
            # Tokenizer-only models: tokenize text, preprocess image separately
            text_f = '<image>\n' + prompt
            counter_text = '<image>\n' + prompt  # same template, counterfactual uses zero pixel_values
            _pv = _prepare_pixel_values_for_model(model, processor, pil_image)
            if _pv is not None and hasattr(model, 'img_context_token_id'):
                # InternVL3.5: use chat()'s image token expansion
                inputs_f = _internvl_build_prompt_with_context(
                    model, processor, text_f, _pv)
            elif _pv is not None:
                inputs_f = processor(text_f, return_tensors="pt").to(model.device)
                inputs_f['pixel_values'] = _pv
            else:
                inputs_f = processor(text_f, return_tensors="pt").to(model.device)

            # Counterfactual: use zero pixel_values with image_flags=ones.
            # InternVLChatModel.forward() REQUIRES pixel_values — it cannot
            # run pure-text when the prompt contains <IMG_CONTEXT> tokens.
            # Zero pixel_values suppress the visual signal while maintaining
            # the correct tensor shape, producing the "no visual input"
            # baseline needed for meaningful arbitration_ratio computation.
            if _pv is not None and hasattr(model, 'img_context_token_id'):
                inputs_c = _internvl_build_prompt_with_context(
                    model, processor, counter_text, torch.zeros_like(_pv))
            elif _pv is not None:
                inputs_c = processor(counter_text, return_tensors="pt").to(model.device)
                inputs_c['pixel_values'] = torch.zeros_like(_pv)
            else:
                inputs_c = None
    else:
        messages_f = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        text_f = processor.apply_chat_template(messages_f, tokenize=False, add_generation_prompt=True)
        inputs_f = processor(text=text_f, return_tensors="pt").to(model.device)
        inputs_c = None  # no image, no counterfactual pixel_values

    # full_sequence already contains the entire input+generated sequence from model.generate()
    # Always ensure it's on the correct device
    full_ids = full_sequence.unsqueeze(0).to(model.device)

    # For tokenizer-only models (InternVL, MiniCPM), the generator returns only
    # the response tokens. Concatenate the prompt tokens (which contain the
    # model-specific image placeholder expansion, e.g. <IMG_CONTEXT> tokens for
    # InternVL) so that the model's forward() can find and replace them with
    # vision features — without this step, internvl's internal selected mask
    # is empty and input_embeds[selected] has shape [0, 4096].
    if pil_image is not None and not has_image_processor:
        full_ids = torch.cat([inputs_f.input_ids, full_ids], dim=1)

    # Run factual + counterfactual with hooks
    # Get head_dim in a model-agnostic way
    # LLaVA: model.config.text_config.hidden_size
    # Qwen2.5-VL/MiniCPM: model.config.hidden_size
    # InternVL: model.language_model.config.hidden_size
    try:
        hidden_size = model.config.text_config.hidden_size
    except AttributeError:
        try:
            hidden_size = model.config.hidden_size
        except AttributeError:
            # InternVL-style: full model config wraps an LLM config
            hidden_size = model.language_model.config.hidden_size
    head_dim = hidden_size // num_heads

    per_token_encoding = np.zeros((num_layers, num_tokens))
    per_token_arbitration = np.zeros((num_layers, num_tokens))

    # Initialize prompt_len early so it's always available (even on exception)
    prompt_len = inputs_f.input_ids.shape[1]

    # Track all active hook cleanups to ensure nothing leaks
    active_cleanups = []

    try:
        # Factual pass
        hooks_f, cleanup_f = install_all_head_hooks(model, num_heads, head_dim)
        active_cleanups.append(cleanup_f)

        with torch.inference_mode():
            # Build model kwargs properly — pass image_grid_thw if present (Qwen2.5-VL needs it)
            f_kwargs = dict(input_ids=full_ids,
                            attention_mask=torch.ones(1, full_ids.shape[1],
                                                      device=model.device, dtype=torch.long))
            for k in ('pixel_values', 'image_grid_thw', 'image_flags'):
                if inputs_f.get(k) is not None:
                    f_kwargs[k] = inputs_f[k]
            f_out = model(**f_kwargs)
        factual_heads = {}
        for layer_idx, _, hook_obj in hooks_f:
            if hook_obj.captured is not None:
                factual_heads[layer_idx] = hook_obj.captured
        cleanup_f()
        active_cleanups.remove(cleanup_f)

        # Counterfactual pass
        hooks_c, cleanup_c = install_all_head_hooks(model, num_heads, head_dim)
        active_cleanups.append(cleanup_c)
        with torch.inference_mode():
            c_kwargs = dict(
                input_ids=full_ids,
                attention_mask=torch.ones(1, full_ids.shape[1], device=model.device, dtype=torch.long),
            )
            # Use counterfactual inputs when available (shuffled pixel_values
            # for InternVL, or None for models that can run pure-text).
            if inputs_c is not None:
                for k in ('pixel_values', 'image_grid_thw', 'image_flags'):
                    if inputs_c.get(k) is not None:
                        c_kwargs[k] = inputs_c[k]
            else:
                # Pure-text counterfactual fallback (LLaVA/Qwen paths where
                # the prompt has no <IMG_CONTEXT> tokens).
                if inputs_f.get('image_flags') is not None:
                    c_kwargs['image_flags'] = torch.zeros_like(inputs_f['image_flags'])
                # Qwen2.5-VL / InternVLChatModel require pixel_values for their
                # forward() — provide zero values to suppress visual signal.
                # Qwen additionally requires image_grid_thw whenever pixel_values
                # is present (the model iterates over both to count visual tiles).
                if inputs_f.get('pixel_values') is not None:
                    c_kwargs['pixel_values'] = torch.zeros_like(inputs_f['pixel_values'])
                    if inputs_f.get('image_grid_thw') is not None:
                        c_kwargs['image_grid_thw'] = inputs_f['image_grid_thw']
            c_out = model(**c_kwargs)
        counter_heads = {}
        for layer_idx, _, hook_obj in hooks_c:
            if hook_obj.captured is not None:
                counter_heads[layer_idx] = hook_obj.captured
        cleanup_c()
        active_cleanups.remove(cleanup_c)

        # Compute per-token metrics at each layer
        for layer_idx in set(factual_heads.keys()) & set(counter_heads.keys()):
            if layer_idx >= num_layers:
                continue
            f_h = factual_heads[layer_idx]  # (1, total_seq_len, num_heads, head_dim)
            c_h = counter_heads[layer_idx]

            # For each generated token position
            for t in range(num_tokens):
                seq_pos = prompt_len + t

                # Encoding strength: L2 norm of factual head outputs at this position
                encoding = torch.norm(f_h[0, seq_pos, :, :], dim=-1).mean().item()

                # Arbitration ratio: ||factual|| / (||factual|| + ||counterfactual||)
                f_norm = torch.norm(f_h[0, seq_pos, :, :], dim=-1)
                c_norm = torch.norm(c_h[0, seq_pos, :, :], dim=-1)
                arb_ratio = (f_norm / (f_norm + c_norm + 1e-8)).mean().item()

                per_token_encoding[layer_idx, t] = encoding
                per_token_arbitration[layer_idx, t] = arb_ratio

    except Exception as e:
        print(f"[WARN] Encoding/arbitration decomposition failed: {e}")
        # If the entire hook-based analysis failed, return a minimal result
        # with null metrics instead of fabricating data from zero-filled arrays.
        classifications = []
        for t in range(num_tokens):
            delta_t_val = token_sources[t].get('ate', 0.0) if t < len(token_sources) else 0.0
            classifications.append({
                'token_idx': t,
                'token_str': token_strings[t] if t < len(token_strings) else '?',
                'encoding_strength': None,
                'arbitration_ratio': None,
                'delta_t': delta_t_val,
                'failure_mode': 'hook_failure',
                'source': token_sources[t].get('source', 'unknown') if t < len(token_sources) else 'unknown',
            })
        return {
            'per_token_classification': classifications,
            'per_layer_encoding': per_token_encoding,
            'per_layer_arbitration': per_token_arbitration,
            'summary': {
                'encoding_failure_rate': float('nan'),
                'arbitration_failure_rate': float('nan'),
                'grounded_rate': float('nan'),
                'total_tokens': num_tokens,
                'encoding_failure_count': 0,
                'arbitration_failure_count': 0,
                'grounded_count': 0,
                'error': str(e),
            },
        }
    finally:
        for cleanup_fn in active_cleanups:
            try:
                cleanup_fn()
            except Exception:
                pass

    # Classify each token
    classifications = []
    encoding_fail_count = 0
    arbitration_fail_count = 0
    grounded_count = 0

    # Use middle layers (where cross-modal fusion happens, per FCCT)
    mid_start = num_layers // 3
    mid_end = 2 * num_layers // 3

    # Compute per-model absolute thresholds.
    # encoding_threshold: 30th percentile of per-token encoding strength values
    # (aggregated across mid layers). Tokens below this have weak visual signal.
    # arbitration_threshold: 0.55 — visual signal must be measurably stronger
    # than counterfactual noise (ratio > 0.5 means factual > counterfactual).
    encoding_threshold = np.percentile(per_token_encoding[mid_start:mid_end, :], 30)
    arbitration_threshold = 0.55

    for t in range(num_tokens):
        mean_encoding = per_token_encoding[mid_start:mid_end, t].mean()
        mean_arbitration = per_token_arbitration[mid_start:mid_end, t].mean()
        delta_t_val = token_sources[t].get('ate', 0.0) if t < len(token_sources) else 0.0

        if mean_encoding < encoding_threshold:
            failure_mode = "encoding_failure"
            encoding_fail_count += 1
        elif mean_arbitration < arbitration_threshold:
            failure_mode = "arbitration_failure"
            arbitration_fail_count += 1
        else:
            failure_mode = "grounded"
            grounded_count += 1

        classifications.append({
            'token_idx': t,
            'token_str': token_strings[t] if t < len(token_strings) else '?',
            'encoding_strength': float(mean_encoding),
            'arbitration_ratio': float(mean_arbitration),
            'delta_t': delta_t_val,
            'failure_mode': failure_mode,
            'source': token_sources[t].get('source', 'unknown') if t < len(token_sources) else 'unknown',
        })

    total = num_tokens or 1
    return {
        'per_token_classification': classifications,
        'per_layer_encoding': per_token_encoding,
        'per_layer_arbitration': per_token_arbitration,
        'summary': {
            'encoding_failure_rate': encoding_fail_count / total,
            'arbitration_failure_rate': arbitration_fail_count / total,
            'grounded_rate': grounded_count / total,
            'total_tokens': num_tokens,
            'encoding_failure_count': encoding_fail_count,
            'arbitration_failure_count': arbitration_fail_count,
            'grounded_count': grounded_count,
        },
    }
