#!/usr/bin/env python3
"""
Counterfactual Image Generation
==========================================
★ Novel contribution: Progressive dynamic counterfactuals that evolve
  during generation, enabling causal testing of visual dependency decay.

Level 1 — Static baselines (existing methods, for comparison):
  - shuffle_image_patches()     Uniform patch shuffling (SDCD-style)
  - add_gaussian_noise()        Noise injection
  - replace_with_random()       Random image replacement

Level 2 — ★ Progressive Visual Erasure (main contribution):
  - progressive_erasure()       Gradually erases more patches over steps.
                                Tests: "When does the model stop needing vision?"

Level 3 — Ablation variants:
  - attention_guided_erasure()  Only erases high-attention regions.
                                Tests: "Is attention causally functional?"
  - semantic_preserving_swap()  Swaps semantically similar patches.
                                Tests: "Spatial structure vs semantic content?"

All methods produce pixel_values with identical shape to input, preserving
sequence length, position encoding, and computational graph structure.

Usage:
    from counterfactual_utils import progressive_erasure

    # At each decoding step t (out of total T):
    cf_image = progressive_erasure(pixel_values, erasure_fraction=t/T)
"""

import torch
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
#  Internal helpers
# ═══════════════════════════════════════════════════════════════════════════

def _extract_patches(pixel_values, patch_size=14):
    """Extract patches from pixel_values tensor.
    Returns: (patches, n_h, n_w, orig_shape) or (None, ...) on failure."""
    if pixel_values is None:
        return None, 0, 0, None
    if isinstance(pixel_values, list):
        # Handle batched multi-image input
        results = [_extract_patches(pv, patch_size) for pv in pixel_values]
        return [r[0] for r in results], [r[1] for r in results], [r[2] for r in results], None

    B, C, H, W = pixel_values.shape
    if H % patch_size != 0 or W % patch_size != 0:
        pad_h = (patch_size - H % patch_size) % patch_size
        pad_w = (patch_size - W % patch_size) % patch_size
        if pad_h > 0 or pad_w > 0:
            pixel_values = torch.nn.functional.pad(pixel_values, (0, pad_w, 0, pad_h), mode='reflect')
        B, C, H, W = pixel_values.shape

    n_h = H // patch_size
    n_w = W // patch_size
    N = n_h * n_w
    patches = pixel_values.view(B, C, n_h, patch_size, n_w, patch_size)
    patches = patches.permute(0, 1, 2, 4, 3, 5)  # (B, C, n_h, n_w, ps, ps)
    patches = patches.reshape(B, C, N, patch_size, patch_size)  # (B, C, N, ps, ps)
    return patches, n_h, n_w, pixel_values.shape


def _reconstruct_from_patches(patches, n_h, n_w, orig_shape):
    """Reconstruct image from patch tensor."""
    B, C, N, ps, _ = patches.shape
    patches = patches.view(B, C, n_h, n_w, ps, ps)
    patches = patches.permute(0, 1, 2, 4, 3, 5)
    reconstructed = patches.reshape(orig_shape)
    return reconstructed


def _set_seed(seed):
    if seed is not None:
        return torch.random.get_rng_state()
    return None


def _restore_seed(prev_state):
    if prev_state is not None:
        torch.random.set_rng_state(prev_state)


# ═══════════════════════════════════════════════════════════════════════════
#  Level 1: Static Baselines (existing methods, for comparison)
# ═══════════════════════════════════════════════════════════════════════════

def shuffle_image_patches(pixel_values, patch_size=14, seed=None):
    """Uniform patch shuffling (SDCD baseline). All patches randomly permuted."""
    if pixel_values is None:
        return None
    prev = _set_seed(seed)
    if isinstance(pixel_values, list):
        result = [shuffle_image_patches(pv, patch_size, seed) for pv in pixel_values]
        _restore_seed(prev)
        return result

    patches, n_h, n_w, orig_shape = _extract_patches(pixel_values, patch_size)
    B, C, N, ps, _ = patches.shape
    shuffled = torch.zeros_like(patches)
    for b in range(B):
        shuffled[b] = patches[b, :, torch.randperm(N), :, :]
    result = _reconstruct_from_patches(shuffled, n_h, n_w, orig_shape)
    _restore_seed(prev)
    return result


def black_image(pixel_values, seed=None):
    """Replace image with zeros — strongest causal contrast.

    A pure black image provides zero visual information, forcing the
    model to rely entirely on language priors.  Compared to patch shuffle
    (which preserves color/texture statistics), black-image creates a
    larger logit gap on visually-grounded tokens, producing stronger Δ_t.

    This is the strongest valid counterfactual for causal visual grounding
    tests — it maximally isolates the marginal effect of the visual
    modality on token-level predictions.
    """
    if pixel_values is None:
        return None
    prev = _set_seed(seed)
    if isinstance(pixel_values, list):
        result = [black_image(pv, seed) for pv in pixel_values]
        _restore_seed(prev)
        return result
    result = torch.zeros_like(pixel_values)
    _restore_seed(prev)
    return result


def add_gaussian_noise(pixel_values, std=0.1, seed=None):
    """Add Gaussian noise for graded counterfactual."""
    if pixel_values is None:
        return None
    prev = _set_seed(seed)
    if isinstance(pixel_values, list):
        result = [add_gaussian_noise(pv, std, seed) for pv in pixel_values]
        _restore_seed(prev)
        return result
    noise = torch.randn_like(pixel_values) * std
    result = pixel_values + noise
    _restore_seed(prev)
    return result


def replace_with_random(pixel_values, image_pool, seed=None):
    """Replace with random image from pool."""
    if pixel_values is None or not image_pool:
        return None
    prev = _set_seed(seed)
    idx = torch.randint(0, len(image_pool), (1,)).item()
    _restore_seed(prev)
    return image_pool[idx]


# ═══════════════════════════════════════════════════════════════════════════
#  Level 2: ★ Progressive Visual Erasure (main contribution)
# ═══════════════════════════════════════════════════════════════════════════

def progressive_erasure(pixel_values, erasure_fraction=0.0, patch_size=14,
                         mode="shuffle", seed=None):
    """
    ★ Novel: Gradually erase visual information proportional to erasure_fraction.

    This is the KEY INNOVATION: instead of a fixed counterfactual applied to
    all tokens, the counterfactual EVOLVES during generation. Each decoding
    step t gets a counterfactual with erasure_fraction = t / total_steps.

    This directly causally tests the core hypothesis: visual dependency
    decays during generation. If the model truly stops depending on vision
    in later tokens, then progressive erasure should have DIMINISHING IMPACT
    on Δ_t — the model becomes progressively less sensitive to visual degradation.

    Args:
        pixel_values: Normalized image tensor (B, C, H, W) from processor.
        erasure_fraction: Float 0.0–1.0. Fraction of patches to erase.
            0.0 = original image (no erasure)
            0.5 = half patches erased
            1.0 = all patches erased (fully shuffled)
        patch_size: ViT patch size (default 14).
        mode: "shuffle" (permute patches) or "noise" (replace with Gaussian).
        seed: Optional reproducibility seed.

    Returns:
        Tensor of identical shape with [erasure_fraction] of patches degraded.

    Scientific prediction:
        Δ_t(clean, progressive_erasure(f)) should DECREASE as f increases.
        The rate of decrease reveals the model's visual dependency at each step.

        If later tokens show smaller Δ_t drop → visual dependency has decayed.
        If later tokens show same Δ_t drop → model still relies on vision.
    """
    if pixel_values is None:
        return None
    if erasure_fraction <= 0.0:
        return pixel_values
    prev = _set_seed(seed)

    if isinstance(pixel_values, list):
        result = [progressive_erasure(pv, erasure_fraction, patch_size, mode, seed)
                   for pv in pixel_values]
        _restore_seed(prev)
        return result

    patches, n_h, n_w, orig_shape = _extract_patches(pixel_values, patch_size)
    B, C, N, ps, _ = patches.shape

    n_erase = max(1, int(N * erasure_fraction))
    if erasure_fraction >= 1.0:
        n_erase = N

    result_patches = patches.clone()

    for b in range(B):
        erase_idx = torch.randperm(N)[:n_erase]
        if mode == "shuffle":
            # Shuffle the selected patches among themselves
            perm = erase_idx[torch.randperm(n_erase)]
            result_patches[b, :, erase_idx, :, :] = patches[b, :, perm, :, :]
        elif mode == "noise":
            # Replace with Gaussian noise matching patch statistics
            patch_std = patches[b].std()
            noise = torch.randn(n_erase, C, ps, ps, device=patches.device) * patch_std * 0.5
            result_patches[b, :, erase_idx, :, :] = noise.permute(1, 0, 2, 3)

    result = _reconstruct_from_patches(result_patches, n_h, n_w, orig_shape)
    _restore_seed(prev)
    return result


# ═══════════════════════════════════════════════════════════════════════════
#  Level 3A: Attention-Guided Selective Erasure (ablation)
# ═══════════════════════════════════════════════════════════════════════════

def attention_guided_erasure(pixel_values, attention_map, top_k_fraction=0.3,
                              patch_size=14, seed=None):
    """
    Ablation: Only erase patches that receive HIGH ATTENTION from the model.

    Tests the causal role of attention: if shuffling high-attention patches
    reduces Δ_t MORE than shuffling random patches, attention is causally
    functional (not just correlational).

    Args:
        pixel_values: Normalized image tensor (B, C, H, W).
        attention_map: torch.Tensor of shape (n_patches,) or (n_patches_h, n_patches_w).
            Aggregated attention weights per image patch from the factual forward pass.
        top_k_fraction: Fraction of highest-attention patches to erase (0.0–1.0).
        patch_size: ViT patch size.

    Returns:
        Tensor with high-attention patches selectively erased.
    """
    if pixel_values is None:
        return None
    prev = _set_seed(seed)

    if isinstance(pixel_values, list):
        result = [attention_guided_erasure(pv, attention_map, top_k_fraction, patch_size, seed)
                   for pv in pixel_values]
        _restore_seed(prev)
        return result

    patches, n_h, n_w, orig_shape = _extract_patches(pixel_values, patch_size)
    B, C, N, ps, _ = patches.shape

    # Flatten attention map to (N,) per-batch
    if attention_map is not None:
        if attention_map.dim() == 3:
            attn_flat = attention_map.reshape(B, -1)  # (B, N)
        elif attention_map.dim() == 2:
            attn_flat = attention_map.reshape(-1)[:N].unsqueeze(0).expand(B, -1)
        else:
            attn_flat = attention_map.reshape(-1).unsqueeze(0).expand(B, -1)
    else:
        # Fallback: uniform attention
        attn_flat = torch.ones(B, N)

    n_erase = max(1, int(N * top_k_fraction))
    result_patches = patches.clone()

    for b in range(B):
        # Select top-k highest attention patches
        _, top_idx = torch.topk(attn_flat[b], n_erase)
        # Shuffle them
        perm = top_idx[torch.randperm(n_erase)]
        result_patches[b, :, top_idx, :, :] = patches[b, :, perm, :, :]

    result = _reconstruct_from_patches(result_patches, n_h, n_w, orig_shape)
    _restore_seed(prev)
    return result


# ═══════════════════════════════════════════════════════════════════════════
#  Level 3B: Semantic-Preserving Swaps (ablation)
# ═══════════════════════════════════════════════════════════════════════════

def semantic_preserving_swap(pixel_values, patch_embeddings=None,
                              swap_fraction=0.5, spatial_threshold=3,
                              patch_size=14, seed=None):
    """
    Ablation: Swap semantically SIMILAR but spatially DISTANT patches.

    Tests whether models need spatial structure or just semantic content.
    If swapping similar patches across the image has no effect on Δ_t,
    the model relies primarily on semantic content. If Δ_t drops, the model
    needs precise spatial relationships.

    Args:
        pixel_values: Normalized image tensor (B, C, H, W).
        patch_embeddings: Optional (B, N, D) ViT patch embeddings for similarity.
            If None, uses raw pixel similarity in patch space.
        swap_fraction: Fraction of patches to swap (0.0–1.0).
        spatial_threshold: Minimum grid distance for a swap to be allowed.
            Prevents swapping adjacent patches (which wouldn't disrupt structure much).
        patch_size: ViT patch size.

    Returns:
        Tensor with semantically similar distant patches swapped.
    """
    if pixel_values is None:
        return None
    if swap_fraction <= 0.0:
        return pixel_values
    prev = _set_seed(seed)

    if isinstance(pixel_values, list):
        result = [semantic_preserving_swap(pv, patch_embeddings, swap_fraction,
                                            spatial_threshold, patch_size, seed)
                   for pv in pixel_values]
        _restore_seed(prev)
        return result

    patches, n_h, n_w, orig_shape = _extract_patches(pixel_values, patch_size)
    B, C, N, ps, _ = patches.shape
    device = patches.device

    # Compute patch-to-patch similarity
    if patch_embeddings is not None:
        # Use provided ViT embeddings
        if patch_embeddings.dim() == 3:
            emb = patch_embeddings  # (B, N, D)
        else:
            emb = patch_embeddings.reshape(B, N, -1)
    else:
        # Use raw patch pixel values flattened as proxy for semantics
        emb = patches.reshape(B, N, C * ps * ps)  # (B, N, C*ps*ps)

    # Spatial positions (grid coordinates)
    y_coords = torch.arange(n_h, device=device).unsqueeze(1).expand(n_h, n_w).reshape(-1)
    x_coords = torch.arange(n_w, device=device).unsqueeze(0).expand(n_h, n_w).reshape(-1)
    dist_matrix = torch.abs(y_coords.unsqueeze(1) - y_coords.unsqueeze(0)) + \
                  torch.abs(x_coords.unsqueeze(1) - x_coords.unsqueeze(0))  # (N, N), Manhattan

    n_swap = max(1, int(N * swap_fraction))
    result_patches = patches.clone()

    for b in range(B):
        # Compute cosine similarity between all patch pairs
        e = emb[b]  # (N, D)
        e_norm = e / (e.norm(dim=1, keepdim=True) + 1e-8)
        sim = e_norm @ e_norm.T  # (N, N)

        # Mask: only allow swaps between spatially distant patches
        valid_mask = (dist_matrix >= spatial_threshold).float()
        sim_masked = sim * valid_mask

        # Select patches to swap: choose ones with highest "has a similar distant partner" score
        best_partner_sim, _ = sim_masked.max(dim=1)  # (N,)
        _, swap_candidates = torch.topk(best_partner_sim, n_swap)

        # For each candidate, find its best distant swap partner
        swap_pairs = []
        used = set()
        for idx in swap_candidates:
            idx = idx.item()
            if idx in used:
                continue
            valid_partners = (valid_mask[idx] > 0).nonzero(as_tuple=True)[0]
            if len(valid_partners) == 0:
                continue
            best = valid_partners[sim[idx, valid_partners].argmax()].item()
            if best in used:
                continue
            swap_pairs.append((idx, best))
            used.add(idx)
            used.add(best)

        # Execute swaps
        for a, b_idx in swap_pairs:
            tmp = result_patches[b, :, a, :, :].clone()
            result_patches[b, :, a, :, :] = result_patches[b, :, b_idx, :, :]
            result_patches[b, :, b_idx, :, :] = tmp

    result = _reconstruct_from_patches(result_patches, n_h, n_w, orig_shape)
    _restore_seed(prev)
    return result
