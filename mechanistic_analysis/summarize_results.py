#!/usr/bin/env python3
"""Step 2: Print numerical summary of multi-task results."""
import sys, json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from mechanistic_analysis.multi_task_encoding import (
    OUTPUT_DIR, MODEL_SPECS, TASK_TYPES, CAPTIONING_BASELINE,
)

combined = json.load(open(OUTPUT_DIR / 'multi_task_continuous_v3.json'))

# Also save to file for later reading
lines = []

for mk in ['llava-1.5', 'qwen2.5-vl', 'internvl3.5']:
    agg = combined[mk]
    name = MODEL_SPECS[mk]['name']
    cb = CAPTIONING_BASELINE.get(mk, {})
    lines.append(f"\n{'='*70}")
    lines.append(f"  {name}")
    lines.append(f"  Captioning baseline (1000-sample): Enc={cb.get('enc','?')}% "
          f"Arb={cb.get('arb','?')}% Grd={cb.get('grd','?')}%")
    lines.append(f"  {'Task':<14s} {'Lvl':>3s}  {'Δt (mean)':>10s}  {'Enc.Str.':>10s}  "
          f"{'Arb.Ratio':>10s}  {'Tokens':>6s}")
    lines.append(f"  {'-'*60}")
    for t in sorted(TASK_TYPES, key=lambda x: TASK_TYPES[x]['level']):
        label = TASK_TYPES[t]['short_label']
        lvl = agg[f'{t}_level']
        dt = agg[f'{t}_delta_means']['mean']
        es = agg[f'{t}_enc_strengths']['mean']
        ar = agg[f'{t}_arb_ratios']['mean']
        tk = agg[f'{t}_tokens']
        lines.append(f"  {label:<14s} {lvl:>3d}  {dt:>10.4f}  {es:>10.3f}  "
              f"{ar:>10.4f}  {tk:>6d}")

out = '\n'.join(lines)
print(out)
(OUTPUT_DIR / 'summary_output.txt').write_text(out)
