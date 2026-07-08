#!/usr/bin/env python3
"""Step 1: Generate figure and LaTeX table from combined results."""
import sys, json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from mechanistic_analysis.multi_task_encoding import (
    make_figure, make_latex_table, OUTPUT_DIR, FIG_DIR, TABLE_DIR
)

combined = json.load(open(OUTPUT_DIR / 'multi_task_continuous_v3.json'))
print(f"Models loaded: {list(combined.keys())}")
for mk in combined:
    n = combined[mk]['captioning_delta_means']['n']
    print(f"  {mk}: {n} samples")

make_figure(combined, FIG_DIR / 'multi_task_continuous_v3.pdf')
make_latex_table(combined, TABLE_DIR / 'multi_task_table.tex')
print("\nDone: figure + table generated")
