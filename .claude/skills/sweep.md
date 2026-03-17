---
name: sweep
description: Run the GEPA-inspired sweep generation loop. Computes the Pareto frontier over existing sweep results, selects a parent via coverage-weighted sampling, proposes a mutated config, evaluates it, and reports results. Use when the user wants to generate and run new sweep experiments.
argument-hint: "[N] number of sweeps to generate (default: 1)"
allowed-tools: Read, Write, Edit, Bash, Grep, Glob, Agent
---

# GEPA-Inspired Sweep Generation for Trend Following

You are conducting a Pareto-optimal sweep generation loop for a trend-following futures strategy
research project. The project sweeps signal types (cumsum_ma, breakout, ema_cross, dual_ma,
momentum) across volatility windows, smoothing parameters, and signal-specific lookbacks.

Existing sweep results live in `sweep_results/singles.parquet` (4,520+ rows). Each row has:
`signal_type`, `vol_window`, `smoothing`, `param1`, `param2`, `sharpe`, `max_dd`, plus per-decade
(`sharpe_1980`, ..., `sharpe_2020`) and per-category (`cat_sharpe_bonds`, etc.) columns.

The core evaluation code is in `sweep.py`. The signal generators, data loading, and evaluation
functions are all defined there.

The user may pass a number N as `$ARGUMENTS` — generate and run N sweeps sequentially (default: 1).

## Step 1: Compute the Pareto Frontier

Run the following Python script to compute the frontier. Use `uv run` (never bare `python`).

The frontier is multi-objective over **rolling-window Sharpe ratios** (5-year windows, stride 2 years)
to find configs that are non-dominated across different market regimes.

```
uv run python -c "
import json, numpy as np, pandas as pd
from pathlib import Path

RESULTS_DIR = Path('sweep_results')
WINDOW_YEARS = 5
STRIDE_YEARS = 2

# Load all singles results
singles = pd.read_parquet(RESULTS_DIR / 'singles.parquet')
dense = RESULTS_DIR / 'dense.parquet'
if dense.exists():
    d = pd.read_parquet(dense)
    # Tag source
    singles['source'] = 'singles'
    d['source'] = 'dense'
    singles = pd.concat([singles, d], ignore_index=True)
else:
    singles['source'] = 'singles'

# Filter to valid results
singles = singles[singles.sharpe.notna() & (singles.sharpe != 0)].copy()

# Build a unique name for each config
def make_name(row):
    p2 = f'_{int(row.param2)}' if pd.notna(row.param2) else ''
    return f'{row.signal_type}_vw{int(row.vol_window)}_sm{int(row.smoothing)}_p{int(row.param1)}{p2}'

singles['name'] = singles.apply(make_name, axis=1)

# Collect decade sharpe columns as our objectives
decade_cols = [c for c in singles.columns if c.startswith('sharpe_') and c != 'sharpe']
decade_cols = sorted(decade_cols)

# Also include max_dd (negated, since less negative = better)
# Build score matrix: each row is a config, each col is an objective
objectives = decade_cols + ['sharpe', 'neg_max_dd']
singles['neg_max_dd'] = -singles['max_dd']  # higher = better

# Drop rows missing all decade columns
singles = singles.dropna(subset=['sharpe'])

# Fill NaN decade sharpes with a penalty (below worst observed)
for col in decade_cols:
    if col in singles.columns:
        worst = singles[col].min() if singles[col].notna().any() else -1
        singles[col] = singles[col].fillna(worst - 0.5)

scores = singles[objectives].values
names = singles['name'].values
n = len(names)

# Pareto frontier (non-dominated sorting)
dominated = np.zeros(n, dtype=bool)
for i in range(n):
    if dominated[i]:
        continue
    for j in range(n):
        if i == j or dominated[j]:
            continue
        if np.all(scores[j] >= scores[i]) and np.any(scores[j] > scores[i]):
            dominated[i] = True
            break

frontier_mask = ~dominated
frontier_idx = np.where(frontier_mask)[0]
frontier_names = names[frontier_idx]

# Coverage: for each objective, which frontier member is best?
frontier_scores = scores[frontier_idx]
n_obj = frontier_scores.shape[1]
best_per_obj = frontier_scores.argmax(axis=0)
counts = pd.Series(best_per_obj).value_counts().reindex(range(len(frontier_idx)), fill_value=0)

print(f'Candidates: {n}, Objectives: {n_obj} ({objectives})')
print(f'Frontier: {len(frontier_idx)}')
print()
print('Pareto Frontier (coverage-ranked):')
for rank, (fi, count) in enumerate(sorted(enumerate(frontier_idx), key=lambda x: -counts.iloc[x[0]])):
    row = singles.iloc[fi]
    pct = count / n_obj * 100
    print(f'  {rank+1:3d}. {row[\"name\"]:50s}  sharpe={row.sharpe:6.3f}  max_dd={row.max_dd:7.4f}  coverage={pct:5.1f}% ({count}/{n_obj})')
    if rank >= 29:
        remaining = len(frontier_idx) - rank - 1
        if remaining > 0:
            print(f'  ... and {remaining} more frontier members')
        break

# Save frontier as JSON
result = []
for fi_local, fi_global in enumerate(frontier_idx):
    row = singles.iloc[fi_global]
    result.append({
        'name': row['name'],
        'coverage': int(counts.iloc[fi_local]),
        'signal_type': row['signal_type'],
        'vol_window': int(row['vol_window']),
        'smoothing': int(row['smoothing']),
        'param1': int(row['param1']),
        'param2': int(row['param2']) if pd.notna(row['param2']) else None,
        'sharpe': float(row['sharpe']),
        'max_dd': float(row['max_dd']),
        'source': row.get('source', 'singles'),
    })
Path('/tmp/pareto_frontier.json').write_text(json.dumps(result, indent=2))
print()
print(f'Frontier ({len(result)} members) saved to /tmp/pareto_frontier.json')
"
```

Present the frontier table to the user.

## Step 2: Select a Parent

Read `/tmp/pareto_frontier.json`. Sample a parent with probability proportional to coverage count:

- Higher-coverage candidates are more likely to be selected (exploitation)
- But all frontier members have a chance (exploration)
- If a frontier member has 0 coverage, give it a minimum weight of 1

Tell the user which parent was selected and why.

## Step 3: Propose a Mutation

Analyze the parent config and the frontier landscape to propose a child config. Consider:

**What to mutate** (pick 1-3 parameters to change):

| Parameter | Range | Notes |
|-----------|-------|-------|
| `signal_type` | cumsum_ma, breakout, ema_cross, dual_ma, momentum | Can switch signal entirely |
| `vol_window` | 60, 90, 120, 180, 250 | Volatility estimation period |
| `smoothing` | 1–100 | Position smoothing (1 = no smoothing) |
| `lookback` (cumsum_ma) | 40–500 | MA lookback window |
| `window` (breakout, momentum) | 20–500 | Channel/momentum window |
| `fast_span` (ema_cross) | 5–75 | Fast EMA span |
| `slow_span` (ema_cross) | 50–500 | Slow EMA span (must be > 2x fast) |
| `fast_window` (dual_ma) | 10–100 | Fast SMA window |
| `slow_window` (dual_ma) | 100–500 | Slow SMA window (must be > 2x fast) |

**Mutation strategy:**

1. **Explore underrepresented regions**: If the frontier is dominated by one signal type (e.g., all
   cumsum_ma), propose a child using a different signal type.
2. **Perturb the parent**: Change 1-2 params by 20-50% in either direction, keeping within valid ranges.
3. **Combine strategies**: If two frontier members excel in different decades with different param
   values, interpolate or extrapolate.
4. **Decade targeting**: If one decade (e.g., 2020s) is weak across the frontier, target that regime.
5. **For ema_cross/dual_ma**: Ensure `slow > 2 * fast` constraint is maintained.

## Step 4: Evaluate the Mutation

Run the mutation inline using the existing sweep infrastructure. Do NOT create a separate script file.

```
uv run python -c "
import numpy as np, pandas as pd, json, sys
from pathlib import Path

# Import from sweep.py
sys.path.insert(0, '.')
from sweep import (
    load_data, precompute_vols, build_norm_cumret,
    compute_signal, evaluate_by_category,
    VOL_WINDOWS, LEVERAGE, START_YEAR
)

# --- CHILD CONFIG (fill in from Step 3) ---
SIGNAL_TYPE = '$SIGNAL_TYPE'
VOL_WINDOW = $VOL_WINDOW
SMOOTHING = $SMOOTHING
PARAMS = $PARAMS  # e.g. {'lookback': 200} or {'fast_span': 20, 'slow_span': 150}
PARENT_NAME = '$PARENT_NAME'
MUTATION_DESC = '$MUTATION_DESC'
# ---

print('Loading data...')
front, shortlist, contracts, contract_cat = load_data()

print('Precomputing volatilities...')
vols = precompute_vols(front, [VOL_WINDOW])
cumret_cache = build_norm_cumret(front, vols)

norm_ret, cumret = cumret_cache[VOL_WINDOW]
sig = compute_signal(SIGNAL_TYPE, cumret, PARAMS, SMOOTHING)
result = evaluate_by_category(sig, norm_ret, front, contracts, contract_cat)

if result is None:
    print('ERROR: Evaluation returned None (zero variance or no data)')
    sys.exit(1)

print(f'\\nResults for {SIGNAL_TYPE} vw={VOL_WINDOW} sm={SMOOTHING} {PARAMS}:')
print(f'  Sharpe:  {result[\"sharpe\"]:.4f}')
print(f'  Max DD:  {result[\"max_dd\"]:.4f}')
for k, v in sorted(result.items()):
    if k.startswith('sharpe_'):
        print(f'  {k}: {v:.4f}')
    elif k.startswith('cat_sharpe_'):
        print(f'  {k}: {v:.4f}')

# Save to append later
child = {
    'signal_type': SIGNAL_TYPE,
    'vol_window': VOL_WINDOW,
    'smoothing': SMOOTHING,
    'param1': list(PARAMS.values())[0],
    'param2': list(PARAMS.values())[1] if len(PARAMS) > 1 else None,
    **PARAMS,
    **result,
    'parent': PARENT_NAME,
    'mutation': MUTATION_DESC,
}
Path('/tmp/sweep_child_result.json').write_text(json.dumps(child, indent=2, default=str))
print(f'\\nChild result saved to /tmp/sweep_child_result.json')
"
```

Replace the `$PLACEHOLDERS` with actual values from your mutation proposal.

## Step 5: Report and Compare

After evaluation, report:
- The child's Sharpe and max_dd
- How it compares to the parent (improvement/regression on each metric)
- Per-decade stability vs parent
- Whether it would enter the Pareto frontier

Optionally, ask the user if they want to append the result to `sweep_results/singles.parquet`:

```
uv run python -c "
import json, pandas as pd
from pathlib import Path

child = json.loads(Path('/tmp/sweep_child_result.json').read_text())
# Remove non-standard keys before appending
for k in ['parent', 'mutation']:
    child.pop(k, None)

outfile = Path('sweep_results/singles.parquet')
df = pd.read_parquet(outfile)
child_df = pd.DataFrame([child])
df = pd.concat([df, child_df], ignore_index=True)
df.to_parquet(outfile, index=False)
print(f'Appended child to {outfile} (now {len(df)} rows)')
"
```

## Step 6: Repeat

If the user requested N > 1 sweeps, loop back to Step 1 (recompute the frontier with the new
result included) and repeat.

## Important Notes

- Always use `uv run` to execute Python, never bare `python`
- The data loading + vol precomputation takes ~10-20s — this is expected
- Keep the `slow > 2 * fast` constraint for ema_cross and dual_ma signals
- Present results concisely: frontier table, parent selection, mutation reasoning, outcome
- If a mutation fails, diagnose the error before retrying (likely a constraint violation)
