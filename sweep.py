#!/workspace/.venv/bin/python
"""
Trend following parameter sweep.

Explores signal types, lookback windows, smoothing, and volatility estimation
across 5 futures categories (softs, hards, bonds, equities, forex).
Parameters are shared within each category.

Output: sweep_results/*.parquet + sweep_results/progress.json
"""

import argparse
import itertools
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data():
    front = pd.read_parquet("front.parquet")
    shortlist = pd.read_parquet("shortlist.parquet")

    # Build contract -> category mapping
    contract_cat = shortlist.set_index("contract")["category"]
    front["category"] = front.index.get_level_values("contract").map(contract_cat)
    contracts = shortlist.contract.tolist()

    return front, shortlist, contracts, contract_cat


# ---------------------------------------------------------------------------
# Volatility precomputation
# ---------------------------------------------------------------------------

def precompute_vols(front, vol_windows):
    """Precompute vol for each window. Returns {window: Series}."""
    abs_dc = front.d_close.abs()
    vols = {}
    for w in tqdm(vol_windows, desc="Precomputing vol windows"):
        vols[w] = (
            abs_dc.groupby("contract").rolling(w).mean()
            .reset_index(level=0, drop=True)
            .groupby("contract").shift()
        )
    return vols


def build_norm_cumret(front, vols):
    """For each vol window, compute norm_ret and cumret. Returns {w: (norm_ret, cumret)}."""
    cache = {}
    for w, vol in tqdm(vols.items(), desc="Building cumret cache"):
        nr = front.d_close / vol
        cr = nr.groupby("contract").cumsum()
        cache[w] = (nr, cr)
    return cache


# ---------------------------------------------------------------------------
# Signal generators
# ---------------------------------------------------------------------------

def _groll_mean(s, window):
    """groupby('contract').rolling(window).mean(), dropping extra level."""
    return s.groupby("contract").rolling(window).mean().reset_index(level=0, drop=True)


def _groll_max(s, window):
    return s.groupby("contract").rolling(window).max().reset_index(level=0, drop=True)


def _groll_min(s, window):
    return s.groupby("contract").rolling(window).min().reset_index(level=0, drop=True)


def _apply_smoothing(raw_signal, smoothing):
    if smoothing > 1:
        return _groll_mean(raw_signal, smoothing)
    return raw_signal


def signal_cumsum_ma(cumret, lookback, smoothing):
    """Current strategy: price above its own rolling MA."""
    ma = _groll_mean(cumret, lookback)
    raw = cumret.gt(ma).groupby("contract").shift(fill_value=False)
    raw = raw.astype(float) * 2 - 1
    return _apply_smoothing(raw, smoothing)


def signal_breakout(cumret, window, smoothing):
    """Donchian channel: position within N-day high/low range → [-1, +1]."""
    high = _groll_max(cumret, window)
    low = _groll_min(cumret, window)
    rng = high - low
    raw = (2 * (cumret - low) / rng - 1).clip(-1, 1)
    raw = raw.groupby("contract").shift(fill_value=0)
    return _apply_smoothing(raw, smoothing)


def signal_ema_cross(cumret, fast_span, slow_span, smoothing):
    """EMA crossover: fast EMA > slow EMA."""
    ema_fast = cumret.groupby("contract").transform(
        lambda x: x.ewm(span=fast_span, min_periods=fast_span).mean()
    )
    ema_slow = cumret.groupby("contract").transform(
        lambda x: x.ewm(span=slow_span, min_periods=slow_span).mean()
    )
    raw = ema_fast.gt(ema_slow).groupby("contract").shift(fill_value=False)
    raw = raw.astype(float) * 2 - 1
    return _apply_smoothing(raw, smoothing)


def signal_dual_ma(cumret, fast_window, slow_window, smoothing):
    """Dual SMA crossover: fast SMA > slow SMA."""
    ma_fast = _groll_mean(cumret, fast_window)
    ma_slow = _groll_mean(cumret, slow_window)
    raw = ma_fast.gt(ma_slow).groupby("contract").shift(fill_value=False)
    raw = raw.astype(float) * 2 - 1
    return _apply_smoothing(raw, smoothing)


def signal_momentum(cumret, window, smoothing):
    """Rate of change: cumret > cumret.shift(N)."""
    raw = cumret.gt(cumret.groupby("contract").shift(window))
    raw = raw.groupby("contract").shift(fill_value=False)
    raw = raw.astype(float) * 2 - 1
    return _apply_smoothing(raw, smoothing)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

START_YEAR = "1980"
LEVERAGE = 0.006


def evaluate_by_category(signal, norm_ret, front, contracts, contract_cat):
    """
    Evaluate signal per category and overall.
    Returns dict with sharpe, per-decade sharpes, max_dd, and per-category sharpes.
    """
    rets = norm_ret * signal
    port_rets = rets.loc[contracts].groupby("date").sum()
    pnl = port_rets[START_YEAR:]

    if len(pnl) == 0 or pnl.std() == 0:
        return None

    risk = LEVERAGE / pnl.std()
    pnl_scaled = pnl * risk

    overall_sharpe = pnl_scaled.mean() / pnl_scaled.std() * 252**0.5

    # Per-decade
    decade = pnl_scaled.index.get_level_values("date").year // 10 * 10
    decade_sharpes = {}
    for dec, grp in pnl_scaled.groupby(decade):
        if len(grp) > 100 and grp.std() > 0:
            decade_sharpes[int(dec)] = grp.mean() / grp.std() * 252**0.5

    # Max drawdown
    cum = pnl_scaled.cumsum()
    max_dd = (cum - cum.cummax()).min()

    # Per-category sharpe
    cat_sharpes = {}
    for cat in contract_cat.unique():
        cat_contracts = contract_cat[contract_cat == cat].index.tolist()
        cat_contracts = [c for c in cat_contracts if c in contracts]
        if not cat_contracts:
            continue
        cat_rets = rets.loc[cat_contracts].groupby("date").sum()[START_YEAR:]
        if len(cat_rets) == 0 or cat_rets.std() == 0:
            continue
        cat_risk = LEVERAGE / cat_rets.std()
        cat_pnl = cat_rets * cat_risk
        cat_sharpes[cat] = cat_pnl.mean() / cat_pnl.std() * 252**0.5

    result = {
        "sharpe": overall_sharpe,
        "max_dd": max_dd,
        "n_days": len(pnl),
    }
    for k, v in decade_sharpes.items():
        result[f"sharpe_{k}"] = v
    for k, v in cat_sharpes.items():
        result[f"cat_sharpe_{k}"] = v

    return result


# ---------------------------------------------------------------------------
# Parameter grids
# ---------------------------------------------------------------------------

VOL_WINDOWS = [60, 90, 120, 180, 250]
SMOOTHINGS = [1, 5, 10, 20, 30, 50, 75, 100]

GRIDS = {
    "cumsum_ma": {
        "lookback": [40, 60, 90, 120, 150, 180, 200, 250, 300, 350, 400, 500],
    },
    "breakout": {
        "window": [20, 40, 60, 90, 120, 150, 180, 250, 350, 500],
    },
    "ema_cross": {
        "fast_span": [5, 10, 20, 30, 50, 75],
        "slow_span": [50, 75, 100, 150, 200, 250, 350, 500],
    },
    "dual_ma": {
        "fast_window": [10, 20, 30, 50, 75, 100],
        "slow_window": [100, 150, 200, 250, 300, 400, 500],
    },
    "momentum": {
        "window": [20, 40, 60, 90, 120, 150, 180, 250, 350, 500],
    },
}


def generate_combos(signal_type):
    """Yield (vol_window, smoothing, params_dict) for a signal type."""
    grid = GRIDS[signal_type]
    param_names = list(grid.keys())

    if signal_type in ("ema_cross", "dual_ma"):
        # Filter: fast < slow and slow/fast >= 2
        p1_name, p2_name = param_names
        pairs = [
            (f, s) for f in grid[p1_name] for s in grid[p2_name]
            if f < s and s / f >= 2
        ]
        for vw in VOL_WINDOWS:
            for sm in SMOOTHINGS:
                for f, s in pairs:
                    yield vw, sm, {p1_name: f, p2_name: s}
    else:
        pname = param_names[0]
        for vw in VOL_WINDOWS:
            for sm in SMOOTHINGS:
                for val in grid[pname]:
                    yield vw, sm, {pname: val}


def compute_signal(signal_type, cumret, params, smoothing):
    """Dispatch to the right signal function."""
    if signal_type == "cumsum_ma":
        return signal_cumsum_ma(cumret, params["lookback"], smoothing)
    elif signal_type == "breakout":
        return signal_breakout(cumret, params["window"], smoothing)
    elif signal_type == "ema_cross":
        return signal_ema_cross(cumret, params["fast_span"], params["slow_span"], smoothing)
    elif signal_type == "dual_ma":
        return signal_dual_ma(cumret, params["fast_window"], params["slow_window"], smoothing)
    elif signal_type == "momentum":
        return signal_momentum(cumret, params["window"], smoothing)
    else:
        raise ValueError(f"Unknown signal type: {signal_type}")


# ---------------------------------------------------------------------------
# Phase 1: Single signal sweep
# ---------------------------------------------------------------------------

def run_singles(front, contracts, contract_cat, cumret_cache, output_dir):
    """Sweep all single-signal parameter combos."""
    outfile = output_dir / "singles.parquet"
    results = []
    done_keys = set()

    # Resume support
    if outfile.exists():
        existing = pd.read_parquet(outfile)
        results = existing.to_dict("records")
        for r in results:
            _p2 = r.get("param2", np.nan)
            done_keys.add((r["signal_type"], r["vol_window"], r["smoothing"],
                           r.get("param1"), _p2 if pd.notna(_p2) else -1))
        print(f"Resuming singles: {len(results)} already done")

    total = sum(1 for st in GRIDS for _ in generate_combos(st))
    pbar = tqdm(total=total, desc="Singles", initial=len(results))

    for signal_type in GRIDS:
        for vw, sm, params in generate_combos(signal_type):
            p1 = float(list(params.values())[0])
            p2 = float(list(params.values())[1]) if len(params) > 1 else np.nan
            p2_key = p2 if not np.isnan(p2) else -1
            key = (signal_type, vw, sm, p1, p2_key)
            if key in done_keys:
                pbar.update(1)
                continue

            norm_ret, cumret = cumret_cache[vw]
            try:
                sig = compute_signal(signal_type, cumret, params, sm)
                ev = evaluate_by_category(sig, norm_ret, front, contracts, contract_cat)
            except Exception as e:
                ev = None

            if ev is not None:
                row = {
                    "signal_type": signal_type,
                    "vol_window": vw,
                    "smoothing": sm,
                    "param1": p1,
                    "param2": p2,
                    **{k: v for k, v in params.items()},
                    **ev,
                }
                results.append(row)

            pbar.update(1)

            # Incremental save every 500
            if len(results) % 500 == 0:
                pd.DataFrame(results).to_parquet(outfile, index=False)

    pbar.close()
    df = pd.DataFrame(results)
    df.to_parquet(outfile, index=False)
    print(f"Singles complete: {len(df)} results saved")
    return df


# ---------------------------------------------------------------------------
# Phase 2: Per-category evaluation (same grid, separate eval per category)
# ---------------------------------------------------------------------------

def run_by_category(front, contracts, contract_cat, cumret_cache, output_dir):
    """
    For each category, find the best signal params independently.
    This evaluates each signal on each category's contracts alone.
    """
    outfile = output_dir / "by_category.parquet"
    results = []
    done_keys = set()

    if outfile.exists():
        existing = pd.read_parquet(outfile)
        results = existing.to_dict("records")
        for r in results:
            _p2 = r.get("param2", np.nan)
            done_keys.add((r["category"], r["signal_type"], r["vol_window"],
                           r["smoothing"], r.get("param1"), _p2 if pd.notna(_p2) else -1))
        print(f"Resuming by_category: {len(results)} already done")

    categories = sorted(contract_cat.unique())
    cat_contracts = {
        cat: [c for c in contracts if contract_cat.get(c) == cat]
        for cat in categories
    }

    n_combos = sum(1 for st in GRIDS for _ in generate_combos(st))
    total = n_combos * len(categories)
    pbar = tqdm(total=total, desc="By category", initial=len(results))

    for signal_type in GRIDS:
        for vw, sm, params in generate_combos(signal_type):
            p1 = float(list(params.values())[0])
            p2 = float(list(params.values())[1]) if len(params) > 1 else np.nan
            p2_key = p2 if not np.isnan(p2) else -1

            norm_ret, cumret = cumret_cache[vw]
            try:
                sig = compute_signal(signal_type, cumret, params, sm)
            except Exception:
                pbar.update(len(categories))
                continue

            for cat in categories:
                key = (cat, signal_type, vw, sm, p1, p2_key)
                if key in done_keys:
                    pbar.update(1)
                    continue

                cc = cat_contracts[cat]
                if not cc:
                    pbar.update(1)
                    continue

                try:
                    rets = (norm_ret * sig).loc[cc].groupby("date").sum()[START_YEAR:]
                    if len(rets) == 0 or rets.std() == 0:
                        pbar.update(1)
                        continue
                    r = LEVERAGE / rets.std()
                    pnl = rets * r
                    sharpe = pnl.mean() / pnl.std() * 252**0.5
                    cum = pnl.cumsum()
                    max_dd = (cum - cum.cummax()).min()

                    # decade sharpes
                    dec = pnl.index.get_level_values("date").year // 10 * 10
                    dec_sharpes = {}
                    for d, g in pnl.groupby(dec):
                        if len(g) > 100 and g.std() > 0:
                            dec_sharpes[int(d)] = g.mean() / g.std() * 252**0.5

                    results.append({
                        "category": cat,
                        "signal_type": signal_type,
                        "vol_window": vw,
                        "smoothing": sm,
                        "param1": p1,
                        "param2": p2,
                        **params,
                        "sharpe": sharpe,
                        "max_dd": max_dd,
                        **{f"sharpe_{k}": v for k, v in dec_sharpes.items()},
                    })
                except Exception:
                    pass

                pbar.update(1)

            if len(results) % 2000 == 0:
                pd.DataFrame(results).to_parquet(outfile, index=False)

    pbar.close()
    df = pd.DataFrame(results)
    df.to_parquet(outfile, index=False)
    print(f"By category complete: {len(df)} results saved")
    return df


# ---------------------------------------------------------------------------
# Phase 3: ES overlay sweep
# ---------------------------------------------------------------------------

def run_es_overlay(front, contracts, contract_cat, cumret_cache, output_dir,
                   singles_df):
    """Sweep ES overlay signal params, evaluate combined portfolio."""
    outfile = output_dir / "es_overlay.parquet"
    results = []

    if outfile.exists():
        existing = pd.read_parquet(outfile)
        results = existing.to_dict("records")
        print(f"Resuming es_overlay: {len(results)} already done")
        if len(results) > 0:
            return pd.DataFrame(results)

    # Get the best overall signal from singles to use as the base portfolio
    if singles_df is None or len(singles_df) == 0:
        print("No singles results, skipping ES overlay")
        return pd.DataFrame()

    best = singles_df.sort_values("sharpe", ascending=False).iloc[0]
    print(f"ES overlay base: {best.signal_type} sharpe={best.sharpe:.3f}")

    # Reconstruct the best portfolio signal
    best_vw = int(best.vol_window)
    best_sm = int(best.smoothing)
    best_params = {}
    for k in GRIDS[best.signal_type]:
        best_params[k] = int(best[k]) if k in best else int(best.param1)

    norm_ret_base, cumret_base = cumret_cache[best_vw]
    base_sig = compute_signal(best.signal_type, cumret_base, best_params, best_sm)
    base_rets = (norm_ret_base * base_sig).loc[contracts].groupby("date").sum()[START_YEAR:]
    base_risk = LEVERAGE / base_rets.std()
    base_pnl = base_rets * base_risk

    # ES overlay sweep
    es_lookbacks = [40, 60, 90, 120, 150, 200, 250, 300, 400]
    es_smoothings = [1, 10, 20, 30, 50, 75, 90, 120]
    es_vol_windows = VOL_WINDOWS

    combos = list(itertools.product(es_vol_windows, es_lookbacks, es_smoothings))
    for vw, lb, sm in tqdm(combos, desc="ES overlay"):
        try:
            norm_ret, cumret = cumret_cache[vw]
            es_cumret = cumret.loc["ES"]
            es_norm_ret = norm_ret.loc["ES"]

            es_ma = es_cumret.rolling(lb).mean()
            raw = es_cumret.gt(es_ma).shift(fill_value=False).astype(float) * 2 - 1
            if sm > 1:
                raw = raw.rolling(sm).mean()

            # ES position: signal / vol, then scale
            es_vol = front.loc["ES", "d_close"].abs().rolling(vw).mean().shift()
            es_pos = raw / es_vol
            es_ret = es_pos * front.loc["ES", "d_close"]
            es_ret = es_ret[START_YEAR:]
            es_risk = LEVERAGE / es_ret.std()
            es_pnl = es_ret * es_risk

            # Combined
            combined = base_pnl.add(es_pnl, fill_value=0)
            combined_sharpe = combined.mean() / combined.std() * 252**0.5
            es_sharpe = es_pnl.mean() / es_pnl.std() * 252**0.5

            # Correlation
            aligned = pd.concat([base_pnl, es_pnl], axis=1).dropna()
            corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])

            cum = combined.cumsum()
            max_dd = (cum - cum.cummax()).min()

            results.append({
                "vol_window": vw,
                "lookback": lb,
                "smoothing": sm,
                "es_sharpe": es_sharpe,
                "combined_sharpe": combined_sharpe,
                "correlation": corr,
                "max_dd": max_dd,
            })
        except Exception:
            pass

    df = pd.DataFrame(results)
    df.to_parquet(outfile, index=False)
    print(f"ES overlay complete: {len(df)} results saved")
    return df


# ---------------------------------------------------------------------------
# Phase 4: Blend top singles
# ---------------------------------------------------------------------------

def run_blends(front, contracts, contract_cat, cumret_cache, output_dir,
               singles_df):
    """Blend top-K single signals pairwise."""
    outfile = output_dir / "blends.parquet"
    results = []

    if outfile.exists():
        existing = pd.read_parquet(outfile)
        results = existing.to_dict("records")
        print(f"Resuming blends: {len(results)} already done")
        if len(results) > 0:
            return pd.DataFrame(results)

    if singles_df is None or len(singles_df) == 0:
        print("No singles results, skipping blends")
        return pd.DataFrame()

    # Take top 40 by sharpe
    top = singles_df.sort_values("sharpe", ascending=False).head(40).reset_index(drop=True)

    # Precompute signals for top entries
    print("Precomputing top signal series...")
    sig_cache = {}
    for idx, row in tqdm(top.iterrows(), total=len(top), desc="Signal cache"):
        vw = int(row.vol_window)
        sm = int(row.smoothing)
        params = {}
        for k in GRIDS[row.signal_type]:
            if k in row.index:
                params[k] = int(row[k])

        norm_ret, cumret = cumret_cache[vw]
        sig = compute_signal(row.signal_type, cumret, params, sm)
        # Store the per-contract daily returns
        sig_rets = (norm_ret * sig).loc[contracts].groupby("date").sum()[START_YEAR:]
        sig_cache[idx] = sig_rets

    # Pairwise blends with equal weight
    pairs = list(itertools.combinations(range(len(top)), 2))
    for i, j in tqdm(pairs, desc="Blends"):
        try:
            blended = (sig_cache[i] + sig_cache[j]) / 2
            if blended.std() == 0:
                continue
            r = LEVERAGE / blended.std()
            pnl = blended * r
            sharpe = pnl.mean() / pnl.std() * 252**0.5
            cum = pnl.cumsum()
            max_dd = (cum - cum.cummax()).min()

            results.append({
                "idx_a": i,
                "idx_b": j,
                "type_a": top.loc[i, "signal_type"],
                "type_b": top.loc[j, "signal_type"],
                "sharpe_a": top.loc[i, "sharpe"],
                "sharpe_b": top.loc[j, "sharpe"],
                "blend_sharpe": sharpe,
                "max_dd": max_dd,
                "improvement": sharpe - max(top.loc[i, "sharpe"], top.loc[j, "sharpe"]),
            })
        except Exception:
            pass

    df = pd.DataFrame(results)
    df.to_parquet(outfile, index=False)
    print(f"Blends complete: {len(df)} results saved")
    return df


# ---------------------------------------------------------------------------
# Phase 5: Bootstrap stability for top signals
# ---------------------------------------------------------------------------

def run_bootstrap(front, contracts, contract_cat, cumret_cache, output_dir,
                  singles_df, n_boot=1000, block_years=1):
    """Block bootstrap to estimate Sharpe confidence intervals."""
    outfile = output_dir / "bootstrap.parquet"

    if outfile.exists():
        existing = pd.read_parquet(outfile)
        print(f"Bootstrap already done: {len(existing)} results")
        return existing

    if singles_df is None or len(singles_df) == 0:
        print("No singles results, skipping bootstrap")
        return pd.DataFrame()

    top = singles_df.sort_values("sharpe", ascending=False).head(100).reset_index(drop=True)
    rng = np.random.default_rng(42)

    results = []
    for idx, row in tqdm(top.iterrows(), total=len(top), desc="Bootstrap"):
        vw = int(row.vol_window)
        sm = int(row.smoothing)
        params = {}
        for k in GRIDS[row.signal_type]:
            if k in row.index:
                params[k] = int(row[k])

        norm_ret, cumret = cumret_cache[vw]
        sig = compute_signal(row.signal_type, cumret, params, sm)
        rets = (norm_ret * sig).loc[contracts].groupby("date").sum()[START_YEAR:]

        # Build year blocks
        years = rets.index.get_level_values("date").year
        unique_years = sorted(years.unique())
        n_years = len(unique_years)
        blocks = [rets[years == y].values for y in unique_years]

        boot_sharpes = []
        for _ in range(n_boot):
            idxs = rng.integers(0, n_years, size=n_years)
            sample = np.concatenate([blocks[i] for i in idxs])
            if sample.std() == 0:
                continue
            s = sample.mean() / sample.std() * 252**0.5
            boot_sharpes.append(s)

        boot_sharpes = np.array(boot_sharpes)
        results.append({
            "signal_type": row.signal_type,
            "vol_window": vw,
            "smoothing": sm,
            "param1": row.param1,
            "param2": row.get("param2", np.nan),
            "sharpe": row.sharpe,
            "boot_mean": boot_sharpes.mean(),
            "boot_std": boot_sharpes.std(),
            "boot_5pct": np.percentile(boot_sharpes, 5),
            "boot_25pct": np.percentile(boot_sharpes, 25),
            "boot_50pct": np.percentile(boot_sharpes, 50),
            "boot_75pct": np.percentile(boot_sharpes, 75),
            "boot_95pct": np.percentile(boot_sharpes, 95),
        })

    df = pd.DataFrame(results)
    df.to_parquet(outfile, index=False)
    print(f"Bootstrap complete: {len(df)} results saved")
    return df


# ---------------------------------------------------------------------------
# Phase 6: Dense grid around best regions
# ---------------------------------------------------------------------------

def run_dense_grid(front, contracts, contract_cat, cumret_cache, output_dir,
                   singles_df):
    """Finer grid around the top-performing parameter regions."""
    outfile = output_dir / "dense.parquet"
    results = []

    if outfile.exists():
        existing = pd.read_parquet(outfile)
        results = existing.to_dict("records")
        print(f"Resuming dense: {len(results)} already done")
        if len(results) > 0:
            return pd.DataFrame(results)

    if singles_df is None or len(singles_df) == 0:
        print("No singles results, skipping dense grid")
        return pd.DataFrame()

    # For each signal type, find the best combo and sweep densely around it
    for stype in GRIDS:
        subset = singles_df[singles_df.signal_type == stype]
        if len(subset) == 0:
            continue
        best = subset.sort_values("sharpe", ascending=False).iloc[0]

        best_vw = int(best.vol_window)
        best_sm = int(best.smoothing)
        best_p1 = int(best.param1)
        best_p2 = int(best.param2) if pd.notna(best.param2) else None

        # Dense ranges: ±30% of best values, finer steps
        def dense_range(center, lo_frac=0.5, hi_frac=1.5, steps=20):
            lo = max(2, int(center * lo_frac))
            hi = int(center * hi_frac)
            return sorted(set(range(lo, hi + 1, max(1, (hi - lo) // steps))))

        vw_range = dense_range(best_vw)
        sm_range = dense_range(best_sm)
        p1_range = dense_range(best_p1)

        # Only sweep the 3 most impactful dimensions at a time
        # Fix vol_window to best, sweep (param1, smoothing)
        combos = list(itertools.product(p1_range, sm_range))
        for p1, sm in tqdm(combos, desc=f"Dense {stype}"):
            try:
                norm_ret, cumret = cumret_cache[best_vw]
                params = {list(GRIDS[stype].keys())[0]: p1}
                if best_p2 is not None:
                    params[list(GRIDS[stype].keys())[1]] = best_p2
                sig = compute_signal(stype, cumret, params, sm)
                ev = evaluate_by_category(sig, norm_ret, front, contracts, contract_cat)
                if ev:
                    results.append({
                        "signal_type": stype,
                        "vol_window": best_vw,
                        "smoothing": sm,
                        "param1": p1,
                        "param2": float(best_p2) if best_p2 is not None else np.nan,
                        **params,
                        **ev,
                    })
            except Exception:
                pass

    df = pd.DataFrame(results)
    df.to_parquet(outfile, index=False)
    print(f"Dense grid complete: {len(df)} results saved")
    return df


# ---------------------------------------------------------------------------
# Phase 7: Composite portfolio (best params per category)
# ---------------------------------------------------------------------------

def run_composite(front, contracts, contract_cat, cumret_cache, output_dir,
                  by_cat_df):
    """
    Build composite portfolio using best signal per category.
    Save the composite PnL and per-category allocations.
    """
    outfile = output_dir / "composite.parquet"
    pnl_file = output_dir / "composite_pnl.parquet"

    if outfile.exists():
        print("Composite already done")
        return pd.read_parquet(outfile)

    if by_cat_df is None or len(by_cat_df) == 0:
        print("No by_category results, skipping composite")
        return pd.DataFrame()

    categories = sorted(by_cat_df.category.unique())
    composite_info = []
    category_pnls = {}

    for cat in categories:
        cat_df = by_cat_df[by_cat_df.category == cat]
        best = cat_df.sort_values("sharpe", ascending=False).iloc[0]
        vw = int(best.vol_window)
        sm = int(best.smoothing)
        params = {}
        for k in GRIDS[best.signal_type]:
            if k in best.index:
                params[k] = int(best[k])

        norm_ret, cumret = cumret_cache[vw]
        sig = compute_signal(best.signal_type, cumret, params, sm)

        cc = [c for c in contracts if contract_cat.get(c) == cat]
        rets = (norm_ret * sig).loc[cc].groupby("date").sum()[START_YEAR:]
        if len(rets) > 0 and rets.std() > 0:
            r = LEVERAGE / rets.std()
            category_pnls[cat] = rets * r

        composite_info.append({
            "category": cat,
            "signal_type": best.signal_type,
            "vol_window": vw,
            "smoothing": sm,
            **params,
            "sharpe": best.sharpe,
        })

    # Combined composite
    all_pnl = pd.DataFrame(category_pnls)
    combined = all_pnl.sum(axis=1)
    if combined.std() > 0:
        composite_sharpe = combined.mean() / combined.std() * 252**0.5
    else:
        composite_sharpe = 0

    info_df = pd.DataFrame(composite_info)
    info_df.loc[len(info_df)] = {
        "category": "COMPOSITE",
        "signal_type": "blend",
        "sharpe": composite_sharpe,
    }
    info_df.to_parquet(outfile, index=False)

    # Save the daily PnL
    all_pnl["composite"] = combined
    all_pnl.to_parquet(pnl_file)

    print(f"Composite: {composite_sharpe:.3f} Sharpe")
    print(info_df.to_string())
    return info_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def save_progress(output_dir, phase, status):
    pfile = output_dir / "progress.json"
    progress = {}
    if pfile.exists():
        progress = json.loads(pfile.read_text())
    progress[phase] = {"status": status, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    pfile.write_text(json.dumps(progress, indent=2))


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print("Loading data...")
    front, shortlist, contracts, contract_cat = load_data()

    print("Precomputing volatilities...")
    vols = precompute_vols(front, VOL_WINDOWS)
    cumret_cache = build_norm_cumret(front, vols)

    phases = args.phases
    if "all" in phases:
        phases = ["singles", "by_category", "es_overlay", "blends",
                  "bootstrap", "dense", "composite"]

    singles_df = None
    by_cat_df = None

    for phase in phases:
        print(f"\n{'='*60}")
        print(f"Phase: {phase}")
        print(f"{'='*60}")
        save_progress(output_dir, phase, "running")

        if phase == "singles":
            singles_df = run_singles(front, contracts, contract_cat, cumret_cache, output_dir)
        elif phase == "by_category":
            by_cat_df = run_by_category(front, contracts, contract_cat, cumret_cache, output_dir)
        elif phase == "es_overlay":
            if singles_df is None and (output_dir / "singles.parquet").exists():
                singles_df = pd.read_parquet(output_dir / "singles.parquet")
            run_es_overlay(front, contracts, contract_cat, cumret_cache, output_dir, singles_df)
        elif phase == "blends":
            if singles_df is None and (output_dir / "singles.parquet").exists():
                singles_df = pd.read_parquet(output_dir / "singles.parquet")
            run_blends(front, contracts, contract_cat, cumret_cache, output_dir, singles_df)
        elif phase == "bootstrap":
            if singles_df is None and (output_dir / "singles.parquet").exists():
                singles_df = pd.read_parquet(output_dir / "singles.parquet")
            run_bootstrap(front, contracts, contract_cat, cumret_cache, output_dir, singles_df)
        elif phase == "dense":
            if singles_df is None and (output_dir / "singles.parquet").exists():
                singles_df = pd.read_parquet(output_dir / "singles.parquet")
            run_dense_grid(front, contracts, contract_cat, cumret_cache, output_dir, singles_df)
        elif phase == "composite":
            if by_cat_df is None and (output_dir / "by_category.parquet").exists():
                by_cat_df = pd.read_parquet(output_dir / "by_category.parquet")
            run_composite(front, contracts, contract_cat, cumret_cache, output_dir, by_cat_df)

        save_progress(output_dir, phase, "complete")

    print("\n\nAll phases complete!")
    save_progress(output_dir, "all", "complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trend following parameter sweep")
    parser.add_argument("--phases", nargs="+", default=["all"],
                        choices=["singles", "by_category", "es_overlay", "blends",
                                 "bootstrap", "dense", "composite", "all"])
    parser.add_argument("--output-dir", default="sweep_results")
    args = parser.parse_args()
    main(args)
