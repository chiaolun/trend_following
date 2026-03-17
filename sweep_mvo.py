#!/usr/bin/env python
"""
Mean-Variance Optimization (MVO) for trend-following signals.

Uses the signal vector as an expected-return proxy and a regularized
exponentially-weighted covariance matrix to determine contract weights.

Usage:
    uv run python sweep_mvo.py                          # defaults
    uv run python sweep_mvo.py --reg 0.50 --halflife 126
    uv run python sweep_mvo.py --grid                   # run full grid search
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms as mtrans
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent))
from sweep import (
    LEVERAGE,
    START_YEAR,
    build_norm_cumret,
    compute_signal,
    load_data,
    precompute_vols,
)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_SIGNAL_TYPE = "momentum"
DEFAULT_VOL_WINDOW = 60
DEFAULT_SMOOTHING = 1
DEFAULT_SIGNAL_PARAMS = {"window": 247}

DEFAULT_HALFLIFE = 126
DEFAULT_REG = 0.50
WARMUP_YEARS = 5
WARMUP_DAYS = 252 * WARMUP_YEARS

OUTPUT_DIR = Path("gepa")


# ---------------------------------------------------------------------------
# Core MVO
# ---------------------------------------------------------------------------

def compute_mvo_weights(nr_filled, sig_wide, halflife, reg, warmup=WARMUP_DAYS):
    """
    Compute daily MVO weight matrix.

    Parameters
    ----------
    nr_filled : DataFrame (dates x contracts)
        Vol-normalized returns with NaNs filled to 0.
    sig_wide : DataFrame (dates x contracts)
        Signal values (±1 or continuous).
    halflife : int
        Exponential decay halflife in days for covariance estimation.
    reg : float
        Regularization parameter λ in [0, 1].
        Σ_reg = (1-λ)Σ_sample + λ·diag(Σ_sample)
    warmup : int
        Number of days before MVO weights are used (equal-weight during warmup).

    Returns
    -------
    weights : ndarray (n_dates x n_contracts)
    """
    n_dates, n_contracts = nr_filled.shape
    ewm_alpha = 1 - np.exp(-np.log(2) / halflife)

    ewm_mean = np.zeros(n_contracts)
    ewm_cov = np.eye(n_contracts)
    weights = np.zeros((n_dates, n_contracts))

    for i in range(n_dates):
        ret_i = nr_filled[i]
        sig_i = sig_wide[i]

        # Compute weights BEFORE updating covariance with today's return,
        # so the cov matrix only uses returns through day i-1.
        if i < warmup:
            weights[i] = sig_i
        else:
            # Regularize: shrink toward diagonal
            diag_cov = np.diag(np.diag(ewm_cov))
            cov_reg = (1 - reg) * ewm_cov + reg * diag_cov

            # Solve w = Σ^{-1} μ
            try:
                w = np.linalg.solve(cov_reg, sig_i)
            except np.linalg.LinAlgError:
                w = sig_i

            # Normalize to match equal-weight gross exposure
            gross = np.abs(w).sum()
            if gross > 0:
                w = w / gross * np.abs(sig_i).sum()

            weights[i] = w

        # Update covariance AFTER computing weights (lagged: uses returns through day i)
        diff = ret_i - ewm_mean
        ewm_mean = ewm_mean * (1 - ewm_alpha) + ret_i * ewm_alpha
        ewm_cov = (1 - ewm_alpha) * (ewm_cov + ewm_alpha * np.outer(diff, diff))

    return weights


def compute_portfolio_returns(weights, nr_filled, index, eval_start):
    """Compute scaled portfolio returns from weight and return matrices."""
    port_ret = pd.Series(
        (weights * nr_filled).sum(axis=1),
        index=index,
    )[eval_start:]
    if port_ret.std() > 0:
        port_ret = port_ret * (LEVERAGE / port_ret.std())
    return port_ret


def portfolio_stats(ret):
    """Return (sharpe, max_dd) for a daily return series."""
    sharpe = ret.mean() / ret.std() * np.sqrt(252)
    cum = ret.cumsum()
    max_dd = (cum - cum.cummax()).min()
    return sharpe, max_dd


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_data(signal_type=DEFAULT_SIGNAL_TYPE, vol_window=DEFAULT_VOL_WINDOW,
                 smoothing=DEFAULT_SMOOTHING, signal_params=None):
    """Load data and compute wide-format returns and signals."""
    if signal_params is None:
        signal_params = DEFAULT_SIGNAL_PARAMS

    print("Loading data...")
    front, shortlist, contracts, contract_cat = load_data()

    print("Precomputing volatilities...")
    vols = precompute_vols(front, [vol_window])
    cumret_cache = build_norm_cumret(front, vols)
    norm_ret, cumret = cumret_cache[vol_window]

    sig = compute_signal(signal_type, cumret, signal_params, smoothing)

    # Unstack to date x contract
    nr_wide = norm_ret.loc[contracts].unstack("contract").sort_index()
    sig_wide = sig.loc[contracts].unstack("contract").sort_index()

    common_idx = nr_wide.index.intersection(sig_wide.index)
    nr_wide = nr_wide.loc[common_idx]
    sig_wide = sig_wide.loc[common_idx]

    eval_start = nr_wide.index[WARMUP_DAYS]
    print(f"Contracts: {len(nr_wide.columns)}, "
          f"dates: {len(nr_wide)}, "
          f"eval from: {eval_start.strftime('%Y-%m-%d')}")

    return nr_wide, sig_wide, eval_start


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

def run_single(halflife, reg, nr_wide, sig_wide, eval_start):
    """Run MVO with given parameters, return scaled daily returns."""
    nr_vals = nr_wide.fillna(0).values
    sig_vals = np.nan_to_num(sig_wide.values)

    weights = compute_mvo_weights(nr_vals, sig_vals, halflife, reg)
    return compute_portfolio_returns(weights, nr_vals, nr_wide.index, eval_start)


def run_equal_weight(nr_wide, sig_wide, eval_start):
    """Run equal-weight baseline (signal as weight)."""
    nr_vals = nr_wide.fillna(0).values
    sig_vals = np.nan_to_num(sig_wide.values)
    return compute_portfolio_returns(sig_vals, nr_vals, nr_wide.index, eval_start)


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------

def run_grid(nr_wide, sig_wide, eval_start):
    """Run full regularization × halflife grid, print results."""
    eq_ret = run_equal_weight(nr_wide, sig_wide, eval_start)
    s_eq, dd_eq = portfolio_stats(eq_ret)
    print(f"\nEqual-weight: Sharpe={s_eq:.4f}, MaxDD={dd_eq:.4f}")

    regs = [0.30, 0.50, 0.70, 0.80, 0.90, 0.95]
    halflives = [63, 126, 252]

    print(f"\n{'reg':>5}  {'hl=63':>12}  {'hl=126':>12}  {'hl=252':>12}")
    results = {}
    for reg in regs:
        row = f"{reg:5.2f}"
        for hl in halflives:
            ret = run_single(hl, reg, nr_wide, sig_wide, eval_start)
            s, dd = portfolio_stats(ret)
            row += f"  {s:5.2f}/{dd:6.3f}"
            results[(hl, reg)] = ret
        print(row)

    return results, eq_ret


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_comparison(results, output_dir=OUTPUT_DIR):
    """Plot cumulative returns, drawdowns, rolling Sharpe."""
    colors = {}
    color_cycle = ["C0", "C2", "C3", "C1"]
    for i, label in enumerate(results):
        colors[label] = color_cycle[i % len(color_cycle)]

    fig, axes = plt.subplots(3, 1, figsize=(14, 14))

    ax = axes[0]
    for label, ret in results.items():
        lw = 2 if "Equal" in label else 1.5
        ls = "--" if "Equal" in label else "-"
        ax.plot(ret.cumsum().index, ret.cumsum().values,
                label=label, lw=lw, ls=ls, color=colors[label])
    ax.set_title("MVO vs Equal-Weight (5yr cov warmup)", fontsize=12)
    ax.set_ylabel("Cumulative Return")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for label, ret in results.items():
        cum = ret.cumsum()
        dd = cum - cum.cummax()
        lw = 2 if "Equal" in label else 1.5
        ls = "--" if "Equal" in label else "-"
        ax.plot(dd.index, dd.values, label=label, lw=lw, ls=ls, color=colors[label])
    ax.set_title("Drawdowns")
    ax.set_ylabel("Drawdown")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    for label, ret in results.items():
        rs = ret.rolling(252 * 5).mean() / ret.rolling(252 * 5).std() * np.sqrt(252)
        lw = 2 if "Equal" in label else 1.5
        ls = "--" if "Equal" in label else "-"
        ax.plot(rs.index, rs.values, label=label, lw=lw, ls=ls, color=colors[label])
    ax.axhline(0, color="black", lw=0.5)
    ax.set_title("Rolling 5-Year Sharpe")
    ax.set_ylabel("Sharpe")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = output_dir / "gepa_mvo_vs_equal.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def plot_monthly(ret, label, output_dir=OUTPUT_DIR):
    """Plot monthly returns heatmap."""
    sharpe, dd = portfolio_stats(ret)

    monthly = ret.resample("ME").sum()
    mdf = pd.DataFrame({
        "year": monthly.index.year,
        "month": monthly.index.month,
        "ret": monthly.values,
    })
    pivot = mdf.pivot_table(values="ret", index="year", columns="month", aggfunc="sum")
    pivot.columns = [f"{m:02d}" for m in pivot.columns]
    annual_ret = pivot.sum(axis=1).rename("Return")
    annual_sharpe = (
        pivot.mean(axis=1) / pivot.std(axis=1) * np.sqrt(12)
    ).rename("Sharpe")
    annuals = pd.concat([annual_ret, annual_sharpe], axis=1)

    fig, (ax1, ax2, ax3) = plt.subplots(
        ncols=3, sharey=True, figsize=(14, 8),
        width_ratios=[12, 1, 1], gridspec_kw={"wspace": 0.05},
    )
    common = dict(cmap="RdBu", center=0, cbar=False, annot=True,
                  annot_kws={"ha": "right"})

    sns.heatmap(pivot, ax=ax1, fmt=".1%", **common)
    ax1.set_title(f"MVO Monthly Returns: {label} (S={sharpe:.3f}, DD={dd:.3f})")
    ax1.set_xlabel("Month")
    ax1.set_ylabel("Year")

    sns.heatmap(annuals[["Return"]], ax=ax2, fmt=".1%", **common)
    ax2.set_title("Return")
    ax2.set_xlabel("")
    ax2.set_ylabel("")
    ax2.tick_params(left=False)

    sns.heatmap(annuals[["Sharpe"]], ax=ax3, fmt=".2f", **common)
    ax3.set_title("Sharpe")
    ax3.set_xlabel("")
    ax3.set_ylabel("")
    ax3.tick_params(left=False)

    for t in (t for ax in [ax1, ax2, ax3] for t in ax.texts):
        trans = t.get_transform()
        offs = mtrans.ScaledTranslation(0.45, 0.0, mtrans.IdentityTransform())
        t.set_transform(offs + trans)

    plt.tight_layout()
    out = output_dir / "gepa_mvo_monthly.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_signal_params(args):
    """Build signal params dict from CLI args."""
    if args.signal_type in ("momentum", "breakout"):
        window = args.window if args.window is not None else DEFAULT_SIGNAL_PARAMS.get("window", 247)
        return {"window": window}
    elif args.signal_type == "cumsum_ma":
        lookback = args.window if args.window is not None else 130
        return {"lookback": lookback}
    elif args.signal_type == "ema_cross":
        fast = args.window if args.window is not None else 20
        slow = args.param2 if args.param2 is not None else 100
        return {"fast_span": fast, "slow_span": slow}
    elif args.signal_type == "dual_ma":
        fast = args.window if args.window is not None else 50
        slow = args.param2 if args.param2 is not None else 200
        return {"fast_window": fast, "slow_window": slow}
    return DEFAULT_SIGNAL_PARAMS


def main():
    parser = argparse.ArgumentParser(description="MVO trend-following optimization")

    # Signal parameters
    parser.add_argument("--signal-type", default=DEFAULT_SIGNAL_TYPE,
                        choices=["momentum", "breakout", "cumsum_ma", "ema_cross", "dual_ma"],
                        help=f"Signal type (default: {DEFAULT_SIGNAL_TYPE})")
    parser.add_argument("--vol-window", type=int, default=DEFAULT_VOL_WINDOW,
                        help=f"Volatility estimation window in days (default: {DEFAULT_VOL_WINDOW})")
    parser.add_argument("--smoothing", type=int, default=DEFAULT_SMOOTHING,
                        help=f"Signal smoothing window (default: {DEFAULT_SMOOTHING})")
    parser.add_argument("--window", type=int, default=None,
                        help="Signal window/lookback param1 (default: 247 for momentum)")
    parser.add_argument("--param2", type=int, default=None,
                        help="Signal param2 (slow_span/slow_window for ema_cross/dual_ma)")

    # MVO parameters
    parser.add_argument("--reg", type=float, default=DEFAULT_REG,
                        help=f"Regularization λ (default: {DEFAULT_REG})")
    parser.add_argument("--halflife", type=int, default=DEFAULT_HALFLIFE,
                        help=f"EWM covariance halflife in days (default: {DEFAULT_HALFLIFE})")
    parser.add_argument("--warmup-years", type=int, default=WARMUP_YEARS,
                        help=f"Minimum years of data before MVO kicks in (default: {WARMUP_YEARS})")

    # Modes
    parser.add_argument("--grid", action="store_true",
                        help="Run full regularization × halflife grid")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip plot generation")
    args = parser.parse_args()

    # Build signal params dict from CLI args
    signal_params = _build_signal_params(args)

    global WARMUP_DAYS
    WARMUP_DAYS = 252 * args.warmup_years

    OUTPUT_DIR.mkdir(exist_ok=True)

    print(f"Signal: {args.signal_type} vw={args.vol_window} sm={args.smoothing} {signal_params}")
    print(f"MVO: halflife={args.halflife} reg={args.reg} warmup={args.warmup_years}yr")

    nr_wide, sig_wide, eval_start = prepare_data(
        signal_type=args.signal_type,
        vol_window=args.vol_window,
        smoothing=args.smoothing,
        signal_params=signal_params,
    )

    if args.grid:
        grid_results, eq_ret = run_grid(nr_wide, sig_wide, eval_start)

        if not args.no_plot:
            # Pick representative configs for the comparison plot
            plot_results = {"Equal-weight": eq_ret}
            for hl, reg, label in [
                (126, 0.50, "MVO reg=0.50 hl=126"),
                (252, 0.80, "MVO reg=0.80 hl=252"),
                (252, 0.95, "MVO reg=0.95 hl=252"),
            ]:
                if (hl, reg) in grid_results:
                    plot_results[label] = grid_results[(hl, reg)]
            plot_comparison(plot_results)
            plot_monthly(grid_results.get((252, 0.80), eq_ret), "reg=0.80 hl=252")
    else:
        eq_ret = run_equal_weight(nr_wide, sig_wide, eval_start)
        mvo_ret = run_single(args.halflife, args.reg, nr_wide, sig_wide, eval_start)

        s_eq, dd_eq = portfolio_stats(eq_ret)
        s_mvo, dd_mvo = portfolio_stats(mvo_ret)

        print(f"\nEqual-weight: Sharpe={s_eq:.4f}, MaxDD={dd_eq:.4f}")
        print(f"MVO (hl={args.halflife}, λ={args.reg}): "
              f"Sharpe={s_mvo:.4f}, MaxDD={dd_mvo:.4f}")

        if not args.no_plot:
            label = f"MVO reg={args.reg} hl={args.halflife}"
            plot_comparison({"Equal-weight": eq_ret, label: mvo_ret})
            plot_monthly(mvo_ret, label)


if __name__ == "__main__":
    main()
