# Mean-Variance Optimization (MVO)

Uses the momentum signal as an expected-return vector and a regularized
exponentially-weighted covariance matrix to determine contract weights,
rather than weighting each contract equally by its signal.

## Method

Each day, solve for the optimal weight vector:

```
w_t = Σ_reg,t⁻¹  μ_t
```

where:
- `μ_t` is the signal vector (±1 per contract from the momentum signal)
- `Σ_reg,t` is a regularized covariance matrix of vol-normalized returns, estimated using an exponentially-weighted moving window with a given halflife, using only returns through day t-1 (properly lagged — no lookahead)

Regularization shrinks the sample covariance toward its diagonal (zero cross-correlation):

```
Σ_reg = (1 - λ) Σ_sample + λ · diag(Σ_sample)
```

Higher λ → closer to equal-weight; lower λ → more aggressive covariance exploitation.

After solving, weights are rescaled so that gross exposure matches the equal-weight baseline.

## Warmup

A minimum of **5 years** (1260 trading days) of data is required before MVO weights are used. During warmup, equal-weight signal positions are used. Evaluation begins 1991-01-24.

## Signal parameters

All variants use the best momentum signal: `vw=60, sm=1, window=247`.

## Results (eval from 1991)

| Method               | Halflife | λ    | Sharpe | Max DD |
| -------------------- | -------- | ---- | ------ | ------ |
| Equal-weight         | —        | —    | 1.12   | -0.194 |
| MVO                  | 63       | 0.50 | 1.31   | -0.181 |
| MVO                  | 63       | 0.70 | 1.31   | -0.198 |
| MVO                  | 126      | 0.50 | 1.28   | -0.173 |
| MVO                  | 252      | 0.50 | 1.25   | -0.169 |

Full grid (Sharpe / Max DD):

| λ    | hl=63        | hl=126       | hl=252       |
| ---- | ------------ | ------------ | ------------ |
| 0.30 | 1.28 / -0.19 | 1.24 / -0.18 | 1.22 / -0.18 |
| 0.50 | 1.31 / -0.18 | 1.28 / -0.17 | 1.25 / -0.17 |
| 0.70 | 1.31 / -0.20 | 1.28 / -0.19 | 1.25 / -0.18 |
| 0.80 | 1.29 / -0.20 | 1.26 / -0.19 | 1.24 / -0.19 |
| 0.90 | 1.24 / -0.20 | 1.22 / -0.19 | 1.21 / -0.19 |
| 0.95 | 1.19 / -0.20 | 1.18 / -0.19 | 1.18 / -0.19 |

MVO improves Sharpe by 0.07–0.19 across the grid. The best Sharpe (1.31) comes from λ=0.50–0.70 with hl=63. The best DD (-0.169) comes from λ=0.50 hl=252. No setting degrades DD meaningfully versus equal-weight.

## Reproducing the best variant

```
uv run python sweep_mvo.py \
  --signal-type momentum --vol-window 60 --smoothing 1 --window 247 \
  --halflife 63 --reg 0.50 --warmup-years 5
```

## Plots

- [Cumulative returns, drawdowns, rolling Sharpe](gepa_mvo_vs_equal.png)
- [Monthly returns heatmap (reg=0.50 hl=63)](gepa_mvo_monthly.png)

## Takeaways

- MVO with properly lagged covariance improves Sharpe from 1.12 to ~1.31 (+17%) with comparable or slightly better drawdown.
- λ=0.50 with short halflife (63 days) maximizes Sharpe. Longer halflives (252) slightly reduce DD.
- The improvement is modest but consistent — the covariance matrix contains real exploitable structure, but much less than a lookahead-biased estimate would suggest.
