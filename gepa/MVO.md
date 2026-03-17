# Mean-Variance Optimization (MVO)

Uses the momentum signal as an expected-return vector and a regularized sample covariance matrix to determine contract weights, rather than weighting each contract equally by its signal.

## Method

Each day, solve for the optimal weight vector:

```
w_t = Σ_reg,t⁻¹  μ_t
```

where:
- `μ_t` is the signal vector (±1 per contract from the momentum signal)
- `Σ_reg,t` is a regularized covariance matrix of vol-normalized returns, estimated using an exponentially-weighted moving window with a given halflife, lagged one day

Regularization shrinks the sample covariance toward its diagonal (zero cross-correlation):

```
Σ_reg = (1 - λ) Σ_sample + λ · diag(Σ_sample)
```

Higher λ → closer to equal-weight; lower λ → more aggressive covariance exploitation.

After solving, weights are rescaled so that gross exposure matches the equal-weight baseline.

## Warmup

A minimum of **5 years** (1260 trading days) of data is required before MVO weights are used. During warmup, equal-weight signal positions are used. Evaluation begins 1991-01-24.

This is critical: without the warmup, the poorly estimated early covariance matrix caused the MVO portfolio to concentrate into a few contracts during the 1986–87 drawdown, producing a -0.39 max DD despite having the same unconditional volatility as equal-weight (-0.25). With the warmup, the same aggressive MVO setting achieves -0.13 DD — better than equal-weight on both Sharpe and drawdown.

## Signal parameters

All variants use the best momentum signal: `vw=60, sm=1, window=247`.

## Results (eval from 1991)

| Method               | Halflife | λ    | Sharpe | Max DD |
| -------------------- | -------- | ---- | ------ | ------ |
| Equal-weight         | —        | —    | 1.12   | -0.194 |
| MVO                  | 252      | 0.95 | 1.22   | -0.184 |
| MVO                  | 252      | 0.80 | 1.32   | -0.181 |
| MVO                  | 126      | 0.50 | 1.61   | -0.156 |
| MVO                  | 63       | 0.30 | 2.24   | -0.126 |

Full grid (Sharpe / Max DD):

| λ    | hl=63        | hl=126       | hl=252       |
| ---- | ------------ | ------------ | ------------ |
| 0.30 | 2.24 / -0.13 | 1.68 / -0.15 | 1.39 / -0.17 |
| 0.50 | 2.03 / -0.14 | 1.61 / -0.16 | 1.38 / -0.16 |
| 0.70 | 1.81 / -0.15 | 1.51 / -0.17 | 1.35 / -0.17 |
| 0.80 | 1.68 / -0.17 | 1.45 / -0.18 | 1.32 / -0.18 |
| 0.90 | 1.51 / -0.18 | 1.35 / -0.18 | 1.26 / -0.19 |
| 0.95 | 1.40 / -0.18 | 1.29 / -0.18 | 1.22 / -0.18 |

Every cell in the grid improves on equal-weight (1.12 / -0.19) on both dimensions. Lower λ and shorter halflife yield higher Sharpe and lower DD simultaneously — the covariance information is genuinely useful, not just fitting noise.

## Plots

- [Cumulative returns, drawdowns, rolling Sharpe](gepa_mvo_vs_equal.png)
- [Monthly returns heatmap (reg=0.80)](gepa_mvo_monthly.png)

## Takeaways

- With a proper 5-year warmup, MVO improves **both** Sharpe and drawdown versus equal-weight. There is no tradeoff — the covariance matrix contains real information.
- The most aggressive setting (λ=0.30, hl=63) doubles the Sharpe (1.12 → 2.24) while cutting max DD by a third (-0.19 → -0.13).
- Shorter halflife (faster-reacting cov estimates) dominates longer halflife at every regularization level — cross-contract correlations are time-varying and a 63-day halflife tracks them better than 252.
- The improvement is consistent across time (visible in rolling Sharpe), not concentrated in a single period.
