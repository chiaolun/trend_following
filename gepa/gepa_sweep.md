# GEPA Sweep Results Log

Started: 2026-03-17

## Top Results by Sharpe


| Config                             | Sharpe    | Max DD | Notes                                          |
| ---------------------------------- | --------- | ------ | ---------------------------------------------- |
| **momentum vw=60 sm=1 window=247** | **1.051** | -0.250 | **BEST SHARPE** (sharp peak, possible overfit) |
| momentum vw=60 sm=2 window=247     | 1.031     | -0.251 |                                                |
| momentum vw=60 sm=3 window=246     | 1.031     | -0.252 |                                                |
| momentum vw=60 sm=30 window=224    | 1.010     | -0.238 | Robust peak                                    |
| momentum vw=60 sm=32 window=215    | 1.005     | -0.223 |                                                |
| momentum vw=60 sm=25 window=215    | 1.000     | -0.205 | Best balanced (2020s: 1.021)                   |


## Top Results by DD


| Config                                 | Sharpe | Max DD     | Notes       |
| -------------------------------------- | ------ | ---------- | ----------- |
| **breakout vw=90 sm=12 window=255**    | 0.948  | **-0.150** | **BEST DD** |
| breakout vw=90 sm=12 window=252        | 0.951  | -0.150     |             |
| breakout vw=120 sm=12 window=260       | 0.934  | -0.151     |             |
| breakout vw=90 sm=12 window=265        | 0.951  | -0.152     |             |
| ema_cross vw=60 sm=12 fast=10 slow=140 | 0.864  | -0.152     |             |
| ema_cross vw=60 sm=10 fast=10 slow=160 | 0.840  | -0.152     |             |


## Best Risk-Adjusted (Sharpe / abs DD)

| Config                           | Sharpe | Max DD | Ratio |
| -------------------------------- | ------ | ------ | ----- |
| breakout vw=90 sm=12 window=252  | 0.951  | -0.150 | 6.34  |
| breakout vw=60 sm=12 window=240  | 0.972  | -0.155 | 6.27  |
| breakout vw=90 sm=12 window=265  | 0.951  | -0.152 | 6.26  |
| breakout vw=60 sm=12 window=260  | 0.973  | -0.157 | 6.20  |
| breakout vw=120 sm=12 window=260 | 0.934  | -0.151 | 6.19  |

## Plots

- [Pareto composite: cumulative, drawdown, rolling Sharpe](gepa_composite.png)
- [Composite monthly returns heatmap](gepa_monthly_composite.png) — **Sharpe 1.029, DD -0.165**
- [Individual top configs: cumulative, drawdown, rolling Sharpe](gepa_top_results.png)
- Monthly heatmaps:
  - [momentum sm=1 w=247 (S=1.05)](gepa_monthly_mom_sm1_w247_S105.png)
  - [momentum sm=32 w=215 (S=1.01)](gepa_monthly_mom_sm32_w215_S101.png)
  - [momentum sm=14 w=214 (S=0.99)](gepa_monthly_mom_sm14_w214_S099.png)
  - [breakout vw=120 w=260 (DD=-.15)](gepa_monthly_brk_vw120_w260_DD-15.png)
  - [breakout vw=60 w=260 (S=0.97)](gepa_monthly_brk_vw60_w260_S097.png)
  - [ema_cross f=10 s=140 (DD=-.15)](gepa_monthly_ema_f10_s140_DD-15.png)

## Stats (~87 iterations, ~270+ configs evaluated)

- Rows in singles.parquet: 4858

## Key Findings

### Momentum (best for raw Sharpe)

- **Peak**: vw=60, sm=1, window=247 → Sharpe 1.051 (sharp spike, possibly overfit)
- **Robust peak**: vw=60, sm=28-32, window=222-226 → Sharpe 1.01, DD -0.24
- **Balanced**: vw=60, sm=25, window=215 → Sharpe 1.00, DD -0.21, 2020s Sharpe 1.02
- **Low-DD momentum**: vw=60, sm=14, window=214 → Sharpe 0.99, DD -0.20

### Breakout (best for low drawdown & risk-adjusted)

- **DD floor**: ~-0.150 at vw=90, sm=11-13, window=250-260
- **Sharpe-optimized**: vw=60, sm=12, window=240-260 → Sharpe 0.97, DD -0.16
- Remarkably robust: sm=1-20 all work; window 230-300 all strong
- **Best overall risk-adjusted signal type** (Sharpe/|DD| > 6.2)

### EMA Cross (surprisingly competitive on DD)

- Can achieve DD -0.152 with sm=12 fast=10 slow=140 (Sharpe 0.86)
- Best Sharpe: vw=60, sm=15-18, fast=20, slow=100-110 → Sharpe 0.88

### Cumsum MA: caps at ~0.83 Sharpe

### Dual MA: consistently weakest (<0.71)

### Cross-cutting

- vw=60 dominates for Sharpe; vw=90-120 for DD
- sm=12 is a robust default across signal types
- DD floor around -0.15 is structural (single worst drawdown event)

