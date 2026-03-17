# Strategy Parametrizations

## Common Parameters

All strategies share the following parameters:

| Parameter | Code | Description |
| --------- | ---- | ----------- |
| **Vol window** (`vw`) | `vol_window` | Rolling window (in trading days) for estimating each contract's volatility. Volatility is the rolling mean of absolute daily price changes: `vol_t = mean(\|ΔP\|, t-vw, t-1)`. Used to normalize returns: `norm_ret_t = ΔP_t / vol_t`. Typical values: 60, 90, 120, 180, 250. |
| **Smoothing** (`sm`) | `smoothing` | Rolling mean window applied to the raw signal before trading. When `sm=1`, no smoothing is applied (raw signal used directly). When `sm>1`, the signal is replaced by its `sm`-day rolling mean per contract. Higher smoothing reduces turnover but adds lag. |

### Volatility normalization

All strategies operate on **volatility-normalized cumulative returns**:

```
norm_ret_t = ΔP_t / vol_{t-1}           (vol estimated with vw-day lookback, lagged 1 day)
cumret_t   = Σ norm_ret_s  for s ≤ t     (per contract)
```

This ensures that all contracts contribute roughly equal risk regardless of their price level or volatility.

### Signal smoothing

For any raw signal `sig_raw_t ∈ {-1, +1}` (or `[-1, +1]` for breakout):

```
sig_t = rolling_mean(sig_raw, sm)        if sm > 1
sig_t = sig_raw_t                        if sm = 1
```

Smoothing converts the discrete ±1 signal into a continuous position in `[-1, +1]`, blending recent signal history.

### Position sizing and evaluation

Portfolio daily return:

```
port_ret_t = Σ_i  sig_{i,t-1} × norm_ret_{i,t}     (summed over contracts i)
```

Positions are lagged one day (trade on next open). Returns are scaled to a fixed leverage target for comparability.

---

## Signal Types

### 1. Momentum (`momentum`)

**Idea**: Go long if the normalized price is higher than it was `w` days ago; short otherwise.

| Parameter | Code | Range | Description |
| --------- | ---- | ----- | ----------- |
| **Window** (`w`) | `param1` | 20–500 | Lookback period for rate-of-change comparison |

**Signal construction**:

```
sig_raw_t = +1   if cumret_t > cumret_{t-w}
            -1   otherwise
```

This is a pure time-series momentum signal. It asks: "has this contract's risk-adjusted price gone up or down over the last `w` days?"

**Best regions found**: `vw=60, sm=1–35, w=210–250`

---

### 2. Breakout (`breakout`)

**Idea**: Measure where the current price sits within its recent Donchian channel (N-day high/low range), mapping to a continuous position.

| Parameter | Code | Range | Description |
| --------- | ---- | ----- | ----------- |
| **Window** (`w`) | `param1` | 20–500 | Lookback for the high/low channel |

**Signal construction**:

```
high_t = max(cumret_{t-w}, ..., cumret_t)
low_t  = min(cumret_{t-w}, ..., cumret_t)
range_t = high_t - low_t

sig_raw_t = clip( 2 × (cumret_t - low_t) / range_t - 1,  -1, +1 )
```

Unlike the other signals which produce discrete ±1, breakout produces a **continuous signal** in `[-1, +1]`:
- `+1` when price is at the channel high
- `-1` when at the channel low
- `0` when at the midpoint

This gradual positioning is one reason breakout achieves lower drawdowns.

**Best regions found**: `vw=60–120, sm=10–15, w=240–270`

---

### 3. Cumsum MA (`cumsum_ma`)

**Idea**: Go long if the current normalized cumulative return is above its own rolling moving average; short otherwise.

| Parameter | Code | Range | Description |
| --------- | ---- | ----- | ----------- |
| **Lookback** (`lookback`) | `param1` | 40–500 | Rolling MA window |

**Signal construction**:

```
ma_t = rolling_mean(cumret, lookback)

sig_raw_t = +1   if cumret_t > ma_t
            -1   otherwise
```

This is a classic trend-following signal equivalent to "price above its moving average."

**Best regions found**: `vw=60, sm=1–20, lookback=100–130`

---

### 4. EMA Cross (`ema_cross`)

**Idea**: Go long when a fast exponential moving average crosses above a slow one; short otherwise.

| Parameter | Code | Range | Constraint | Description |
| --------- | ---- | ----- | ---------- | ----------- |
| **Fast span** (`fast_span`) | `param1` | 5–75 | | EMA span for the fast average |
| **Slow span** (`slow_span`) | `param2` | 50–500 | `slow > 2 × fast` | EMA span for the slow average |

**Signal construction**:

```
ema_fast_t = EWM(cumret, span=fast_span)
ema_slow_t = EWM(cumret, span=slow_span)

sig_raw_t = +1   if ema_fast_t > ema_slow_t
            -1   otherwise
```

The EMA span parameter `s` corresponds to a decay factor `α = 2/(s+1)`. A span of 20 gives ~5% weight to observations 20 days ago.

**Best regions found**: `vw=60, sm=10–20, fast=10–20, slow=80–150`

---

### 5. Dual MA (`dual_ma`)

**Idea**: Same as EMA cross but using simple (equal-weight) moving averages instead of exponential.

| Parameter | Code | Range | Constraint | Description |
| --------- | ---- | ----- | ---------- | ----------- |
| **Fast window** (`fast_window`) | `param1` | 10–100 | | SMA window for the fast average |
| **Slow window** (`slow_window`) | `param2` | 100–500 | `slow > 2 × fast` | SMA window for the slow average |

**Signal construction**:

```
sma_fast_t = rolling_mean(cumret, fast_window)
sma_slow_t = rolling_mean(cumret, slow_window)

sig_raw_t = +1   if sma_fast_t > sma_slow_t
            -1   otherwise
```

**Best regions found**: underperforms other signal types (Sharpe < 0.71)

---

## Parameter Interactions

The three shared parameters (`vw`, `sm`, `w`/signal params) interact:

- **`vw` controls the volatility regime**. Lower `vw` (60) reacts faster to vol changes, producing more aggressive position sizing. Higher `vw` (120–250) smooths vol estimates, reducing position churn but potentially mis-sizing during regime shifts.

- **`sm` controls signal lag vs. noise**. At `sm=1` (no smoothing), the signal reacts immediately but whipsaws on noisy days. Higher `sm` (20–40) averages out noise but delays entry/exit. For breakout, smoothing matters less because the signal is already continuous.

- **Signal window (`w`/`lookback`/`fast`/`slow`) controls the trend timescale**. Shorter windows capture faster trends but generate more false signals. Longer windows are more selective but enter later and miss short trends.

The optimal balance depends on the signal type. Momentum works best with `sm=1–35` and `w=210–250` (roughly 1-year lookback). Breakout prefers `w=240–270` with any `sm`. EMA cross needs `fast=10–20, slow=80–150` — a ratio of roughly 5–10x between the two spans.
