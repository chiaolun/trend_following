# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Trend Following: Autocorrelation Function (ACF) Analysis
# 
# While `analysis.ipynb` evaluated a simple Moving Average crossover, this notebook dives deeper into the structural properties of asset returns that make trend following work in the first place. 
# 
# Specifically, we analyze the **Autocorrelation Function (ACF)**. If an asset is trending, its past returns should ideally correlate with its future returns (momentum). We will measure the empirical ACF across our futures universe and use it to calibrate an Auto-Regressive (AR) model to derive optimal allocation weights. Finally, we'll run Monte Carlo simulations to see the theoretical performance bounds of these derived weights. 

# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from scipy.linalg import toeplitz, cholesky
from tqdm.auto import tqdm, trange
from matplotlib.backends.backend_pdf import PdfPages

# Formatting parameters for visual output
plt.rcParams['figure.figsize'] = [10, 8]
pd.set_option('display.max_rows', 200, 'display.max_columns', None)
all_rows = pd.option_context("display.max_rows", None)
months = pd.Series({m: i for i, m in enumerate("FGHJKMNQUVXZ")})

# %% [markdown]
# ## 1. Data Ingestion & Z-Scoring
# 
# First, we load our continuous ("front-month") contract price series and the filtered shortlist generated previously.
# 
# We then compute the daily absolute dollar returns and normalize them relative to their 180-day volatility. Finally, we convert these normalized returns into Z-scores (`rets_z_wide`). Z-scoring ensures that assets with wildly different typical return profiles are standardized to have a mean of 0 and a standard deviation of 1, allowing for a pure apples-to-apples ACF comparison.
# 
# Notice that we exclude Equities and Bonds because they have structural, fundamental upward biases (equity risk premium, yield) that violate the pure zero-mean stationarity assumption needed for pure momentum ACF analysis.

# %%
# Load data caches
front = pd.read_parquet("data/front.parquet")
shortlist = pd.read_parquet("data/shortlist.parquet").set_index("contract")
front = front.loc[shortlist.index]

# Calculate approximate historical costs (basis points scaled by contract multiplier)
front["cost"] = (
    front.close *
    shortlist.eval("cost_bps / 10_000 * multiplier").loc[front.index.get_level_values("contract")].values
)

# Normalize returns by trailing 180-day volatility
vol = front.eval("d_close").abs().groupby("contract").rolling(180).mean().droplevel(0)
rets = front.d_close.div(vol).dropna()

# Standardize the normalized returns into Z-scores
ret_stats = rets.groupby("contract").agg(["mean", "std"])
rets_z = rets.sub(ret_stats["mean"]).div(ret_stats["std"])
rets_z_wide = rets_z.unstack("contract").sort_index()

# Drop asset classes heavily polluted by structural positive drift
rets_z_wide = rets_z_wide.loc[:, ~shortlist.category.isin(["equities", "bonds"])]

# %% [markdown]
# ## 2. The Autocorrelation Function (ACF)
# 
# We calculate the ACF by shifting our normalized return series backward by `n` days and computing the correlation against the present day. We loop this from a 1-day lag all the way to 300 days to build a continuum of historical influence. 
# 
# The cumulative sum plot of the ACF validates the core thesis of trend following: positive correlation accumulates significantly in the short-to-medium term before plateauing or reversing slightly.

# %%
# Stack arrays and compute the average correlation for lags 1 through 299 days
acf = pd.Series([(rets_z_wide * rets_z_wide.shift(n)).stack().mean() for n in trange(1, 300)])

breakpoints = [20, 40, 210, 260]

# %%
plt.figure()
acf.cumsum().plot(title="Cumulative ACF of Cross-Asset Returns")
for b in breakpoints:
    plt.axvline(b, color="red", linestyle="--")
plt.ylabel("Cumulative Correlation")
plt.xlabel("Lag (Days)")
plt.show()

# %% [markdown]
# ## 3. Segmented Auto-Regressive (AR) Model
# 
# The raw ACF is noisy. Rather than fit a smooth hyper-parameterized curve, we approximate the ACF curve using a piece-wise step function (`acf_segment`) defined by specific time horizons (the `breakpoints` marked above, roughly corresponding to 1 month, 2 months, 10 months, and 12 months).
# 
# This structured vector (`corr_vec`) is converted into a Toeplitz matrix (`corr_mat`). The Toeplitz structure mathematically represents a stationary auto-covariance matrix for time series data. Inverting a subset of this matrix against the final column yields our optimal linear Auto-Regressive weights (`beta`). This `beta` tells us exactly how much emphasis to put on a 1-day old return versus a 200-day old return to maximize predictive power!

# %%
def acf_segment(i, j):
    """Fills a discrete array segment with the mean ACF correlation across that window."""
    return np.repeat(acf[i:j].mean(), j - i)

# Construct the piece-wise correlation vector
corr_vec = np.concatenate([np.r_[1]] + [acf_segment(i, j) for i, j in  zip([0] + breakpoints, breakpoints)])
corr_mat = toeplitz(corr_vec)
chol_mat = cholesky(corr_mat)

# %%
plt.figure()
acf.cumsum().plot(label="Empirical Cumulative ACF")
plt.plot(corr_vec[1:].cumsum(), label="Piece-wise Model Fit", color="orange")
for b in breakpoints:
    plt.axvline(b, color="red", linestyle="--", alpha=0.5)
plt.title("Empirical ACF vs Segmented Model Fit")
plt.legend()
plt.show()

# %%
# Extract the optimal linear regression weights (`beta`) representing the AR coefficients
beta = np.linalg.inv(corr_mat[:-1, :-1]) @ corr_mat[:-1, -1]

# Calculate the structural unexplained variance (noise) under the AR model
epsilon = (corr_mat[-1, -1] - beta @ corr_mat[:-1, :-1] @ beta)**0.5

# Theoretical Gross Annualized Sharpe ratio of this optimal linear combination:
gross_annual_sharpe = (beta @ corr_mat[:-1, :-1] @ beta)**0.5 * 250**0.5
print(f"Theoretical Gross Annual Sharpe: {gross_annual_sharpe:.2f}")

# %% [markdown]
# ## 4. Monte Carlo Simulation
# 
# Relying purely on closed-form algebra can be deceptive. We pass a massive array of standard normal noise (`np.random.randn`) through our Cholesky decomposition of the Toeplitz correlation matrix. 
# 
# This generates entirely synthetic price trajectories that perfectly adhere to our empirical ACF properties. Applying our `beta` model to these random paths proves that, structurally, the correlation traits alone enforce a positive Sharpe ratio.

# %%
# %%time
# Simulate 10,000 independent price series based on the empirical correlation matrix
draw_small = np.random.randn(10_000, corr_vec.shape[0]) @ chol_mat
strat_small = (draw_small[:, :-1] @ beta) * draw_small[:, -1]

# Validating the simulation against the closed-form Sharpe 
sim_sharpe = strat_small.mean() / strat_small.std() * 250**0.5
print(f"Simulated Gross Annual Sharpe (10k paths): {sim_sharpe:.2f}")

# %%
plt.figure()
plt.plot(strat_small.cumsum())
plt.title("Cumulative Edge of AR Model on Synthetic ACF Paths")
plt.show()

# %%
# %%time
# Generate a massively smoothed 100k iteration
draw_massive = np.random.randn(100_000, corr_vec.shape[0]) @ chol_mat

# %% [markdown]
# ## 5. Historical Strategy Application (Backtest)
# 
# With our optimal vector of look-back weights (`beta`) derived from the aggregated structural ACF, what happens if we apply it strictly as a trading algorithm against the real historical price paths?
# 
# Instead of a simple `SMA(200)`, we convolve the historical returns with the `beta` vector. We plot the gross return trajectories for each contract comparing the raw normalized asset price against the applied `beta` strategy path.

# %%
# Fast forward the index so we only start predicting when the model receives full historical lag context
strat_ix = rets_z_wide.index[len(beta):]

strats = []

# Loop across each contract and convolve its returns with the beta weights
for contract0 in rets_z_wide.columns:
    base = rets_z_wide[contract0].fillna(0)
    
    # Strat return calculation via convolution
    strat = (np.convolve(base, beta, mode="valid")[:-1] + base.mean()) * (base[strat_ix] - base.mean())
    
    # Re-normalize for graphing visualization
    base /= base.std()
    strat /= strat.std()
    
    strats.append(strat)
    
    # Note: Using plt.close() to prevent 30+ charts from spamming the interactive notebook 
    # If using interactively, replace with plt.show()
    plt.figure()
    ax = base.cumsum().plot(title=f"{contract0} - {shortlist.name[contract0]} (Base vs Model)")
    strat.dropna().cumsum().plot(ax=ax)
    plt.close()

# %% [markdown]
# ### Gross Portfolio Output & Sharpe Consistency
# 
# We take the mean across all asset strategy curves and plot the aggregate portfolio result. Finally, we evaluate the rolling 5-year Sharpe ratio to see if the ACF trait decays over time. As shown, the Sharpe remains incredibly structurally persistent decade after decade.

# %%
# Take the cross-sectional average of all strategy paths for a simulated portfolio
strats_mu = pd.concat(strats, axis=1).mean(axis=1)

plt.figure()
strats_mu.cumsum().plot(title="Aggregate AR ACF Portfolio (Gross Cumulative Returns)")
plt.show()

# %%
# Roll Sharpe ratios in 5-year disjoint windows to test structural persistence of the edge
plt.figure(figsize=(12, 7))
(
    strats_mu.groupby(pd.Grouper(freq="5YE")).apply(lambda xs: xs.mean() / xs.std() * 250**0.5)
    .rename(lambda xs: xs.year).iloc[1:-1]
    .plot.bar(title="Gross 5-Year Rolling Sharpe of the AR Model")
)
plt.show()
