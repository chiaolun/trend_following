# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Trend Following: A Century of Evidence?
#
# In this notebook, we backtest a classic **Trend Following** strategy. Trend following is a systematic investment strategy that simply buys assets that have been going up and sells short assets that have been going down.

# %%
import re
import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tqdm.auto import tqdm
from matplotlib.backends.backend_pdf import PdfPages

pd.set_option("display.max_rows", 200, "display.max_columns", None)
all_rows = pd.option_context("display.max_rows", None)
months = pd.Series({m: i for i, m in enumerate("FGHJKMNQUVXZ")})

# %% [markdown]
# ## 1. Transaction Costs from AQR
# ![image.png](attachment:image.png)
#
# Every time we buy or sell, we pay transaction costs (commissions, bid-ask spread slippage). Data for historical transaction costs was extracted from AQR's [trend following whitepaper](https://www.trendfollowing.com/whitepaper/Century_Evidence_Trend_Following.pdf).
#
# Notice that costs are defined by **Asset Class** and **Time Period**. Trading equities in 1880 was vastly more expensive than trading them in 2003 because of modern electronic markets.

# %%
transaction_costs = pd.read_csv("data/transaction_costs.csv")
transaction_costs = transaction_costs.rename(
    columns={"start_year": "time_period"}
).drop(columns=["end_year"])
transaction_costs

# %% [markdown]
# ## 2. Data Ingestion - Specifications
#
# We trade **Futures Contracts**. A futures contract is an agreement to buy or sell an asset at a predetermined price at a specified time in the future. We use specs from Commodity Systems Incorporated (CSI).
#
# Because a single contract often controls a large amount of the underlying asset (e.g., 1000 barrels of crude oil), we need a **multiplier**. The multiplier turns a 1-point move in the quoted futures price into real dollars.

# %%
with duckdb.connect("data/csi_data.duckdb", read_only=True) as con:
    spec = con.execute("SELECT * FROM spec").fetchdf()
    dailies_symbols = set(
        con.execute("SELECT DISTINCT symbol FROM dailies").fetchdf().symbol
    )


def compute_multiplier(row):
    cs = row["contract_size"]
    units = row["units"]
    num = float(re.search(r"[\d]+(?:\.\d+)?", cs.replace(",", "")).group())
    if "X INDEX" in cs.upper():
        return num
    elif units in ("POINTS", "PERCENT"):
        return num / 100
    elif units.startswith("CENTS"):
        return num * 0.01
    elif "JPY 100" in units:
        return num / 100
    else:
        return num


category_map = {
    "ENERGY": "hard",
    "METALS": "hard",
    "INDEXES-U.S.": "equities",
    "INDEXES-EUROPEAN": "equities",
    "INDEXES-ASIAN": "equities",
    "FOREX": "forex",
    "GOVT NOTES": "bonds",
    "GOVT BONDS": "bonds",
    "FOOD & FIBER": "soft",
    "GRAINS & OILSEEDS": "soft",
    "LIVESTOCK & MEATS": "soft",
}

costs = (
    transaction_costs.sort_values("time_period")
    .groupby("asset_class")
    .transaction_cost.last()
)

# Structurally clean pipeline: spec remains immutable!
metadata = (
    spec.assign(
        contract=lambda df: df.symbol.apply(
            lambda s: s[:-1] if s.endswith("2") and s[:-1] in dailies_symbols else s
        ),
        multiplier=lambda df: df.apply(compute_multiplier, axis=1),
    )
    .set_index("contract")
    .assign(
        category=lambda df: df["category"].map(category_map),
        exch_symbol=lambda df: df.index,
        cost_bps=lambda df: (
            df["category"]
            .map(
                {
                    "bonds": "Bonds",
                    "soft": "Commodities",
                    "hard": "Commodities",
                    "forex": "Currencies",
                    "equities": "Equities",
                }
            )
            .map(costs)
        ),
    )
    .rename(
        columns={
            "market": "Name",
            "category": "Futures Category",
            "exchange": "Exchange",
            "currency": "Currency",
            "exch_symbol": "Exch Symbol",
        }
    )
    .drop(
        columns=["point_value", "symbol", "units", "contract_size", "origin_filename"]
    )
)

metadata

# %% [markdown]
# ## 3. Daily Prices & Futures Expiration
#
# Unlike stocks, futures contracts **expire**. For example, there's a March Crude Oil contract and a June Crude Oil contract.
#
# Expiring contracts are specified by an **expiry string** (e.g. `2024H` indicates March 2024, using standard financial month letters). The `mod_month` calculates how far away the expiry is from the current trading date, allowing us to eventually select the most "active" contract to trade.

# %%
month_to_letter = dict(enumerate("FGHJKMNQUVXZ", 1))

with duckdb.connect("data/csi_data.duckdb", read_only=True) as con:
    dailies = (
        con.execute("SELECT * FROM dailies WHERE origin_filename NOT LIKE '%Spot%'")
        .fetchdf()
        .rename(columns={"symbol": "contract", "oi": "open_interest"})
        .assign(
            expiry=lambda df: (
                (df.numeric_delivery_month // 100).astype(str)
                + (df.numeric_delivery_month % 100).map(month_to_letter)
            ),
            date=lambda df: pd.to_datetime(df.date),
        )
        .set_index(["contract", "expiry", "date"])
    )

dailies.sort_index(inplace=True)
prev_close = dailies.groupby(["contract", "expiry"]).close.shift()

# d_close is the absolute daily dollar return
dailies["d_close"] = dailies.close.sub(prev_close).mul(metadata.multiplier)

# Calculate months until expiration
dailies["expiry_int"] = dailies.eval("expiry").str.slice(0, 4).astype(
    int
) * 12 + dailies.eval("expiry").str.slice(4, 5).map(months)
dailies["mod_month"] = dailies.eval(
    "expiry_int - (date.dt.year + (expiry_int.mod(12) < date.dt.month.sub(1))) * 12"
)
dailies[["expiry_int", "mod_month"]] = dailies[["expiry_int", "mod_month"]].astype(
    "Int64"
)
dailies = dailies.query("mod_month.between(0, 11)").copy()
dailies["mod_month"] = pd.Categorical.from_codes(dailies["mod_month"], months.index)
dailies = dailies.loc[
    lambda df: df.index.get_level_values("contract").isin(metadata.index)
]

dailies

# %% [markdown]
# ## 4. Liquidity Filtering (Shortlisting)
#
# We only want to trade highly liquid markets. Thinly traded markets have wide bid-ask spreads, which eat into profits. We rank contracts by `dollar_vol` (average daily dollar volume) and shortlist the top 50 highly traded contracts.

# %%
contract_volumes = (
    dailies.assign(decade=lambda df: df.eval("date.dt.year.floordiv(10).mul(10)"))
    .groupby(["contract", "decade"])
    .agg({"volume": "mean", "d_close": "std"})
    .rename(columns={"d_close": "dollar_vol_per"})
    .reset_index()
)

contract_volumes["multiplier"] = contract_volumes.contract.map(metadata.multiplier)
contract_volumes["currency"] = contract_volumes.contract.map(metadata.Currency)
contract_volumes["category"] = contract_volumes.contract.map(
    metadata["Futures Category"]
)
contract_volumes["cost_bps"] = contract_volumes.contract.map(metadata["cost_bps"])
contract_volumes["exchange"] = contract_volumes.contract.map(metadata.Exchange)
contract_volumes["exch_symbol"] = contract_volumes.contract.map(metadata["Exch Symbol"])
contract_volumes["name"] = contract_volumes.contract.map(metadata["Name"])
contract_volumes["dollar_vol"] = contract_volumes.eval("dollar_vol_per * volume")
contract_volumes["start_year"] = contract_volumes.contract.map(
    dailies.eval("date").groupby("contract").min().dt.year.astype("Int64")
)

shortlist = (
    contract_volumes.query("start_year < 2000 & decade == 2010")
    .assign(
        cat_rank=lambda df: (
            df.groupby(["category", "decade"])
            .dollar_vol.rank(ascending=False)
            .fillna(0)
            .astype(int)
        )
    )
    .sort_values(["cat_rank", "dollar_vol"], ascending=[True, False])
    .head(50)[
        [
            "dollar_vol",
            "dollar_vol_per",
            "volume",
            "currency",
            "contract",
            "exchange",
            "exch_symbol",
            "category",
            "cat_rank",
            "name",
            "start_year",
            "multiplier",
            "cost_bps",
        ]
    ]
    .reset_index(drop=True)
)
with all_rows:
    display(
        shortlist.style.format(
            subset=["dollar_vol", "dollar_vol_per", "volume"], formatter="{:.1e}"
        ).set_properties(subset=["name"], **{"white-space": "pre-wrap"})
    )

# %% [markdown]
# ### Inspecting Volume Transitions
# To ensure our rollover logic works, we can generate PDF reports visualizing the trading volume of different expiration months over time for each contract. This helps us verify that trading activity indeed transitions smoothly from one active contract to the next as expiration approaches.

# %%
# Identify the maximum volume for any expiration month on a given day
top_ratio = (
    dailies.query("date > '2008'")
    .sort_values("volume")
    .groupby(["contract", "date"])
    .last()
    .groupby(["contract", "mod_month"])
    .volume.count()
)
top_ratio /= top_ratio.groupby("contract").transform("max")

# %%
# Export bar charts showing the ratio of volume for each expiration month
with PdfPages("reports/expiry_volumes.pdf") as pdf:
    for contract_rank in tqdm(shortlist.index):
        title = (
            "{contract_rank} - {contract} / {exch_symbol}({exchange}) / {name}".format(
                contract_rank=contract_rank, **shortlist.loc[contract_rank].to_dict()
            )
        )

        plt.figure(dpi=100)

        ax = top_ratio.loc[shortlist.contract[contract_rank]].plot.bar(
            figsize=(8, 5), title=title
        )

        ax.axhline(y=1 / 12, color="red")

        pdf.savefig()
        plt.close("all")

# %%
# Calculate the percentage share of open interest for each contract month over time
volume_shares = (
    dailies.set_index("mod_month", append=True)
    .reorder_levels(["contract", "date", "mod_month", "expiry"])
    .sort_index()
    .groupby(["contract", "date", "mod_month"], observed=True)
    .open_interest.first()
    .pipe(lambda xs: xs / xs.groupby(["contract", "date"]).transform("sum"))
    .sort_index()
)

# %%
# Export line charts showing how the volume share of the 'front' month rolls over
with PdfPages("reports/volume_share.pdf") as pdf:
    for contract_rank in tqdm(shortlist.index):
        title = (
            "{contract_rank} - {contract} / {exch_symbol}({exchange}) / {name}".format(
                contract_rank=contract_rank, **shortlist.loc[contract_rank].to_dict()
            )
        )

        plt.figure(dpi=100)

        (
            volume_shares.loc[shortlist.contract[contract_rank]]
            .unstack("mod_month")
            .loc["2020":"2021"]
            .plot(figsize=(8, 5), title=title)
        )

        pdf.savefig()
        plt.close("all")

# %% [markdown]
# ## 5. Continuous Contracts (The "Front" Month)
#
# Since futures expire, we cannot use a single contract to simulate holding an asset for 10 years. Instead, we **roll** our position: we hold the "front" (most active) contract, and before it expires and ceases trading, we sell it and buy the next active expiration cycle.
#
# The `top_ratio` and `volume_fraction` logic dynamically determines which contract month possesses the majority of the trading volume so we can stitch together a single `front` continuous time series of daily returns.

# %%
top_ratio = (
    dailies.query("date > '2008'")
    .sort_values("volume")
    .groupby(["contract", "date"])
    .last()
    .groupby(["contract", "mod_month"])
    .volume.count()
)
top_ratio /= top_ratio.groupby("contract").transform("max")

active_months = top_ratio.index[top_ratio.gt(1 / 12)]
dailies = (
    dailies.reset_index("expiry")
    .set_index("mod_month", append=True)
    .reorder_levels(["contract", "mod_month", "date"])
    .loc[lambda df: df.index.droplevel("date").isin(active_months)]
    .reset_index("mod_month")
    .set_index("expiry", append=True)
)

volumes = dailies.volume.loc[lambda xs: xs > 0]
volumes /= volumes.groupby(["contract", "date"]).transform("sum")
volumes = volumes.sort_index().rename("volume_fraction")
volumes = volumes.groupby(["contract", "expiry"]).shift()

# This produces the stitched continuous series: "front"
front = (
    volumes.loc[lambda df: df.groupby(["contract", "expiry"]).cummax().gt(1 / 12)]
    .reset_index()
    .sort_values("expiry")
    .groupby(["contract", "date"])
    .first()
    .set_index("expiry", append=True)
    .join(dailies)
    .reset_index("expiry")
)

# %% [markdown]
# ## 6. Volatility Normalization
#
# A 1% change in crude oil means a totally different amount of risk than a 1% change in 10-Year Treasury Notes. If we traded the exact same number of contracts, our portfolio would be heavily skewed towards the most volatile assets (like energy).
#
# We calculate a 180-day moving average of absolute dollar returns (`vol`) and use it to scale our positions. This means for volatile assets we trade fewer contracts, and for stable assets we trade more contracts.

# %%
front["vol"] = (
    front.d_close.abs()
    .groupby("contract")
    .rolling(180)
    .mean()
    .reset_index(level=0, drop=True)
    .groupby("contract")
    .shift()
)

front.to_parquet("data/front.parquet")
shortlist.to_parquet("data/shortlist.parquet")

# %% [markdown]
# ### Volatility Normalized Prices & Correlations
# Before we generate trading signals, it's helpful to visualize the volatility-normalized price paths of our shortlisted contracts. We also generate a correlation matrix to understand how these assets move in relation to one another. High correlations between assets mean redundant risk in the portfolio, which a trend follower must monitor and control.

# %%
# Export plots of the volatility-normalized price history for each contract
with PdfPages("reports/prices.pdf") as pdf:
    for contract in tqdm(shortlist.itertuples(), total=len(shortlist)):
        title = (
            f"{contract.Index} - {contract.contract} / "
            f"{contract.exch_symbol}({contract.exchange}) / {contract.name}"
        )
        plt.figure(dpi=100)

        (
            front.loc[contract.contract]
            .eval("d_close / vol")
            .cumsum()
            .plot(figsize=(8, 5), title=title)
        )

        pdf.savefig()
        plt.close()

# %%
# Compute a correlation matrix of the daily returns since 2010
cmat = (
    front.loc[shortlist.contract]
    .eval("d_close / vol")
    .unstack("contract")
    .sort_index()["2010":]
    .corr()
)

# Export horizontal bar charts showing top correlations for each contract
with PdfPages("reports/correlations.pdf") as pdf:
    for contract in tqdm(shortlist.contract):
        plt.figure(figsize=(8, 5), dpi=100)
        ax = (
            cmat[contract]
            .rename(metadata.index + " - " + metadata.Name)
            .iloc[lambda xs: np.argsort(xs.abs().values)[::-1]]
            .dropna()
            .head(10)[::-1]
            .plot.barh(title="Correlations", xlim=(-1, 1))
        )
        ax.axvline(0.5, color="pink")
        ax.axvline(-0.5, color="pink")
        ax.yaxis.set_label_coords(-1.9, 0.5)
        plt.subplots_adjust(left=0.6)
        pdf.savefig()
        plt.close("all")

# %% [markdown]
# ## 7. Signal Generation (The Trend Algorithm)
#
# The core algorithm is simple. Consider the asset's current price. If it is higher than its 200-day moving average, we are in an uptrend (+1 signal: BUY/LONG). If it's lower, we are in a downtrend (-1 signal: SELL/SHORT).
#
# A 30-day smoothing moving average reduces "whipsaw" (flipping wildly between the +1 and -1 boundary).

# %%
signals = []
for ndays in [200]:
    signals.append(
        front.eval("d_close / vol")
        .groupby("contract")
        .cumsum()
        .pipe(
            lambda xs: (
                xs.groupby("contract")
                .rolling(ndays)
                .mean()
                .reset_index(level=0, drop=True)
                .lt(xs)
            )
        )
        .groupby("contract")
        .shift(fill_value=False)
        .pipe(lambda xs: xs * 2 - 1)
        .groupby("contract")
        .rolling(30)
        .mean()
        .reset_index(level=0, drop=True)
    )
signal = pd.concat(signals, axis=1).mean(axis=1)

# %% [markdown]
# ## 8. Portfolio Construction & Performance
#
# The theoretical return of holding this position is simply the normalized daily return (`d_close / vol`) multiplied by the signal generated on the previous day.
#
# We then sum across all contracts to find the portfolio's simulated `pnl`. We normalize risk with a target portfolio leverage. We will then plot the long-term trend performance vs a simple Buy-&-Hold of those exact same futures.

# %%
start_year = "1980"

rets = front.eval("d_close / vol * @signal")
pnl = rets.loc[shortlist.contract].groupby("date").sum()[start_year:]
bnh_rets = front.eval("d_close / vol")
leverage = 0.006
risk = leverage / pnl.std()


def plot_dd(c_pl, title):
    xlim = (c_pl.index[0], c_pl.index[-1])
    c_pl.plot(lw=1, color="black", title=title, xlim=xlim)
    plt.ylabel("Cumulative vol-normalized points")
    plt.fill_between(c_pl.index, c_pl, c_pl.cummax(), facecolor="red", alpha=0.5)


# Let's see our grand trend following results!
plt.figure(figsize=(10, 8), dpi=100)
plot_dd(pnl.mul(risk).cumsum(), "Trend Following")

# Compare against Buy-And-Hold
c_bnh = (
    bnh_rets.loc[shortlist.contract]
    .groupby("date")
    .sum()[start_year:]
    .mul(risk)
    .cumsum()
)
plt.figure(figsize=(10, 8), dpi=100)
plot_dd(c_bnh, title="Buy and Hold")

# %% [markdown]
# ### Deep Dive: Performance Reports
# Let's generate detailed PDF tear sheets showing the cumulative returns of the trend following strategy vs. a buy-and-hold baseline for each individual contract, and aggregated across asset categories.

# %%
# Export plots illustrating trend vs. buy-and-hold equity curves
with PdfPages("reports/returns.pdf") as pdf:
    c_pl = pnl.mul(risk).cumsum()
    plt.figure(figsize=(8, 5), dpi=100)
    plot_dd(c_pl, title="Trend")

    ax = (
        front.loc[shortlist.contract]
        .groupby("date")
        .expiry.count()
        .cummax()
        .plot(secondary_y=True)
    )
    ax.set_ylabel("Contract Count", rotation=-90, labelpad=20)
    ax.set_ylim(0, None)

    pdf.savefig()
    plt.close("all")

    c_bnh_pdf = (
        bnh_rets.loc[shortlist.contract]
        .groupby("date")
        .sum()[start_year:]
        .mul(risk)
        .cumsum()
    )
    plt.figure(figsize=(8, 5), dpi=100)
    plot_dd(c_bnh_pdf, title="Buy and Hold")

    pdf.savefig()
    plt.close("all")

    for contract in tqdm(shortlist.itertuples(), total=len(shortlist)):
        title = (
            f"{contract.Index} - {contract.contract} / "
            f"{contract.exch_symbol}({contract.exchange}) / {contract.name}"
        )

        all_rets = pd.DataFrame(
            {
                "trend": rets.loc[contract.contract],
                "buy and hold": bnh_rets.loc[contract.contract],
            }
        )

        plt.figure(dpi=100)

        ax = signal.loc[contract.contract].plot(
            secondary_y=True, color="grey", alpha=0.3
        )
        ax.set_ylabel("Signal", rotation=-90)

        ax = (
            all_rets.loc[start_year:]
            .cumsum()
            .plot(
                figsize=(8, 5),
                title=title,
                xlim=(pnl.index[0], pnl.index[-1]),
                ax=ax,
            )
        )
        ax.left_ax.set_ylabel("Cumulative vol-normalized points")
        ax.legend(*ax.left_ax.get_legend_handles_labels(), loc="upper left")

        pdf.savefig()
        plt.close("all")

# %%
# Aggregate theoretical returns by standard Futures categories (e.g., Equities, Metals)
all_rets_by_category = (
    pd.DataFrame(
        {
            "trend": rets,
            "buy and hold": bnh_rets,
        }
    )
    .loc[shortlist.contract]
    .join(metadata[["Futures Category"]])
    .rename(columns={"Futures Category": "category"})
    .groupby(["category", "date"])
    .sum()
)

# %%
with PdfPages("reports/returns_by_category.pdf") as pdf:
    for category in tqdm(
        all_rets_by_category.index.get_level_values("category").drop_duplicates()
    ):
        plt.figure(figsize=(8, 5), dpi=100)
        plot_dd(
            all_rets_by_category.loc[category]["trend"].loc[start_year:].cumsum(),
            title=f"{category} trend",
        )
        pdf.savefig()
        plt.close("all")

        plt.figure(figsize=(8, 5), dpi=100)
        plot_dd(
            all_rets_by_category.loc[category]["buy and hold"]
            .loc[start_year:]
            .cumsum(),
            title=f"{category} buy and hold",
        )
        pdf.savefig()
        plt.close("all")

# %%
# Analyze returns broken down by rolling 5-year periods
decade = (pnl.index.get_level_values("date").year // 5) * 5
by_period = {}
legend = []
for _, pnl0 in pnl.groupby(decade):
    yr0 = pnl0.index.year[0]
    pnl0.index = ((pnl0.index - pd.Timestamp(yr0, 1, 1)) / pd.Timedelta(days=1)).astype(
        int
    )
    by_period[yr0] = pnl0
by_period = pd.DataFrame(by_period)

# %%
plt.figure(figsize=(12, 7), dpi=100)
by_period.cumsum().ffill().plot(ax=plt.gca())
by_period.apply(lambda xs: xs.mean() / xs.std() * 250**0.5)

# %% [markdown]
# ### Benchmark Comparison: Trend Following vs. S&P 500
# To contextualize the performance, we compare our Trend Following portfolio against a simple buy-and-hold of the S&P 500 (using the 'ES' futures contract). We evaluate rolling correlations, combined portfolio Sharpe ratios, and drawdowns.

# %%
# Generate a simple moving average crossover signal for the S&P 500
ES_signal = (
    front.loc["ES"]
    .eval("d_close / vol")
    .cumsum()
    .pipe(lambda xs: xs.rolling(200).mean().lt(xs))
    .shift(fill_value=False)
    .rolling(90)
    .mean()
)

# Calculate targeted S&P 500 returns based on leverage
ES_position = ES_signal.div(front.vol["ES"])
ES_ret = ES_position.mul(front.d_close["ES"])
ES_risk = leverage / ES_ret.std()

# %%
# Calculate the rolling 5-Year correlation between our Trend Following Portfolio and the S&P 500.
# A low or negative correlation demonstrates that trend following acts as a strong diversifier to traditional equities.
(
    pd.concat(
        [
            ES_ret.rename("ES"),
            pnl.rename("Portfolio"),
        ],
        axis=1,
        sort=False,
    )
    .dropna()
    .groupby(pd.Grouper(freq="5YE"))
    .corr()
    .unstack()
    .iloc[:, 1]
    .rename(lambda xs: xs.year)
)

# %%
# Visualize the combined Gross Sharpe Ratio over rolling 5-year periods.
# This shows how a combined 50/50 portfolio (SP500 + Trend) performs risk-wise compared to each standalone asset.
plt.figure(figsize=(12, 7), dpi=100)
(
    (ES_ret.mul(ES_risk) + pnl.mul(risk))
    .groupby(pd.Grouper(freq="5YE"))
    .apply(lambda xs: xs.mean() / xs.std() * 250**0.5)
    .rename(lambda xs: xs.year)
    .plot.bar(title="Gross Sharpe (S&P 500 + Trend Following Combined)")
)

# %% [markdown]
# ### Comparison to Industry Benchmarks
# We fetch the definitive SG Trend Index (formerly IASG) from the web to see how our home-brewed, naive trend following algorithm correlates with the multi-billion dollar quantitative hedge fund industry baseline.

# %%
plt.figure(figsize=(10, 8), dpi=100)
plot_dd(ES_ret.mul(ES_risk).cumsum(), "SP500 Baseline")

# %%
plt.figure(figsize=(10, 8), dpi=100)
plot_dd(
    (ES_ret.mul(ES_risk) + pnl.mul(risk)).cumsum(), "Combined SP500 and Trend Following"
)

# %%
# Download and parse index from https://www.iasg.com/en/indexes/trend-following-index/historical-data
iasg_index = (
    pd.read_html("data/tf_index.html")[0]
    .set_index("Year")
    .rename_axis(columns="Month")
    .drop(["YTD", "DD"], axis=1)
    .stack("Month")
    .rename("return")
    .reset_index()
    .assign(
        ts=lambda df: pd.to_datetime(
            df["Year"].astype(str) + df["Month"], format="%Y%b"
        )
    )
    .set_index("ts")["return"]
    .sort_index()
    .transform(lambda xs: xs / xs.abs().rolling(12).mean())
)

# Align the benchmark timeframe with our portfolio logic
comparison = pd.concat(
    [
        iasg_index.dropna().groupby(pd.Grouper(freq="ME")).sum()[1:].rename("index"),
        pnl.groupby(pd.Grouper(freq="ME")).sum()[1:].rename("ours"),
    ],
    axis=1,
    sort=False,
).dropna()

# Compute decadal correlation between our backtest and the industry benchmark
(comparison.groupby(lambda xs: (xs.year // 10) * 10).corr().unstack().iloc[:, 1])

# %% [markdown]
# ### Calendar Year Returns & Drawdowns
# This suite of bar charts visualizes the annual returns for the S&P 500, the Trend Following strategy, and a combined 50/50 portfolio. Finally, it plots the maximum drawdown experienced each year.

# %%
plt.figure(figsize=(12, 9), dpi=100)
(
    ES_ret.mul(ES_risk)
    .add(1)
    .groupby(pd.Grouper(freq="YE"))
    .prod()
    .sub(1)
    .rename(lambda xs: xs.year)
    .plot.bar(title="ES returns")
)

# %%
plt.figure(figsize=(12, 9), dpi=100)
(
    pnl.mul(risk)
    .add(1)
    .groupby(pd.Grouper(freq="YE"))
    .prod()
    .sub(1)
    .rename(lambda xs: xs.year)
    .plot.bar(title="TF returns")
)

# %%
plt.figure(figsize=(12, 9), dpi=100)
(
    ES_ret.mul(ES_risk)
    .add(pnl.mul(risk), fill_value=0)
    .add(1)
    .groupby(pd.Grouper(freq="YE"))
    .prod()
    .sub(1)
    .rename(lambda xs: xs.year)
    .plot.bar(title="Total returns")
)

# %%
plt.figure(figsize=(12, 9), dpi=100)
(
    ES_ret.mul(ES_risk)
    .add(pnl.mul(risk), fill_value=0)
    .add(1)
    .cumprod()
    .pipe(lambda xs: xs.div(xs.cummax()) - 1)
    .groupby(pd.Grouper(freq="YE"))
    .min()
    .rename(lambda xs: xs.year)
    .plot.bar(title="Drawdowns")
)

# %% [markdown]
# ## 9. Real-World Friction & Transaction Costs
#
# In reality, you cannot trade "0.45 contracts" of Crude Oil. You have to round your targets to nearest integer contracts at your broker.
#
# Say you have a \$1,000,000 account value (`balance`). What happens to performance if we round strictly to integer chunks? We estimate the number of real contracts, compute exactly how many times we trade per dollar, and view the gross profit margin. If margins are too low, the strategy will fail to overcome commissions!

# %%
# Calculate "trades" (changes in position size or completely rolling into a new expiry month)
contracts_per_dollar = signal.div(front.vol).loc[shortlist.contract].mul(risk).dropna()
trades_per_dollar = (
    contracts_per_dollar.groupby(["contract", front.expiry])
    .diff()
    .fillna(contracts_per_dollar)
)
gross_per_dollar = contracts_per_dollar.mul(front.d_close.loc[shortlist.contract])

# Bar chart assessing how much gross profit is made per "trade" historically. The margin must exceed slippage constraints.
plt.figure(figsize=(12, 7), dpi=100)
ax = (
    pd.concat(
        [
            contracts_per_dollar.rename("contracts"),
            trades_per_dollar.abs().rename("trades"),
            gross_per_dollar.rename("gross"),
        ],
        axis=1,
    )
    .query("date.dt.year >= 1980")
    .groupby(lambda xs: xs[1].year // 5 * 5)
    .sum()
    .pipe(lambda df: df["gross"] / df["trades"] * 2)
    .plot.bar(title="Gross profit per roundtrip")
)

# Apply a simulated initial balance to project fractional allocations onto hard integer contract boundaries.
balance = 1e6
actual_positions = contracts_per_dollar.mul(balance).round()

# %% [markdown]
# ### Contract Rollover Frequencies
# When holding continuous contracts for long durations, we are forced to routinely exit our expiring position and enter the next active month.
# Here we isolate the raw discrete number of contracts traded *inclusive* of synthetic rollover transactions versus those exclusively caused by the trend signal flipping.

# %%
# Trace the daily difference in rounded contract positions for a given month label (inclusive of expiring rolls)
actual_trades = (
    actual_positions.groupby(["contract", front.expiry]).diff().fillna(actual_positions)
)

# Trace the strict difference across the continuum regardless of expiration month (exclusive of calendar rolls)
actual_trades_wo_rolls = (
    actual_positions.groupby(["contract"]).diff().fillna(actual_positions)
)

# %%
# Visual tabular breakdown displaying a sample of 20 random active trading days
(
    actual_trades.loc[lambda xs: xs.ne(0)]
    .reorder_levels([1, 0])
    .sort_index()
    .unstack("contract")
    .tail(20)
    .loc[:, lambda xs: xs.notnull().any()]
    .sort_index(axis=1)
    .rename(lambda xs: xs.date())
    .style.format("{:.0f}", na_rep="")
)

# %%
# Total sum of physical contracts traded vs. signal contracts traded
contracts_traded = actual_trades.abs().groupby("date").sum()
contracts_traded_wo_rolls = actual_trades_wo_rolls.abs().groupby("date").sum()

# %%
# Backcalculate our real dollar returns applying the discrete positioning against daily point drifts
actual_returns = (
    actual_positions.mul(front.d_close.loc[shortlist.contract]).groupby("date").sum()
)


# Sharpe Ratios to evaluate the risk-adjusted returns
def sharpe(xs):
    return xs.mean() / xs.std() * 250**0.5


# Adjustment accounts for drag when rounding sizes. We need an appropriate normalization scaling metric to graph apples-to-apples.
adjustment = actual_returns.div(balance).mean() / pnl.mul(risk).mean()

print("Sharpe Comparison:")
sharpe(
    pd.DataFrame(
        {
            "ideal (fractional)": pnl.mul(risk),
            f"ideal x {adjustment:0.2f}": pnl.mul(risk).mul(adjustment),
            "rounded (realworld)": actual_returns.div(balance),
        }
    )[start_year:]
)

# %% [markdown]
# ### Visualizing "Real World" Drag
# Theoretical quant models tend not to account for the physical constraints of an isolated retail \$1,000,000 trading portfolio trying to round fractional $VIX$ contracts.
#
# The log returns tracking plot successfully illustrates how closely our discrete matching adheres to the ideal theoretical benchmark. The overall drift acts extremely similarly but introduces slight execution drag.

# %%
(
    pd.DataFrame(
        {
            "ideal": pnl.mul(risk),
            f"ideal x {adjustment:0.2f}": pnl.mul(risk).mul(adjustment),
            "rounded": actual_returns.div(balance),
        }
    )[start_year:]
    .add(1)
    .cumprod()
    .plot(
        logy=True,
        figsize=(16, 12),
        title="Cumulative Return Trajectories: Ideal vs Rounded",
    )
)

# %%
(
    pd.DataFrame(
        {
            f"ideal x {adjustment:0.2f}": pnl.mul(risk).mul(adjustment),
            "rounded": actual_returns.div(balance),
        }
    )[start_year:]
    .add(1)
    .groupby(pd.Grouper(freq="YE"))
    .prod()
    .sub(1)
    .rename(lambda xs: xs.year)
    .plot.bar(figsize=(16, 12))
)
