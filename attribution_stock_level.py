# Attribution Tool
# Copyright (C) 2025 [Niya Somkerd | nysdev.com]
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go


st.title("Equity Mutual Fund Performance Attribution")

st.info(
    """**What We’ll Cover**  
         - Holdings over time (dynamic weights due to price movement and trading).  
         - Daily performance attribution:
Allocation Effect, Selection Effect, trading Effect (i.e., impact of rebalancing/trading decisions)
        """,
    icon="ℹ️",
)

# =============================================================================
# Step 1: Simulate Data

np.random.seed(42)
dates = pd.date_range(start="2023-12-29", end="2024-12-31", freq="B")  # business days
stocks = ["AAPL", "MSFT", "JNJ", "PFE", "XOM"]

prices = pd.DataFrame(index=dates, columns=stocks)
for stock in stocks:
    mean_daily_return = 0.0005  # ≈ 0.05% per day, or 10% annualized
    daily_volatility = 0.02  # ~20-30% annualized volatility
    prices[stock] = 100 + np.cumprod(
        1 + (np.random.normal(mean_daily_return, daily_volatility, len(dates)))
    )

returns = prices.pct_change().fillna(0)

# =============================================================================
# Simulate Dynamic Fund Holdings (Shares)

# Simulate fund holdings in SHARES, changing weekly
holdings = pd.DataFrame(index=dates, columns=stocks)

# Initial shares
initial_shares = {"AAPL": 100, "MSFT": 100, "JNJ": 80, "PFE": 60, "XOM": 50}

for date in dates:
    if date.day % 7 == 0:
        # Weekly rebalance: simulate small trades
        initial_shares = {
            s: v + np.random.randint(-5, 6) for s, v in initial_shares.items()
        }
    holdings.loc[date] = initial_shares

holdings = holdings.shift(1, axis=0)
holdings = holdings.fillna(method="bfill")  # fill with next row

# =============================================================================
# Step 4: Benchmark Static or Monthly Rebalancing Weights

# Monthly benchmark weights
bm_weights_static = {"AAPL": 0.50, "MSFT": 0.20, "JNJ": 0.20, "PFE": 0.15, "XOM": 0.15}
bm_weights = pd.DataFrame(index=dates, columns=stocks)

for date in dates:
    weights = bm_weights_static.copy()
    bm_weights.loc[date] = weights


# =============================================================================
# Step 3:  Calculate Daily Fund Weights from Holdings

# Portfolio value
portfolio_value = (holdings * prices).sum(axis=1)
portfolio_df = portfolio_value.to_frame(name="portfolio_value")

bm_value = (bm_weights * prices).sum(axis=1)

# Daily fund weights
fund_weights = (holdings * prices).div(portfolio_value, axis=0)

fund_weights = fund_weights.shift(1, axis=0)
fund_weights = fund_weights.fillna(method="bfill")  # fill with next row

portfolio_return_cal = (fund_weights * returns).sum(axis=1)

port_returns = portfolio_return_cal
bm_returns = bm_value.pct_change().fillna(0)


port_returns.name = "port_return"
bm_returns.name = "bm_returns"


result = pd.concat([port_returns, bm_returns], axis=1)
result["excess_return"] = port_returns[0] - bm_returns[0]


# =============================================================================
# Step5: Daily Attribution with Dynamic Weights

# Create attribution DataFrame
attrib = pd.DataFrame(index=dates, columns=["Allocation", "Selection"])

# Loop through each day
for i in range(1, len(dates)):
    date = dates[i]
    prev_date = dates[i - 1]

    # Fund weights today and yesterday
    fw_today = fund_weights.loc[date]
    fw_yest = fund_weights.loc[prev_date]

    # BM weights
    bw_today = bm_weights.loc[date]
    bw_yest = bm_weights.loc[prev_date]

    # BM total
    bw_ret_today = bm_returns.loc[date]

    # Returns today
    r = returns.loc[date]

    # Attribution formulas
    allocation = ((fw_yest - bw_yest) * (r - (bw_ret_today))).sum()
    selection = (bw_yest * (r - r)).sum()

    attrib.loc[date] = [allocation, selection]

attrib = attrib.fillna(0)
attrib["Cumulative_Attribution"] = attrib.sum(axis=1)


# =============================================================================
# Step 6: Plot Attribution

# Cumulative attribution plot
fig, ax = plt.subplots(figsize=(12, 6))
((1 + attrib[["Allocation", "Selection"]]).cumprod() - 1).plot(ax=ax, lw=3)
ax.set_title("Cumulative Attribution")
ax.set_ylabel("Cumulative Effect")
ax.grid(True)

# Display in Streamlit
# st.pyplot(fig)

st.subheader("Cumulative Attribution Effects")

fig = go.Figure()

for col in ["Allocation", "Selection"]:
    fig.add_trace(
        go.Scatter(
            x=attrib.index,
            y=((1 + attrib[col]).cumprod() - 1) * 100,  # convert to percent
            mode="lines",
            name=col,
            hovertemplate="%{y:.4f}%<extra>%{fullData.name}</extra>",
        )
    )

fig.update_layout(
    title="Cumulative Attribution",
    xaxis_title="Date",
    yaxis_title="Cumulative Effect (%)",
    template="plotly_white",
    yaxis_tickformat=".4f",
)

st.plotly_chart(fig, use_container_width=True)


# Convert to percentage terms for readability
attrib_percent = ((1 + attrib[["Allocation", "Selection"]]).cumprod() - 1) * 100

port_returns.name = "Portfolio return"
bm_returns.name = "Benchmark return"

total_port_percent = ((1 + port_returns).cumprod() - 1) * 100
total_bm_percent = ((1 + bm_returns).cumprod() - 1) * 100
total_port_percent = pd.DataFrame(total_port_percent, dtype=float)
total_bm_percent = pd.DataFrame(total_bm_percent, dtype=float)

# Get latest values
latest_date = attrib_percent.index[-1]

allocation_pct = attrib_percent.loc[latest_date, "Allocation"]
selection_pct = attrib_percent.loc[latest_date, "Selection"]
total_pct = allocation_pct + selection_pct

# Get last row as Series
latest_date_port = total_port_percent.iloc[-1]
latest_date_bm = total_bm_percent.iloc[-1]

latest_date_port_num = latest_date_port.values
latest_date_bm_num = latest_date_bm.values

# Format each element as string with 2 decimals and % sign
formatted = latest_date_port.map(lambda x: f"{x:.4f}%")
formatted_bm = latest_date_bm.map(lambda x: f"{x:.4f}%")


# Create dynamic summary text
return_text = ""
for stock, value in formatted.items():
    return_text += f"- **{stock}**: {value}\n"

return_text_bm = ""
for stock, value in formatted_bm.items():
    return_text_bm += f"- **{stock}**: {value}\n"


summary_text = f"""
### 📈 Performance Attribution Summary (as of {latest_date.date()})

{return_text}
{return_text_bm}
- **Excess return:** {(latest_date_port_num - latest_date_bm_num)[0]:.4f}%
- **Allocation effect:** {allocation_pct:.4f}%
- **Selection effect:** {selection_pct:.4f}%

{"✅ Positive contribution" if total_pct > 0 else "⚠️ Negative contribution"} to performance.
"""

st.markdown(summary_text)


st.caption("A caption with _italics_ :blue-badge[colors] and emojis :sunglasses:")


# ======

# Create per-stock attribution DataFrame with MultiIndex columns: ('Allocation', stock), ('Selection', stock)
attrib_stocks = pd.DataFrame(
    index=dates,
    columns=pd.MultiIndex.from_product([["Allocation", "Selection"], stocks]),
)

for i in range(1, len(dates)):
    date = dates[i]
    prev_date = dates[i - 1]

    fw_yest = fund_weights.loc[prev_date]
    bw_yest = bm_weights.loc[prev_date]
    bw_today = bm_weights.loc[date]
    bw_ret_today = bm_returns.loc[date]
    r = returns.loc[date]

    # Calculate per-stock attribution components (vectorized)
    allocation_per_stock = (fw_yest - bw_yest) * r
    selection_per_stock = bw_yest * (r - bw_ret_today)

    # Assign to per-stock attribution DataFrame
    for stock in stocks:
        attrib_stocks.loc[date, ("Allocation", stock)] = allocation_per_stock[stock]
        attrib_stocks.loc[date, ("Selection", stock)] = selection_per_stock[stock]


attrib_stocks = attrib_stocks.fillna(0).astype(float)
attrib_stocks_cum = (1 + attrib_stocks).cumprod() - 1

st.write(attrib_stocks)

# Cross-check total (should match your original sums)
total_alloc = attrib_stocks_cum["Allocation"].sum(axis=1)
total_select = attrib_stocks_cum["Selection"].sum(axis=1)
total_cum = total_alloc + total_select

# ==============================================================================

# Dynamic summary per stock on latest date

latest_date = attrib_stocks_cum.index[-1]

# Get row as DataFrame (not Series)
attrib_percent = attrib_stocks_cum.loc[[latest_date]] * 100

# Apply styling for heatmap
styled_df = (
    attrib_percent.style.format("{:.2f}%")
    .background_gradient(cmap="RdYlGn", axis=None)
    .set_caption(f"Cumulative Attribution (%) as of {latest_date.date()}")
)

st.dataframe(styled_df)

# ------


fig = go.Figure()

for stock in stocks:
    fig.add_trace(
        go.Scatter(
            x=attrib_stocks_cum.index,
            y=attrib_stocks_cum[("Allocation", stock)] * 100,
            mode="lines",
            name=f"{stock} Allocation",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=attrib_stocks_cum.index,
            y=attrib_stocks_cum[("Selection", stock)] * 100,
            mode="lines",
            name=f"{stock} Selection",
        )
    )

fig.update_layout(
    title="Cumulative Attribution per Stock (%)",
    xaxis_title="Date",
    yaxis_title="Cumulative Return (%)",
    template="plotly_white",
)

st.plotly_chart(fig, use_container_width=True)

latest_date = attrib_stocks_cum.index[-1]

summary_lines = [f"### 📊 Attribution Summary by Stock (as of {latest_date.date()})\n"]

for stock in stocks:
    alloc = attrib_stocks_cum.loc[latest_date, ("Allocation", stock)] * 100
    select = attrib_stocks_cum.loc[latest_date, ("Selection", stock)] * 100
    total = alloc + select

    # Interpret allocation effect
    if alloc > 0:
        alloc_desc = f"Allocation was positive, meaning the fund had a good weighting in {stock} relative to the benchmark."
    elif alloc < 0:
        alloc_desc = f"Allocation was negative, likely because the fund was overweight {stock} when its returns were weak or negative."
    else:
        alloc_desc = f"Allocation effect was neutral for {stock}."

    # Interpret selection effect
    if select > 0:
        select_desc = f"Selection was positive, meaning {stock} itself performed better than total bm return."
    elif select < 0:
        select_desc = f"Selection was negative, meaning {stock} underperformed compared to total bm return."
    else:
        select_desc = f"Selection effect was neutral for {stock}."

    # Overall sentiment
    sentiment = "added value 👍" if total > 0 else "detracted value ⚠️"

    summary_lines.append(
        f"- **{stock}**: Total impact {total:.2f}%, {sentiment}\n"
        f"  - {alloc_desc}\n"
        f"  - {select_desc}\n"
    )

summary_text = "\n".join(summary_lines)

st.markdown(summary_text)


# =====

st.subheader("Cumulative Attribution by Stock (%)")

# Convert cumulative attribution to percentages
attrib_pct = attrib_stocks_cum * 100

# Apply heatmap style using pandas styling
styled_df = attrib_pct.style.background_gradient(cmap="RdYlGn", axis=None).format(
    "{:.2f}%"
)

st.dataframe(styled_df)
